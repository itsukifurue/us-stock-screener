"""Phase3A: ロジスティック回帰ベースライン検証。

固定済みのVersion2 Feature Set v1を使い、単純で解釈可能なロジスティック回帰が
Version1スコア・ランダム順位・陽性率だけのダミーモデルより安定した順位付け能力を
持つかを検証する。目的は最高性能の追求ではなく、リーク有無・最低限の予測力・
Train->Validationでの性能維持・確率校正の確認。

スコープ上の実務判断(レポートに明記する):
- ElasticNet/L1はsolver="saga"を使うが、計算時間の都合上 max_iter=50 とする
  (収束しきらない可能性があるが、本フェーズは診断目的であり最終性能の追求ではないため許容する)。
- ハイパーパラメータのグリッドスクリーニングはwalk-forwardの最初の3fold(全5foldのうち)で行い、
  各モデル系統(L2/L1/ElasticNet)の最良構成を選んだ上で、その構成のみ全5foldへ拡張する
  (全探索した条件と結果は失わずmodel_metricsへ保存する)。
- カテゴリ特徴量ありなしの比較は、選ばれた最良構成についてのみ行う(全グリッドへは適用しない)。
- L1/ElasticNet(saga)のスクリーニングが時間予算(FAMILY_TIME_BUDGET_SECONDS)を超えた場合、
  残り条件を無理に完走させず、理由を記録した上で縮小グリッド
  (C=[0.1,1.0], l1_ratio=[0.25,0.5], class_weight=Noneのみ、数値特徴量のみ)へ切り替える。

監視項目(標準出力へ都度出力、model_metricsへも都度保存):
  各モデルの開始/終了時刻・処理時間、収束警告(ConvergenceWarning)、n_iter_、
  RSSメモリ使用量、foldごとの完了状況、Test期間非読み込みの確認。

実行:
    python -u scripts/run_phase3a_logistic_baseline.py
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import psutil
import sklearn
from sklearn.calibration import _SigmoidCalibration
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import config
from feature_store.database import FeatureStoreDB
from modeling import baselines, evaluate
from modeling.data import ECONOMIC_COLUMN, get_build_run, load_trainval_dataset, train_val_masks
from modeling.preprocessing import build_preprocessor
from modeling.splits import build_walk_forward_folds, split_by_dates

# 個別の警告(データ由来の欠損等)は抑制するが、ConvergenceWarningだけは監視対象として
# 明示的に捕捉する(warnings.catch_warningsで都度捕捉するため、ここではUserWarningのみ抑制)。
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_PROCESS = psutil.Process()


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    mem_mb = _PROCESS.memory_info().rss / 1024 / 1024
    print(f"[{ts}] [mem={mem_mb:.0f}MB] {msg}", flush=True)


FAMILY_TIME_BUDGET_SECONDS = 1200  # L1/ElasticNet(saga)スクリーニングの時間予算(20分)
REDUCED_C_GRID = [0.1, 1.0]
REDUCED_L1_RATIO_GRID = [0.25, 0.5]

FEATURE_STORE_DB_PATH = config.DATA_DIR / "feature_store.db"
FEATURE_SET_VERSION = "v1"
EMBARGO_DAYS = 10
N_FOLDS = 5
N_SCREEN_FOLDS = 3  # グリッドスクリーニングに使うfold数(全5foldのうち先頭3つ)
SAGA_MAX_ITER = 50
LBFGS_MAX_ITER = 300
RANDOM_SEED = 42

C_GRID = [0.01, 0.1, 1.0, 10.0]
L1_RATIO_GRID = [0.25, 0.5, 0.75]
CLASS_WEIGHT_GRID = [None, "balanced"]


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent.parent).decode().strip()
    except Exception:
        return "unknown"


def build_model_grid() -> list[tuple[str, dict]]:
    grid = [("dummy", {})]
    for C in C_GRID:
        for cw in CLASS_WEIGHT_GRID:
            grid.append(("logreg_l2", {"C": C, "class_weight": cw}))
    for C in C_GRID:
        for cw in CLASS_WEIGHT_GRID:
            grid.append(("logreg_l1", {"C": C, "class_weight": cw}))
    for C in C_GRID:
        for l1r in L1_RATIO_GRID:
            for cw in CLASS_WEIGHT_GRID:
                grid.append(("logreg_elasticnet", {"C": C, "l1_ratio": l1r, "class_weight": cw}))
    return grid


def make_estimator(model_type: str, params: dict):
    if model_type == "dummy":
        return DummyClassifier(strategy="prior")
    if model_type == "logreg_l2":
        return LogisticRegression(solver="lbfgs", max_iter=LBFGS_MAX_ITER, C=params["C"], class_weight=params["class_weight"], l1_ratio=0.0)
    if model_type == "logreg_l1":
        return LogisticRegression(solver="saga", max_iter=SAGA_MAX_ITER, C=params["C"], class_weight=params["class_weight"], l1_ratio=1.0)
    if model_type == "logreg_elasticnet":
        return LogisticRegression(solver="saga", max_iter=SAGA_MAX_ITER, C=params["C"], class_weight=params["class_weight"], l1_ratio=params["l1_ratio"])
    raise ValueError(model_type)


def fit_and_eval(pre_cols, cat_cols, include_categorical, model_type, params, X_train, y_train, X_val, y_val, verbose_label: str | None = None):
    """1件のモデル学習・評価を行う。開始/終了時刻・処理時間・収束警告・n_iter_を記録して返す。"""
    pre = build_preprocessor(pre_cols, cat_cols, include_categorical)
    est = make_estimator(model_type, params)
    pipe = Pipeline([("pre", pre), ("clf", est)])

    t0 = time.time()
    started_at = datetime.now().strftime("%H:%M:%S")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipe.fit(X_train, y_train)
    elapsed = round(time.time() - t0, 2)
    finished_at = datetime.now().strftime("%H:%M:%S")

    convergence_warnings = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    n_iter = getattr(pipe.named_steps["clf"], "n_iter_", None)
    n_iter = n_iter.tolist() if hasattr(n_iter, "tolist") else n_iter

    if verbose_label:
        conv_note = f" [ConvergenceWarning x{len(convergence_warnings)}]" if convergence_warnings else ""
        log(f"  fit {verbose_label}: {started_at}->{finished_at} ({elapsed}s) n_train={len(X_train)} n_iter={n_iter}{conv_note}")

    prob = pipe.predict_proba(X_val)[:, 1]
    metrics = evaluate.classification_metrics(y_val.values, prob)
    metrics.update({
        "fit_started_at": started_at, "fit_finished_at": finished_at, "fit_elapsed_seconds": elapsed,
        "n_iter": n_iter, "convergence_warnings": convergence_warnings,
    })
    return pipe, prob, metrics


def _reduced_grid_for_family(model_type: str) -> list[tuple[str, dict]]:
    if model_type == "logreg_l1":
        return [(model_type, {"C": C, "class_weight": None}) for C in REDUCED_C_GRID]
    if model_type == "logreg_elasticnet":
        return [
            (model_type, {"C": C, "l1_ratio": l1r, "class_weight": None})
            for C in REDUCED_C_GRID for l1r in REDUCED_L1_RATIO_GRID
        ]
    return []


def run_grid_screening(df, meta, target_col, folds, screen_folds, db=None, screen_exp_id: str | None = None) -> pd.DataFrame:
    numeric_cols, cat_cols = meta["numeric_cols"], meta["categorical_cols"]
    grid = build_model_grid()

    # familyごとにグループ化して処理する(時間予算の監視・縮小グリッドへの切り替えのため)
    families: dict[str, list[dict]] = {}
    for model_type, params in grid:
        families.setdefault(model_type, []).append(params)

    results = []
    fallback_log: list[str] = []

    for model_type, params_list in families.items():
        family_t0 = time.time()
        family_budget = FAMILY_TIME_BUDGET_SECONDS if model_type in ("logreg_l1", "logreg_elasticnet") else None
        fallback_triggered = False
        i = 0
        while i < len(params_list):
            params = params_list[i]
            elapsed_family = time.time() - family_t0

            if family_budget is not None and not fallback_triggered and elapsed_family > family_budget:
                remaining = params_list[i:]
                reduced = _reduced_grid_for_family(model_type)
                already_done_params = [r["params"] for r in results if r["model_type"] == model_type]
                reduced_remaining = [p for _, p in reduced if p not in already_done_params]
                msg = (
                    f"[FALLBACK] {model_type}: スクリーニング時間が予算({family_budget}秒)を超過"
                    f"(経過{round(elapsed_family)}秒)。残り{len(remaining)}条件を打ち切り、"
                    f"縮小グリッド(C={REDUCED_C_GRID}, l1_ratio={REDUCED_L1_RATIO_GRID if model_type=='logreg_elasticnet' else 'N/A'}, "
                    f"class_weight=Noneのみ)へ切り替える。残り{len(reduced_remaining)}条件を実行する。"
                )
                log(msg)
                fallback_log.append(msg)
                params_list = params_list[:i] + reduced_remaining
                fallback_triggered = True
                continue

            fold_metrics = []
            for fold in folds[:screen_folds]:
                train_df = split_by_dates(df, fold["train_dates"])
                val_df = split_by_dates(df, fold["val_dates"])
                label = f"{model_type}{params} fold{fold['fold']}"
                try:
                    _, _, m = fit_and_eval(numeric_cols, cat_cols, False, model_type, params, train_df, train_df[target_col], val_df, val_df[target_col], verbose_label=label)
                except Exception as e:
                    log(f"  [ERROR] {label}: {e}")
                    m = {"pr_auc": None, "roc_auc": None, "log_loss": None, "error": str(e)}
                m["fold"] = fold["fold"]
                fold_metrics.append(m)
            pr_aucs = [m["pr_auc"] for m in fold_metrics if m.get("pr_auc") is not None]
            mean_pr = round(float(np.mean(pr_aucs)), 4) if pr_aucs else None
            results.append({"model_type": model_type, "params": params, "mean_pr_auc_screen": mean_pr, "fold_metrics": fold_metrics})
            log(f"[screen done] {model_type} {params}: mean_pr_auc({screen_folds}fold)={mean_pr}")

            if db is not None and screen_exp_id is not None:
                db.insert_model_metric(screen_exp_id, "screen_mean", f"pr_auc__{model_type}", mean_pr, json.dumps({"params": params, "fold_metrics": fold_metrics}))

            i += 1

    if fallback_log and db is not None and screen_exp_id is not None:
        db.insert_model_metric(screen_exp_id, "screen_mean", "fallback_events", None, json.dumps(fallback_log))

    return pd.DataFrame(results)


def select_best_per_family(screen_results: pd.DataFrame) -> dict:
    best = {}
    for family in ["logreg_l2", "logreg_l1", "logreg_elasticnet"]:
        sub = screen_results[screen_results["model_type"] == family].dropna(subset=["mean_pr_auc_screen"])
        if sub.empty:
            continue
        row = sub.loc[sub["mean_pr_auc_screen"].idxmax()]
        best[family] = {"model_type": row["model_type"], "params": row["params"]}
    return best


def run_full_walk_forward(df, meta, target_col, folds, model_type, params, name: str = "") -> list[dict]:
    numeric_cols, cat_cols = meta["numeric_cols"], meta["categorical_cols"]
    fold_results = []
    for fold in folds:
        train_df = split_by_dates(df, fold["train_dates"])
        val_df = split_by_dates(df, fold["val_dates"])
        _, prob, m = fit_and_eval(
            numeric_cols, cat_cols, False, model_type, params,
            train_df, train_df[target_col], val_df, val_df[target_col],
            verbose_label=f"[full5fold] {name} fold{fold['fold']} (val {fold['val_range']})",
        )
        m["fold"] = fold["fold"]
        m["val_dates_range"] = fold["val_range"]
        fold_results.append(m)
    return fold_results


def extract_coefficients(pipe: Pipeline) -> list[dict]:
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return []
    names = pre.get_feature_names_out()
    coefs = clf.coef_[0]
    rows = [{"feature_name": n, "coefficient": float(c), "odds_ratio": float(np.exp(c))} for n, c in zip(names, coefs)]
    rows.sort(key=lambda r: -abs(r["coefficient"]))
    for i, r in enumerate(rows, start=1):
        r["abs_rank"] = i
    return rows


def out_of_fold_train_predictions(df, meta, target_col, folds, model_type, params) -> pd.DataFrame:
    """5fold walk-forwardの各foldのval予測を集めて、Train全体の"out-of-fold"予測を作る
    (確率校正のfit用。Validationは一切使わない)。
    """
    numeric_cols, cat_cols = meta["numeric_cols"], meta["categorical_cols"]
    oof_frames = []
    for fold in folds:
        train_df = split_by_dates(df, fold["train_dates"])
        val_df = split_by_dates(df, fold["val_dates"])
        pipe, prob, _ = fit_and_eval(
            numeric_cols, cat_cols, False, model_type, params,
            train_df, train_df[target_col], val_df, val_df[target_col],
        )
        oof_frames.append(pd.DataFrame({"y": val_df[target_col].values, "prob": prob}))
    return pd.concat(oof_frames, ignore_index=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    experiment_batch_id = uuid.uuid4().hex[:8]
    run_at = datetime.now(timezone.utc).isoformat()
    git_commit = git_commit_hash()
    library_versions = json.dumps({
        "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
        "python": platform.python_version(),
    })

    db = FeatureStoreDB(FEATURE_STORE_DB_PATH)
    all_report_sections: dict[str, list[str]] = {}

    for target_col in ["target_trade_success", "target_15pct_within_10d"]:
        print(f"\n{'='*70}\n目的変数: {target_col}\n{'='*70}")
        section_lines = [f"## 目的変数: `{target_col}`", ""]

        df, meta = load_trainval_dataset(db, universe_variant="A_eligible_universe")
        build_run = meta["build_run"]
        train_mask, val_mask = train_val_masks(df, build_run)
        train_df, val_df = df[train_mask].reset_index(drop=True), df[val_mask].reset_index(drop=True)
        numeric_cols, cat_cols = meta["numeric_cols"], meta["categorical_cols"]

        print(f"universe A(eligible_universe): train={len(train_df)} val={len(val_df)}")
        print(f"陽性率: train={train_df[target_col].mean():.4f} val={val_df[target_col].mean():.4f}")
        section_lines.append(
            f"- universe A(eligible_universe): train={len(train_df)}件(陽性率{train_df[target_col].mean():.4f}) / "
            f"val={len(val_df)}件(陽性率{val_df[target_col].mean():.4f})"
        )

        folds = build_walk_forward_folds(df, build_run["train_start"], build_run["train_end"], n_folds=N_FOLDS, embargo_trading_days=EMBARGO_DAYS)
        print(f"walk-forward folds: {len(folds)}件(embargo={EMBARGO_DAYS}営業日)")

        # ---------- 1. グリッドスクリーニング(先頭3fold、結果は都度model_metricsへ段階保存) ----------
        print("\n[グリッドスクリーニング]")
        screen_exp_id = f"phase3a_{experiment_batch_id}_{target_col}_screen"
        db.upsert_model_experiment({
            "experiment_id": screen_exp_id, "run_at": run_at, "git_commit": git_commit,
            "feature_set_version": FEATURE_SET_VERSION, "universe_variant": "A_eligible_universe",
            "target_variable": target_col, "start_date": build_run["start_date"], "end_date": build_run["end_date"],
            "train_start": build_run["train_start"], "train_end": build_run["train_end"],
            "val_start": build_run["val_start"], "val_end": build_run["val_end"],
            "test_start": build_run["test_start"], "test_end": build_run["test_end"],
            "n_train": len(train_df), "n_val": len(val_df), "n_features": len(numeric_cols),
            "feature_whitelist": json.dumps(numeric_cols), "preprocessing_summary": "median_impute+indicator+standardize, numeric_only",
            "model_type": "grid_screen", "model_params": json.dumps({"grid_size": len(build_model_grid())}),
            "embargo_days": EMBARGO_DAYS, "n_folds": N_SCREEN_FOLDS, "random_seed": RANDOM_SEED,
            "library_versions": library_versions, "cooldown_days": 0, "notes": "全探索したハイパーパラメータの一覧(段階保存)",
        })
        screen_results = run_grid_screening(df, meta, target_col, folds, N_SCREEN_FOLDS, db=db, screen_exp_id=screen_exp_id)
        best_per_family = select_best_per_family(screen_results)
        print(f"\n各系統の最良構成: {best_per_family}")

        # ---------- 2. 全5foldでの評価(dummy + 各系統の最良構成) ----------
        print("\n[全5fold walk-forward評価]")
        winners = {"dummy": {}}
        winners.update(best_per_family)
        full_fold_results: dict[str, list[dict]] = {}
        for name, cfg in winners.items():
            model_type = cfg.get("model_type", "dummy")
            params = cfg.get("params", {})
            fr = run_full_walk_forward(df, meta, target_col, folds, model_type, params, name=name)
            full_fold_results[name] = fr
            mean_pr = np.mean([m["pr_auc"] for m in fr if m.get("pr_auc") is not None])
            print(f"  {name}({model_type},{params}): fold別pr_auc={[m['pr_auc'] for m in fr]}, 平均={round(mean_pr,4)}")
        section_lines.append("\n### Train内部walk-forward結果(全5fold、embargo=10営業日)\n")
        section_lines.append("| モデル | fold1 | fold2 | fold3 | fold4 | fold5 | 平均PR-AUC |")
        section_lines.append("|---|---|---|---|---|---|---|")
        for name, fr in full_fold_results.items():
            pr_list = [m["pr_auc"] for m in fr]
            mean_pr = round(float(np.mean([p for p in pr_list if p is not None])), 4)
            section_lines.append(f"| {name} | " + " | ".join(str(p) for p in pr_list) + f" | {mean_pr} |")

        # ---------- 3. 最終Train->Validation評価(numeric-only) ----------
        print("\n[最終Train->Validation評価]")
        final_models = {}
        for name, cfg in winners.items():
            model_type = cfg.get("model_type", "dummy")
            params = cfg.get("params", {})
            pipe, prob, m = fit_and_eval(numeric_cols, cat_cols, False, model_type, params, train_df, train_df[target_col], val_df, val_df[target_col], verbose_label=f"[final train->val] {name}")
            final_models[name] = {"pipe": pipe, "prob": prob, "metrics": m, "model_type": model_type, "params": params}
            print(f"  {name}: {m}")

        overall_best_name = max(
            (n for n in final_models if n != "dummy"),
            key=lambda n: (final_models[n]["metrics"].get("pr_auc") or -1),
        )
        best = final_models[overall_best_name]
        print(f"\n総合最良構成: {overall_best_name} ({best['model_type']}, {best['params']})")

        section_lines.append("\n### Validation最終評価(numeric-only)\n")
        section_lines.append("| モデル | ROC-AUC | PR-AUC | LogLoss | Brier | Precision | Recall | F1 | 陽性率 | 予測確率平均 |")
        section_lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for name, fm in final_models.items():
            m = fm["metrics"]
            section_lines.append(
                f"| {name} | {m['roc_auc']} | {m['pr_auc']} | {m['log_loss']} | {m['brier']} | "
                f"{m['precision']} | {m['recall']} | {m['f1']} | {m['positive_rate']} | {m['mean_predicted_prob']} |"
            )
        section_lines.append(f"\n総合最良構成(以降の詳細分析対象): **{overall_best_name}** ({best['model_type']}, {best['params']})\n")

        experiment_id = f"phase3a_{experiment_batch_id}_{target_col}_best_{overall_best_name}"
        db.upsert_model_experiment({
            "experiment_id": experiment_id, "run_at": run_at, "git_commit": git_commit,
            "feature_set_version": FEATURE_SET_VERSION, "universe_variant": "A_eligible_universe",
            "target_variable": target_col, "start_date": build_run["start_date"], "end_date": build_run["end_date"],
            "train_start": build_run["train_start"], "train_end": build_run["train_end"],
            "val_start": build_run["val_start"], "val_end": build_run["val_end"],
            "test_start": build_run["test_start"], "test_end": build_run["test_end"],
            "n_train": len(train_df), "n_val": len(val_df), "n_features": len(numeric_cols),
            "feature_whitelist": json.dumps(numeric_cols), "preprocessing_summary": "median_impute+indicator+standardize, numeric_only",
            "model_type": best["model_type"], "model_params": json.dumps(best["params"]),
            "embargo_days": EMBARGO_DAYS, "n_folds": N_FOLDS, "random_seed": RANDOM_SEED,
            "library_versions": library_versions, "cooldown_days": 0, "notes": f"総合最良構成({overall_best_name})",
        })
        for k, v in best["metrics"].items():
            if v is not None:
                db.insert_model_metric(experiment_id, "validation", k, v if isinstance(v, (int, float)) else None, json.dumps(v) if not isinstance(v, (int, float)) else None)

        # ---------- 4. カテゴリ特徴量あり比較(最良構成のみ) ----------
        print("\n[カテゴリ特徴量あり/なし比較(最良構成)]")
        pipe_cat, prob_cat, m_cat = fit_and_eval(numeric_cols, cat_cols, True, best["model_type"], best["params"], train_df, train_df[target_col], val_df, val_df[target_col])
        print(f"  numeric_only: {best['metrics']}")
        print(f"  numeric+categorical: {m_cat}")
        section_lines.append(
            f"\n### カテゴリ特徴量あり/なし比較(最良構成 {overall_best_name})\n\n"
            f"| 構成 | ROC-AUC | PR-AUC | LogLoss | Brier |\n|---|---|---|---|---|\n"
            f"| numeric_only | {best['metrics']['roc_auc']} | {best['metrics']['pr_auc']} | {best['metrics']['log_loss']} | {best['metrics']['brier']} |\n"
            f"| numeric+categorical | {m_cat['roc_auc']} | {m_cat['pr_auc']} | {m_cat['log_loss']} | {m_cat['brier']} |\n"
        )

        # ---------- 5. 係数・オッズ比・fold間符号安定性 ----------
        print("\n[係数・オッズ比・符号安定性]")
        coef_final = extract_coefficients(best["pipe"])
        for row in coef_final[:15]:
            db.insert_model_coefficient(experiment_id, "validation", row["feature_name"], row["coefficient"], row["odds_ratio"], row["abs_rank"])

        # fold毎の係数を計算し符号の安定性を見る
        fold_signs: dict[str, list[int]] = {}
        for fold in folds:
            train_fold_df = split_by_dates(df, fold["train_dates"])
            pre = build_preprocessor(numeric_cols, cat_cols, False)
            est = make_estimator(best["model_type"], best["params"])
            pipe_f = Pipeline([("pre", pre), ("clf", est)])
            pipe_f.fit(train_fold_df, train_fold_df[target_col])
            coefs_f = extract_coefficients(pipe_f)
            for row in coefs_f:
                fold_signs.setdefault(row["feature_name"], []).append(1 if row["coefficient"] > 0 else (-1 if row["coefficient"] < 0 else 0))

        unstable_features = []
        for feat, signs in fold_signs.items():
            if len(set(s for s in signs if s != 0)) > 1:
                unstable_features.append(feat)

        section_lines.append(f"\n### 係数・オッズ比(上位15、絶対値順、{overall_best_name})\n")
        section_lines.append("| 特徴量 | 係数 | オッズ比 | fold間符号安定性 |")
        section_lines.append("|---|---|---|---|")
        for row in coef_final[:15]:
            signs = fold_signs.get(row["feature_name"], [])
            stability = "不安定(符号反転あり)" if row["feature_name"] in unstable_features else "安定"
            section_lines.append(f"| {row['feature_name']} | {row['coefficient']:.4f} | {row['odds_ratio']:.4f} | {stability} |")
        section_lines.append(f"\nfold間で符号が反転した特徴量数: {len(unstable_features)} / {len(fold_signs)}")
        # technical_score_v1の係数を明示
        ts_rows = [r for r in coef_final if "technical_score_v1" in r["feature_name"]]
        if ts_rows:
            section_lines.append(f"\ntechnical_score_v1の係数: {ts_rows[0]}")

        # ---------- 6. Permutation Importance(Validation、PR-AUC/LogLoss中心) ----------
        print("\n[Permutation Importance]")
        # 重要: permutation_importanceへ渡すDataFrameは、ColumnTransformerが実際に参照する
        # numeric_cols+categorical_colsだけに厳密に絞る(かつその順序で列インデックスを揃える)。
        # 元のval_df全体(ticker/signal_date/target列等を含む)を渡すと、importances_mean配列の
        # インデックスとnumeric_colsの対応がずれてしまうバグになるため注意。
        perm_input_cols = numeric_cols + cat_cols
        val_df_perm = val_df[perm_input_cols]
        perm = permutation_importance(
            best["pipe"], val_df_perm, val_df[target_col], scoring=["average_precision", "neg_log_loss"],
            n_repeats=10, random_state=RANDOM_SEED, n_jobs=1,
        )
        perm_pr = perm["average_precision"]
        perm_rows = sorted(
            [{"feature": c, "importance_mean": perm_pr.importances_mean[i], "importance_std": perm_pr.importances_std[i]}
             for i, c in enumerate(perm_input_cols)],
            key=lambda r: -r["importance_mean"],
        )
        section_lines.append("\n### Permutation Importance(PR-AUC、上位15、Validation)\n")
        section_lines.append("| 特徴量 | importance平均 | importance標準偏差 |")
        section_lines.append("|---|---|---|")
        for row in perm_rows[:15]:
            section_lines.append(f"| {row['feature']} | {row['importance_mean']:.5f} | {row['importance_std']:.5f} |")

        # ---------- 7. 確率校正 ----------
        print("\n[確率校正]")
        oof = out_of_fold_train_predictions(df, meta, target_col, folds, best["model_type"], best["params"])
        platt = _SigmoidCalibration().fit(oof["prob"].values, oof["y"].values)
        iso = IsotonicRegression(out_of_bounds="clip").fit(oof["prob"].values, oof["y"].values)
        prob_platt = platt.predict(best["prob"])
        prob_iso = iso.predict(best["prob"])
        y_val_arr = val_df[target_col].values
        cal_uncal = evaluate.classification_metrics(y_val_arr, best["prob"])
        cal_platt = evaluate.classification_metrics(y_val_arr, prob_platt)
        cal_iso = evaluate.classification_metrics(y_val_arr, prob_iso)
        cal_table_uncal = evaluate.calibration_table(y_val_arr, best["prob"])
        ece_uncal = evaluate.expected_calibration_error(cal_table_uncal, len(y_val_arr))
        cal_table_iso = evaluate.calibration_table(y_val_arr, prob_iso)
        ece_iso = evaluate.expected_calibration_error(cal_table_iso, len(y_val_arr))
        print(f"  未校正: brier={cal_uncal['brier']} ece={ece_uncal}")
        print(f"  Platt: brier={cal_platt['brier']}")
        print(f"  Isotonic: brier={cal_iso['brier']} ece={ece_iso}")
        section_lines.append(
            f"\n### 確率校正比較(Trainのout-of-fold予測でfit、Validationで評価)\n\n"
            f"| 校正方式 | Brier | LogLoss | ECE |\n|---|---|---|---|\n"
            f"| 未校正 | {cal_uncal['brier']} | {cal_uncal['log_loss']} | {ece_uncal} |\n"
            f"| Platt | {cal_platt['brier']} | {cal_platt['log_loss']} | N/A |\n"
            f"| Isotonic | {cal_iso['brier']} | {cal_iso['log_loss']} | {ece_iso} |\n"
        )
        section_lines.append("\n#### 予測確率decile別 実際の陽性率(未校正、Validation)\n")
        section_lines.append("| bin | n | 予測確率平均 | 実際の陽性率 |")
        section_lines.append("|---|---|---|---|")
        for row in cal_table_uncal:
            section_lines.append(f"| {row['bin']} | {row['n']} | {row['mean_predicted']} | {row['actual_positive_rate']} |")

        # ---------- 8. 上位K%・日次上位N銘柄の経済成績 ----------
        print("\n[上位K%・日次上位N銘柄の経済成績]")
        val_df_eval = val_df.copy()
        val_df_eval["_model_prob"] = best["prob"]
        section_lines.append("\n### 上位K%の経済成績(Validation、モデル予測確率順)\n")
        section_lines.append("| K% | n | target陽性率 | pnl平均 | pnl中央値 | PF |")
        section_lines.append("|---|---|---|---|---|---|")
        for k in [0.01, 0.03, 0.05, 0.10]:
            s = evaluate.topk_economic_summary(val_df_eval, "_model_prob", target_col, ECONOMIC_COLUMN, k)
            section_lines.append(f"| {int(k*100)}% | {s['n']} | {s['target_positive_rate']} | {s['pnl_mean']} | {s['pnl_median']} | {s['pf']} |")

        section_lines.append("\n### 日次上位N銘柄の経済成績(Validation、モデル予測確率順)\n")
        section_lines.append("| N | 選抜数合計 | target陽性率 | pnl平均 | pnl中央値 | PF |")
        section_lines.append("|---|---|---|---|---|---|")
        for n in [1, 3, 5]:
            s = evaluate.daily_topn_economic_summary(val_df_eval, "_model_prob", target_col, ECONOMIC_COLUMN, n)
            section_lines.append(f"| {n} | {s['n_total_picks']} | {s['target_positive_rate']} | {s['pnl_mean']} | {s['pnl_median']} | {s['pf']} |")

        # ---------- 9. ベースライン比較(Validation) ----------
        print("\n[ベースライン比較]")
        section_lines.append("\n### ベースライン比較(Validation、上位5%の経済成績)\n")
        section_lines.append("| ベースライン | n | target陽性率 | pnl平均 | PF |")
        section_lines.append("|---|---|---|---|---|")
        baseline_scores = {
            "モデル予測確率": val_df_eval["_model_prob"],
            "全体陽性率(参考)": None,
            "technical_score_v1降順": baselines.score_rank_desc(val_df_eval),
            "technical_score_v1昇順": baselines.score_rank_asc(val_df_eval),
            "return_5d降順": baselines.return_5d_rank_desc(val_df_eval),
            "breakout_close_20d_pct降順": baselines.breakout_rank_desc(val_df_eval),
        }
        for name, score in baseline_scores.items():
            if score is None:
                section_lines.append(f"| {name} | {len(val_df_eval)} | {val_df_eval[target_col].mean():.4f} | - | - |")
                continue
            tmp = val_df_eval.copy()
            tmp["_score"] = score
            s = evaluate.topk_economic_summary(tmp, "_score", target_col, ECONOMIC_COLUMN, 0.05)
            section_lines.append(f"| {name} | {s['n']} | {s['target_positive_rate']} | {s['pnl_mean']} | {s['pf']} |")

        # ランダム順位1000回
        rand_dist = baselines.random_rank_topk_distribution(val_df_eval, target_col, ECONOMIC_COLUMN, k_frac=0.05, n_trials=1000, seed0=RANDOM_SEED)
        model_topk = evaluate.topk_economic_summary(val_df_eval, "_model_prob", target_col, ECONOMIC_COLUMN, 0.05)
        model_pos_rate_percentile = baselines.percentile_rank_of_value(
            model_topk["target_positive_rate"], rand_dist["pos_rates_raw"],
        )
        print(f"  ランダム上位5%分布: median={rand_dist['pos_rate_median']} p5={rand_dist['pos_rate_p5']} p95={rand_dist['pos_rate_p95']}")
        print(f"  モデルの上位5%陽性率パーセンタイル(ランダム分布内): {model_pos_rate_percentile}")
        section_lines.append(
            f"\n### ランダム順位1000回試行との比較(上位5%、Validation)\n\n"
            f"モデルの上位5% target陽性率: {model_topk['target_positive_rate']}(ランダム分布内で"
            f"{model_pos_rate_percentile}パーセンタイルに位置) / "
            f"ランダム分布: median={round(rand_dist['pos_rate_median'],4)} "
            f"p5={round(rand_dist['pos_rate_p5'],4)} p95={round(rand_dist['pos_rate_p95'],4)}\n"
        )

        # ---------- 10. cooldown比較 ----------
        print("\n[cooldown比較]")
        section_lines.append("\n### cooldown適用時の変化(同一構成、Validation)\n")
        section_lines.append("| cooldown | train n | val n | PR-AUC | 上位5%陽性率 |")
        section_lines.append("|---|---|---|---|---|")
        for cd in [0, 5, 10]:
            df_cd, meta_cd = load_trainval_dataset(db, universe_variant="A_eligible_universe", cooldown_days=cd)
            tmask_cd, vmask_cd = train_val_masks(df_cd, meta_cd["build_run"])
            tr_cd, va_cd = df_cd[tmask_cd], df_cd[vmask_cd]
            if len(tr_cd) < 100 or len(va_cd) < 20:
                continue
            _, prob_cd, m_cd = fit_and_eval(numeric_cols, cat_cols, False, best["model_type"], best["params"], tr_cd, tr_cd[target_col], va_cd, va_cd[target_col])
            va_cd_eval = va_cd.copy()
            va_cd_eval["_model_prob"] = prob_cd
            s_cd = evaluate.topk_economic_summary(va_cd_eval, "_model_prob", target_col, ECONOMIC_COLUMN, 0.05)
            print(f"  cooldown={cd}: n_train={len(tr_cd)} n_val={len(va_cd)} pr_auc={m_cd['pr_auc']} top5%pos={s_cd['target_positive_rate']}")
            section_lines.append(f"| {cd}営業日 | {len(tr_cd)} | {len(va_cd)} | {m_cd['pr_auc']} | {s_cd['target_positive_rate']} |")

        # ---------- 11. universe B(signal_v1_flag=1サブセット)比較 ----------
        print("\n[universe B(signal_v1_flag=1)比較]")
        df_b, meta_b = load_trainval_dataset(db, universe_variant="B_signal_v1_subset")
        tmask_b, vmask_b = train_val_masks(df_b, meta_b["build_run"])
        tr_b, va_b = df_b[tmask_b], df_b[vmask_b]
        section_lines.append("\n### universe B(signal_v1_flag=1サブセット)比較\n")
        if len(tr_b) > 100 and len(va_b) > 20:
            _, prob_b, m_b = fit_and_eval(numeric_cols, cat_cols, False, best["model_type"], best["params"], tr_b, tr_b[target_col], va_b, va_b[target_col])
            print(f"  B: n_train={len(tr_b)} n_val={len(va_b)} {m_b}")
            section_lines.append(
                f"train={len(tr_b)}件 val={len(va_b)}件 (陽性率train={tr_b[target_col].mean():.4f} val={va_b[target_col].mean():.4f})\n\n"
                f"| 指標 | A(eligible全体) | B(signal_v1のみ) |\n|---|---|---|\n"
                f"| PR-AUC | {best['metrics']['pr_auc']} | {m_b['pr_auc']} |\n"
                f"| ROC-AUC | {best['metrics']['roc_auc']} | {m_b['roc_auc']} |\n"
                f"| Brier | {best['metrics']['brier']} | {m_b['brier']} |\n"
            )
        else:
            section_lines.append("(サンプル不足のためスキップ)")

        all_report_sections[target_col] = section_lines

        # Test封印の自己検証
        assert df["signal_date"].max() < meta["test_start"], "Test期間が学習データに混入しています"
        assert val_df["signal_date"].max() < meta["test_start"], "Validationの最大日がtest_start以上です"
        print(f"\n[Test封印チェック] OK (max date {df['signal_date'].max()} < test_start {meta['test_start']})")

    db.close()

    report_path = config.REPORTS_DIR / f"phase3a_logistic_baseline_{pd.Timestamp.today().date().isoformat()}.md"
    lines = [f"# Phase3A ロジスティック回帰ベースライン検証 ({pd.Timestamp.today().date().isoformat()})", "", f"experiment_batch_id: {experiment_batch_id}", f"git_commit: {git_commit}", ""]
    for target_col, sec in all_report_sections.items():
        lines.extend(sec)
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nレポート出力: {report_path}")


if __name__ == "__main__":
    main()
