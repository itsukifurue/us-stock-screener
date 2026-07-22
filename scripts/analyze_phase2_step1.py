"""Version2 Phase2 Step1: 単変量分析・Version1スコア再分析・完了報告書の生成。

前提: scripts/build_feature_store_phase2.py が実行済みで、data/feature_store.db に
features/labels/candidate_snapshots/universe_membership/build_runs が入っていること。

重要な設計上の制約:
  - bin境界(5分位)はTrain期間のデータだけで決定し、Validationには同じ境界をそのまま適用する
    (全期間を見てbin境界を決めない)。
  - Test期間は対象成績(陽性率・PF・期待値等)を一切集計・表示しない。件数と欠損率のみ扱う。
  - PF/期待値は label.target_trade_pnl_pct(翌日始値エントリー・スリッページ0.2%・手数料0.1%・
    ATR損切り/利確、Version1のbacktest.engineと同一ロジック)の実測値を使う。

実行:
    python scripts/analyze_phase2_step1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

import config
from feature_store.database import FeatureStoreDB

FEATURE_STORE_DB_PATH = config.DATA_DIR / "feature_store.db"

UNIVARIATE_FEATURES = [
    "technical_score_v1", "rsi_14", "atr_pct", "historical_volatility_20d",
    "breakout_close_20d_pct", "volume_ratio_5d", "return_5d",
    "distance_from_20d_high_pct", "bollinger_band_width", "price_vs_ma50_pct",
]
N_BINS = 5
CONDITION_COLUMNS = ["volume_condition", "ma_condition", "ma25_condition", "breakout_condition", "rsi_condition"]


def get_latest_build_run(db: FeatureStoreDB) -> dict:
    row = db.conn.execute("SELECT * FROM build_runs ORDER BY run_at DESC LIMIT 1").fetchone()
    return dict(row) if row else {}


def load_trainval_only(db: FeatureStoreDB, test_start: str) -> pd.DataFrame:
    """Test期間をSQLのWHERE句の段階で除外して読み込む(表示時に隠すのではなく、
    探索処理そのものにTestデータを一切取り込まない)。"""
    rows = db.conn.execute(
        """
        SELECT f.*, l.future_return_1d, l.future_return_3d, l.future_return_5d,
               l.future_return_10d, l.future_return_15d,
               l.future_max_return_5d, l.future_max_return_10d, l.future_max_return_15d,
               l.future_min_return_5d, l.future_min_return_10d, l.future_min_return_15d,
               l.hit_plus_5pct_10d, l.hit_plus_10pct_10d, l.hit_plus_15pct_10d, l.hit_plus_15pct_15d,
               l.hit_stop_atr_1_5_before_plus_15,
               l.days_to_plus_5pct, l.days_to_plus_10pct, l.days_to_plus_15pct,
               l.target_15pct_within_10d, l.target_trade_success, l.target_trade_pnl_pct
        FROM features f
        LEFT JOIN labels l ON f.ticker = l.ticker AND f.signal_date = l.signal_date
        WHERE f.signal_date < :test_start
        ORDER BY f.signal_date, f.ticker
        """,
        {"test_start": test_start},
    ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def count_test_period(db: FeatureStoreDB, test_start: str, test_end: str) -> dict:
    """Test期間は件数・欠損率などの品質情報のみ扱う(目的変数に関する成績は一切集計しない)。"""
    row = db.conn.execute(
        "SELECT COUNT(*) AS n, SUM(CASE WHEN technical_score_v1 IS NULL THEN 1 ELSE 0 END) AS missing_score "
        "FROM features WHERE signal_date >= :s AND signal_date <= :e",
        {"s": test_start, "e": test_end},
    ).fetchone()
    return dict(row)


def compute_pf_ev(pnl_series: pd.Series) -> dict:
    pnl = pnl_series.dropna()
    if len(pnl) == 0:
        return {"n": 0, "pf": None, "ev_pct": None, "win_rate_pct": None}
    wins = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses > 0:
        pf = round(wins / losses, 3)
    elif wins > 0:
        pf = None  # 損失ゼロ(サンプル過少の可能性が高い)。inf表記は誤解を招くためNoneで明示
    else:
        pf = None
    return {
        "n": int(len(pnl)),
        "pf": pf,
        "ev_pct": round(float(pnl.mean()), 3),
        "win_rate_pct": round(float((pnl > 0).mean() * 100), 1),
    }


def fmt_pct(series: pd.Series) -> str:
    s = series.dropna()
    if len(s) == 0:
        return "N/A"
    return f"{round(s.mean() * 100, 1)}%"


def fmt_num(series: pd.Series) -> str:
    s = series.dropna()
    if len(s) == 0:
        return "N/A"
    return f"{round(float(s.mean()), 2)}"


def univariate_analysis(df: pd.DataFrame, train_mask: pd.Series, val_mask: pd.Series) -> list[str]:
    lines = [
        "## 8. 単変量分析(Train期間で決定した5分位、Validationへ同じ境界を適用)",
        "",
        "bin境界はTrain期間の値だけから`pd.qcut`で決定し、Validationにはそのまま適用する"
        "(Validationの分布を見て境界を決め直すことはしない)。",
        "",
    ]
    train_df = df[train_mask]
    val_df = df[val_mask]

    for col in UNIVARIATE_FEATURES:
        if col not in df.columns:
            continue
        train_vals = train_df[col].dropna()
        if len(train_vals) < 50:
            lines.append(f"### {col}\n(Train期間のサンプル数が50未満のためスキップ: {len(train_vals)}件)\n")
            continue
        try:
            _, bin_edges = pd.qcut(train_vals, N_BINS, retbins=True, duplicates="drop")
        except ValueError:
            lines.append(f"### {col}\n(値のばらつきが小さくbin分割不能)\n")
            continue
        bin_edges = np.array(bin_edges, dtype=float)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        lines.append(f"### {col}")
        lines.append("")
        lines.append(
            "| 期間 | 分位区間 | サンプル数 | target_15pct陽性率 | target_success陽性率 | "
            "将来10日リターン平均 | 将来10日リターン中央値 | 将来最大上昇率平均 | 将来最大下落率平均 | "
            "PF(実測) | 期待値(実測%) | 勝率(実測%) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for period_label, sub in [("Train", train_df), ("Validation", val_df)]:
            binned = pd.cut(sub[col], bins=bin_edges)
            for bin_label, group in sub.groupby(binned, observed=True):
                if len(group) == 0:
                    continue
                pf_ev = compute_pf_ev(group["target_trade_pnl_pct"])
                median_10d = group["future_return_10d"].dropna()
                median_val = round(float(median_10d.median()), 2) if len(median_10d) else "N/A"
                lines.append(
                    f"| {period_label} | {bin_label} | {len(group)} | "
                    f"{fmt_pct(group['target_15pct_within_10d'])} | {fmt_pct(group['target_trade_success'])} | "
                    f"{fmt_num(group['future_return_10d'])} | {median_val} | "
                    f"{fmt_num(group['future_max_return_10d'])} | {fmt_num(group['future_min_return_10d'])} | "
                    f"{pf_ev['pf']} | {pf_ev['ev_pct']} | {pf_ev['win_rate_pct']} |"
                )
        lines.append("")
    return lines


def version1_score_analysis(df: pd.DataFrame, trainval_mask: pd.Series) -> list[str]:
    lines = ["## 9. Version1スコアの再分析(Train+Validation期間のみ、Testは対象外)", ""]
    d = df[trainval_mask].copy()

    # --- スコア帯別 ---
    lines.append("### スコア帯別成績")
    lines.append("")
    lines.append("| スコア帯 | サンプル数 | PF | 期待値(%) | 勝率(%) |")
    lines.append("|---|---|---|---|---|")
    bands = [(-1e9, 30, "<30"), (30, 45, "30-45(非シグナル)"), (45, 60, "45-60"), (60, 75, "60-75"), (75, 1e9, "75+")]
    for lo, hi, label in bands:
        sub = d[(d["technical_score_v1"] > lo) & (d["technical_score_v1"] <= hi)]
        pf_ev = compute_pf_ev(sub["target_trade_pnl_pct"])
        lines.append(f"| {label} | {len(sub)} | {pf_ev['pf']} | {pf_ev['ev_pct']} | {pf_ev['win_rate_pct']} |")
    lines.append("")

    # --- 各条件単独の成績(条件を満たした日 vs 満たさなかった日) ---
    lines.append("### 各条件単独の成績(条件を満たした候補日 vs 満たさなかった候補日)")
    lines.append("")
    lines.append("| 条件 | 満たした場合PF | EV(%) | 勝率(%) | n | 満たさない場合PF | EV(%) | 勝率(%) | n |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for cond in CONDITION_COLUMNS:
        if cond not in d.columns:
            continue
        met = d[d[cond] == 1]
        not_met = d[d[cond] == 0]
        pf_met, pf_not = compute_pf_ev(met["target_trade_pnl_pct"]), compute_pf_ev(not_met["target_trade_pnl_pct"])
        lines.append(
            f"| {cond} | {pf_met['pf']} | {pf_met['ev_pct']} | {pf_met['win_rate_pct']} | {pf_met['n']} | "
            f"{pf_not['pf']} | {pf_not['ev_pct']} | {pf_not['win_rate_pct']} | {pf_not['n']} |"
        )
    lines.append("")

    # --- 条件組み合わせ別(満たした条件数) ---
    lines.append("### 条件組み合わせ別成績(満たした条件の数)")
    lines.append("")
    lines.append("| 満たした条件数 | サンプル数 | PF | 期待値(%) | 勝率(%) |")
    lines.append("|---|---|---|---|---|")
    if all(c in d.columns for c in CONDITION_COLUMNS):
        d["_condition_count"] = d[CONDITION_COLUMNS].sum(axis=1)
        for cnt, group in d.groupby("_condition_count"):
            pf_ev = compute_pf_ev(group["target_trade_pnl_pct"])
            lines.append(f"| {int(cnt)} | {len(group)} | {pf_ev['pf']} | {pf_ev['ev_pct']} | {pf_ev['win_rate_pct']} |")
    lines.append("")
    return lines


def same_day_adopted_vs_rejected_analysis(snapshots: pd.DataFrame, labels_df: pd.DataFrame, trainval_dates: set) -> list[str]:
    lines = ["## 9-b. 同一営業日内: 採用された候補 vs 枠不足で見送られた候補", ""]

    snap = snapshots[snapshots["signal_date"].isin(trainval_dates)].copy()
    lab = labels_df[["ticker", "signal_date", "target_trade_pnl_pct", "target_trade_success"]]
    merged = snap.merge(lab, on=["ticker", "signal_date"], how="left")

    adopted = merged[merged["selected_by_v1_flag"] == 1]
    rejected = merged[merged["rejected_due_to_capacity_flag"] == 1]

    lines.append("### 全体集計(単純に全期間の採用トレード vs 全候補ではなく、Version1の採用ルールが")
    lines.append("実際に「枠不足」で弾いた候補だけと比較する)")
    lines.append("")
    lines.append("| 区分 | サンプル数 | PF | 期待値(%) | 勝率(%) |")
    lines.append("|---|---|---|---|---|")
    for label, sub in [("採用された候補", adopted), ("枠不足で見送られた候補", rejected)]:
        pf_ev = compute_pf_ev(sub["target_trade_pnl_pct"])
        lines.append(f"| {label} | {len(sub)} | {pf_ev['pf']} | {pf_ev['ev_pct']} | {pf_ev['win_rate_pct']} |")
    lines.append("")

    # 同一営業日内(signal_date単位)で候補数別に集計
    lines.append("### 候補が3件以下 vs 4件以上競合した日での比較")
    lines.append("")
    qualifying = merged[merged["selected_rank_v1"].notna()]
    day_counts = qualifying.groupby("signal_date")["ticker"].count()
    low_comp_dates = set(day_counts[day_counts <= 3].index)
    high_comp_dates = set(day_counts[day_counts >= 4].index)
    lines.append("| 区分 | 対象日数 | サンプル数(採用) | PF(採用) | EV(%)(採用) | 勝率(%)(採用) |")
    lines.append("|---|---|---|---|---|---|")
    for label, date_set in [("同日候補3件以下", low_comp_dates), ("同日候補4件以上", high_comp_dates)]:
        sub = adopted[adopted["signal_date"].isin(date_set)]
        pf_ev = compute_pf_ev(sub["target_trade_pnl_pct"])
        lines.append(f"| {label} | {len(date_set)} | {pf_ev['n']} | {pf_ev['pf']} | {pf_ev['ev_pct']} | {pf_ev['win_rate_pct']} |")
    lines.append("")

    # 採用順位別
    lines.append("### 採用順位別(その日のスコア順位)の成績(qualifying=スコア>=45の候補のみ対象)")
    lines.append("")
    lines.append("| 順位 | サンプル数 | 採用された割合(%) | PF | EV(%) | 勝率(%) |")
    lines.append("|---|---|---|---|---|---|")
    for rank, group in qualifying.groupby("selected_rank_v1"):
        adopted_rate = round(float((group["selected_by_v1_flag"] == 1).mean() * 100), 1)
        pf_ev = compute_pf_ev(group["target_trade_pnl_pct"])
        lines.append(f"| {int(rank)} | {len(group)} | {adopted_rate} | {pf_ev['pf']} | {pf_ev['ev_pct']} | {pf_ev['win_rate_pct']} |")
    lines.append("")
    return lines


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    db = FeatureStoreDB(FEATURE_STORE_DB_PATH)
    build_run = get_latest_build_run(db)
    train_start, train_end = build_run["train_start"], build_run["train_end"]
    val_start, val_end = build_run["val_start"], build_run["val_end"]
    test_start, test_end = build_run["test_start"], build_run["test_end"]

    # Test期間はSQLのWHERE句の段階で除外する(表示時に隠すのではなく、探索処理そのものに
    # 一切取り込まない)。件数等の品質情報のみ別途カウントする。
    features_labels = load_trainval_only(db, test_start)
    assert features_labels["signal_date"].max() < test_start, "Test期間のレコードが混入しています(監査失敗)"
    test_quality = count_test_period(db, test_start, test_end)

    snapshots_all = pd.DataFrame(db.get_all_candidate_snapshots())
    snapshots = snapshots_all[snapshots_all["signal_date"] < test_start].copy() if not snapshots_all.empty else snapshots_all
    label_count = db.count_labels()
    daily_universe_count = db.count_daily_universe()
    membership = pd.DataFrame(db.conn.execute("SELECT * FROM universe_membership").fetchall(), columns=[
        d[0] for d in db.conn.execute("SELECT * FROM universe_membership LIMIT 1").description
    ]) if db.conn.execute("SELECT COUNT(*) FROM universe_membership").fetchone()[0] else pd.DataFrame()
    db.close()

    train_mask = (features_labels["signal_date"] >= train_start) & (features_labels["signal_date"] < train_end)
    val_mask = (features_labels["signal_date"] >= val_start) & (features_labels["signal_date"] < val_end)
    trainval_mask = train_mask | val_mask
    trainval_dates = set(features_labels.loc[trainval_mask, "signal_date"])

    report_lines: list[str] = []
    report_lines.append(f"# Version2 Phase2 Step1 完了報告 ({pd.Timestamp.today().date().isoformat()})")
    report_lines.append("")
    report_lines.append(f"build_run_id: {build_run.get('build_run_id')}")
    report_lines.append("")

    # 1. ファイル一覧
    report_lines.append("## 1. 作成・変更したファイル")
    report_lines.append("")
    report_lines.append(
        "- `feature_store/schema.sql`(build_runs/universe_membership/daily_universe/"
        "candidate_snapshots追加、labels.target_trade_pnl_pct追加)\n"
        "- `feature_store/database.py`(上記4テーブルのupsert/count/get)\n"
        "- `feature_store/providers.py`(新規、MarketDataProvider/FundamentalDataProvider/"
        "UniverseProvider抽象化)\n"
        "- `feature_store/labels.py`(target_trade_pnl_pct追加)\n"
        "- `backtest/watchlist.py`(PHASE2_WATCHLIST追加、46銘柄)\n"
        "- `scripts/build_feature_store_phase2.py`(新規、Phase2 Step1オーケストレーション)\n"
        "- `scripts/analyze_phase2_step1.py`(新規、本レポート生成スクリプト)"
    )
    report_lines.append("")

    # 2. DBテーブルと件数
    report_lines.append("## 2. DBテーブルと件数")
    report_lines.append("")
    report_lines.append(f"- features: {len(features_labels)}件")
    report_lines.append(f"- labels: {label_count}件")
    report_lines.append(f"- daily_universe: {daily_universe_count}件")
    report_lines.append(f"- candidate_snapshots: {len(snapshots)}件")
    report_lines.append(f"- universe_membership: {len(membership)}件")
    if not membership.empty:
        report_lines.append(f"  - data_fetch_status内訳: {membership['data_fetch_status'].value_counts().to_dict()}")
    report_lines.append("")

    # 3. 対象銘柄と期間
    report_lines.append("## 3. 対象銘柄と期間")
    report_lines.append("")
    report_lines.append(f"- 対象銘柄数: {len(str(build_run.get('symbols','')).split(','))}")
    report_lines.append(f"- 対象銘柄: {build_run.get('symbols')}")
    report_lines.append(f"- 期間: {build_run.get('start_date')} 〜 {build_run.get('end_date')}")
    report_lines.append("")

    # 4. 候補生成ルール
    report_lines.append("## 4. 候補生成ルール")
    report_lines.append("")
    report_lines.append(
        f"- universe_version: {build_run.get('universe_version')}\n"
        f"- 一次スクリーニング条件: price>={config.SCREEN_MIN_PRICE}, "
        f"market_cap>={config.SCREEN_MIN_MARKET_CAP}, avg_volume_20d>={config.SCREEN_MIN_AVG_VOLUME}\n"
        f"- Version1採用ルール: technical_score_v1>={config.BACKTEST_MIN_TECHNICAL_SCORE}でシグナル、"
        f"最大同時保有{config.BACKTEST_MAX_CONCURRENT_POSITIONS}銘柄(枠不足なら見送り)"
    )
    report_lines.append("")

    # 5. Train/Val/Testの日付
    report_lines.append("## 5. Train/Validation/Testの日付境界")
    report_lines.append("")
    report_lines.append(f"- Train: {train_start} 〜 {train_end}")
    report_lines.append(f"- Validation: {val_start} 〜 {val_end}")
    report_lines.append(f"- Test(封印。本レポートでは目的変数に関する成績を一切表示しない): {test_start} 〜 {test_end}")
    report_lines.append(
        f"- Test期間の件数(品質情報のみ。SQLで別カウントしており、上記のfeatures_labels"
        f"DataFrameには最初から含まれていない): {test_quality['n']}件"
        f"(technical_score_v1欠損: {test_quality['missing_score']}件)"
    )
    report_lines.append("")

    # 6. データ品質結果
    report_lines.append("## 6. データ品質結果")
    report_lines.append("")
    report_lines.append("- `reports/feature_store_phase2_quality_*.md` を参照(重複キー・未来日付・"
                         "欠損率3区分・異常値・分割ギャップ等のフルレポート)。")
    report_lines.append("")

    # 7. 生存者バイアス・データ制約
    report_lines.append("## 10. 生存者バイアス等の制約")
    report_lines.append("")
    failed = membership[membership["data_fetch_status"] != "ok"] if not membership.empty else pd.DataFrame()
    report_lines.append(
        "- **生存者バイアス**: yfinanceで取得可能な、現在も上場している銘柄のみを対象にしている。"
        "上場廃止・買収・ティッカー変更した銘柄は含まれておらず、成績は上方バイアスを受けている"
        "可能性が高い。今回、`SQ`(Block Inc、2025年にティッカーをXYZへ変更)がyfinance側で"
        "取得不能(\"possibly delisted\")となった実例が発生しており、ティッカー変更・上場廃止が"
        "実際にデータ欠落を引き起こすことを確認した。\n"
        f"- 取得失敗銘柄数: {len(failed)}件 {list(failed['ticker']) if not failed.empty else ''}\n"
        "- IPO後データしかない銘柄(上場時期の異なる銘柄として意図的に含めた): ARM(2024年9月上場)、"
        "CART(2023年9月上場)、RDDT(2024年3月上場)。これらは対象期間の前半、上場前の日付には"
        "レコードが存在しない(0埋めではなく単純に不在)。\n"
        "- 上場廃止銘柄・point-in-time構成銘柄データは未導入(Phase2 Step1では対応しない、"
        "`feature_store/providers.py`のUniverseProvider/MarketDataProviderを差し替えれば"
        "将来対応可能な設計にしてある)。\n"
        "- セクター・時価総額は現在時点の値を過去に遡って適用しており、真のpoint-in-time属性ではない。"
    )
    report_lines.append("")

    # 8. 単変量分析
    report_lines.extend(univariate_analysis(features_labels, train_mask, val_mask))

    # 9. Version1スコア分析
    report_lines.extend(version1_score_analysis(features_labels, trainval_mask))

    # 9-b. 同日採用対見送り
    if not snapshots.empty:
        report_lines.extend(same_day_adopted_vs_rejected_analysis(snapshots, features_labels, trainval_dates))

    # 11. 未解決事項
    report_lines.append("## 11. Phase 3へ進む前に解決すべき事項")
    report_lines.append("")
    report_lines.append(
        "- `feature_store/market_regime.py`・`feature_store/sector.py`はまだyfinance直接依存"
        "(provider抽象化の対象外)。Provider層を全モジュールへ広げるかはPhase2 Step2着手前に判断する。\n"
        "- 単変量分析・Version1再分析は今回Train+Validationのみ。Testの実績はPhase5まで未確認のまま。\n"
        "- candidate_snapshotsの`selected_rank_v1`は「スコア>=45で信号化された候補」の中の順位であり、"
        "スコア未達候補は`selected_rank_v1=None`(未競合)として区別している。この区別の妥当性は"
        "Phase2 Step2で銘柄数が増えた際に再確認する。\n"
        "- 上場廃止・ティッカー変更銘柄への対応(真の生存者バイアス解消)は依然未着手。\n"
        "- Step2(100-300銘柄)へ進む前に、本レポートの単変量・Version1再分析結果についてユーザー確認を得ること。"
    )
    report_lines.append("")

    report_path = config.REPORTS_DIR / f"phase2_step1_report_{pd.Timestamp.today().date().isoformat()}.md"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"レポート出力: {report_path}")


if __name__ == "__main__":
    main()
