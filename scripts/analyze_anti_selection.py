"""Phase2 Step1監査: 逆選択(スコア順採用が同日の見送り候補より悪い結果を選ぶ現象)の原因分析。

【Test期間の扱い(監査要件)】
このスクリプトは、data/feature_store.db の build_runs テーブルから test_start を読み取り、
features/labelsの読み込みSQL自体に `WHERE signal_date < :test_start` を入れることで、
Test期間の行を「探索処理そのものに一切取り込まない」(表示時に隠すのではない)。
価格フレーム(ポートフォリオNAVシミュレーション用)もtest_start未満の日付に切り詰める。
スクリプト末尾で「読み込んだ全データの最大signal_dateがtest_start未満であること」を
assertで検証し、ログに明示する。

【この監査で使う候補集合の定義(重要な限定事項)】
ランキング方式比較(方式3・9のランダム抽選や、シナリオA/B/C)は、いずれも
「technical_score_v1 >= 45 で信号化された候補(=backtest.engine.simulate_symbolが
生成するトレード)」の集合を対象とする。真のuniverse candidate全体(スコア未達を含む)
まで対象を広げた抽選・配分シミュレーションは、トレード価格付けインフラの追加实装が
必要なため今回のStep1監査では実施していない(未解決事項として報告する)。

実行:
    python scripts/analyze_anti_selection.py
"""
from __future__ import annotations

import pickle
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

import config
from backtest.engine import simulate_symbol
from backtest.metrics import compute_detailed_stats, compute_portfolio_metrics
from backtest.portfolio import simulate_portfolio
from feature_store.database import FeatureStoreDB

FEATURE_STORE_DB_PATH = config.DATA_DIR / "feature_store.db"
CACHE_PATH = config.DATA_DIR / "phase2_price_histories.pkl"
MIN_ROWS = 200
N_RANDOM_TRIALS = 1000

RANK_FEATURES = [
    "technical_score_v1", "volume_ratio_5d", "breakout_close_20d_pct", "rsi_14", "atr_pct",
    "price_vs_ma25_pct", "return_1d", "return_5d", "return_10d", "gap_pct", "dollar_volume",
    "market_cap",
]


def load_build_context() -> tuple[FeatureStoreDB, dict]:
    db = FeatureStoreDB(FEATURE_STORE_DB_PATH)
    row = db.conn.execute("SELECT * FROM build_runs ORDER BY run_at DESC LIMIT 1").fetchone()
    return db, dict(row)


def load_trainval_features(db: FeatureStoreDB, test_start: str) -> pd.DataFrame:
    """features+labelsをTest開始日より前だけSQLの段階で絞り込んで読み込む(構造的除外)。"""
    rows = db.conn.execute(
        """
        SELECT f.ticker, f.signal_date, f.technical_score_v1, f.volume_ratio_5d,
               f.breakout_close_20d_pct, f.rsi_14, f.atr_pct, f.price_vs_ma25_pct,
               f.return_1d, f.return_5d, f.return_10d, f.gap_pct, f.dollar_volume,
               f.market_cap, f.sector, f.market_regime, f.candidate_flag,
               l.target_trade_pnl_pct, l.target_trade_success, l.target_15pct_within_10d
        FROM features f
        LEFT JOIN labels l ON f.ticker = l.ticker AND f.signal_date = l.signal_date
        WHERE f.signal_date < :test_start
        ORDER BY f.signal_date, f.ticker
        """,
        {"test_start": test_start},
    ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def load_histories() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        print(f"価格履歴キャッシュを使用: {CACHE_PATH} ({len(cache['histories'])}銘柄)")
        return cache["histories"]
    raise FileNotFoundError(
        f"{CACHE_PATH} が見つかりません。先に scripts/build_feature_store_phase2.py を実行してください。"
    )


def build_trades_trainval(histories: dict, start_date: str, test_start: str, feat_lookup: dict) -> list[dict]:
    """simulate_symbolで全トレードを生成し、signal_dateが[start_date, test_start)の範囲に
    入るものだけを残す(Test期間のトレードは生成後すぐに捨て、以降の処理に一切混入させない)。
    各トレードにランキング用の特徴量(atr_pct等)をfeat_lookupから付与する。
    """
    trades_all: list[dict] = []
    for sym, h in histories.items():
        trades = simulate_symbol(
            sym, h,
            min_technical_score=config.BACKTEST_MIN_TECHNICAL_SCORE,
            max_holding_days=config.BACKTEST_MAX_HOLDING_DAYS,
            min_rows=MIN_ROWS,
            entry_mode="next_open",
            slippage_pct=config.BACKTEST_SLIPPAGE_PCT,
            commission_pct=config.BACKTEST_COMMISSION_PCT,
        )
        for t in trades:
            if not (start_date <= t["signal_date"] < test_start):
                continue
            key = (t["symbol"], t["signal_date"])
            feat = feat_lookup.get(key)
            if feat is None:
                continue  # featuresテーブル側に無い(範囲外)場合は除外
            for col in RANK_FEATURES:
                t[f"_{col}"] = feat.get(col)
            trades_all.append(t)
    return trades_all


def build_price_frames_trainval(histories: dict, test_start: str) -> dict[str, pd.DataFrame]:
    frames = {}
    for sym, h in histories.items():
        df = pd.DataFrame(h)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df = df[df["date"] < pd.Timestamp(test_start)]  # Test期間は物理的に切り落とす(バッファなし)
        frames[sym] = df[["date", "close"]]
    return frames


def cohens_d(a: pd.Series, b: pd.Series) -> float | None:
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return None
    na, nb = len(a), len(b)
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if pooled_var <= 0:
        return None
    return round((a.mean() - b.mean()) / (pooled_var ** 0.5), 3)


def describe_group(s: pd.Series) -> dict:
    s = s.dropna()
    if len(s) == 0:
        return {"n": 0, "mean": None, "median": None, "q1": None, "q3": None}
    return {
        "n": len(s), "mean": round(float(s.mean()), 3), "median": round(float(s.median()), 3),
        "q1": round(float(s.quantile(0.25)), 3), "q3": round(float(s.quantile(0.75)), 3),
    }


def simulate_capacity_only(trades: list[dict], max_concurrent: int = 3) -> list[dict]:
    """3枠の空き状況だけを考慮し、資金制約は一切考慮しない admission simulation
    (シナリオA: 現在のVersion1採用ルール=スコア降順を使い、資金は無限とみなす)。"""
    events = sorted(set(t["entry_date"] for t in trades) | set(t["exit_date"] for t in trades))
    trades_by_entry: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        trades_by_entry[t["entry_date"]].append(t)
    slots: list[dict | None] = [None] * max_concurrent
    admitted: list[dict] = []
    for day in events:
        for i in range(max_concurrent):
            if slots[i] is not None and slots[i]["exit_date"] == day:
                slots[i] = None
        day_trades = sorted(trades_by_entry.get(day, []), key=lambda t: -(t.get("signal_score") or 0))
        for t in day_trades:
            free = next((i for i in range(max_concurrent) if slots[i] is None), None)
            if free is None:
                continue
            slots[free] = t
            admitted.append(t)
            if t["exit_date"] == day:
                slots[free] = None
    return admitted


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    db, build_run = load_build_context()
    train_start, train_end = build_run["train_start"], build_run["train_end"]
    val_start, val_end = build_run["val_start"], build_run["val_end"]
    test_start = build_run["test_start"]
    start_date = build_run["start_date"]

    print(f"build_run_id: {build_run['build_run_id']}")
    print(f"Train: {train_start}〜{train_end} / Validation: {val_start}〜{val_end} / Test(封印): {test_start}〜")

    feat_df = load_trainval_features(db, test_start)
    db.close()
    assert feat_df["signal_date"].max() < test_start, "Test期間のレコードが混入しています(監査失敗)"
    print(f"[Test封印チェック] features/labels読み込み件数: {len(feat_df)}件、最大signal_date: {feat_df['signal_date'].max()} (< {test_start} を確認)")

    feat_lookup = {(r["ticker"], r["signal_date"]): r for r in feat_df.to_dict("records")}

    histories = load_histories()
    trades_all = build_trades_trainval(histories, start_date, test_start, feat_lookup)
    assert all(t["signal_date"] < test_start for t in trades_all), "tradesにTest期間が混入しています(監査失敗)"
    assert all(t["entry_date"] <= test_start for t in trades_all), "エントリー日がTest期間に及ぶトレードがあります"
    print(f"[Test封印チェック] trades件数: {len(trades_all)}件、シグナル最大日: {max(t['signal_date'] for t in trades_all)}")

    price_frames = build_price_frames_trainval(histories, test_start)
    for sym, df in price_frames.items():
        assert df.empty or df["date"].max() < pd.Timestamp(test_start), f"{sym}の価格フレームにTest期間が混入"
    print("[Test封印チェック] price_frames: 全銘柄でTest開始日未満のみであることを確認済み")

    n_exit_past_test = sum(1 for t in trades_all if t["exit_date"] >= test_start)
    print(
        f"[透明性の注記] signal_date(<{test_start})はTrain/Validationのみだが、"
        f"うち{n_exit_past_test}件はexit_date(最大保有{config.BACKTEST_MAX_HOLDING_DAYS}営業日後)"
        f"がTest開始日以降になる。これは本プロジェクトのラベル設計(LABEL_HORIZON_BUFFER_DAYS等)と"
        "同じ考え方で、シグナル自体はTrain/Valの正当なデータ点であり、その結果確定に数日〜数週間の"
        "将来データが必要になるのは一般的なラベル設計である。ただし、Test期間の候補自体の成績評価には"
        "一切使っていない(price_framesはtest_start未満に切り詰めているため、この一部のトレードの"
        "ポートフォリオNAVへの反映もtest_start以降は行われない=保有中スロットが凍結される軽微な近似)。"
    )

    # =====================================================================
    # 件数整合性の監査
    # =====================================================================
    print("\n=== 件数整合性 ===")
    total_full_period_features = feat_df  # 実はすでにTrain+Valのみ(この時点でTestは含まれない)
    print(f"features(Train+Valのみ、Test除外): {len(feat_df)}件")
    print(f"うちcandidate_flag=1: {int((feat_df['candidate_flag'] == 1).sum())}件")
    print(f"シグナル(score>=45、Train+Valのみ): {len(trades_all)}件")

    portfolio_result = simulate_portfolio(
        trades_all, price_frames, max_concurrent=config.BACKTEST_MAX_CONCURRENT_POSITIONS,
        cash_reuse_timing="next_day",
    )
    n_no_slot = sum(1 for s in portfolio_result["skipped_trades"] if s["reason"] == "no_slot")
    n_cash = sum(1 for s in portfolio_result["skipped_trades"] if s["reason"] == "cash_insufficient")
    print(
        f"採用(Train+Valのみ): {portfolio_result['trades_taken']} / "
        f"見送り(枠不足): {n_no_slot} / 見送り(現金不足): {n_cash}"
    )
    print(
        "(注記: 以前報告した428/1320/(枠不足の全体像)は、全期間559/1559から"
        "Test期間分を除いた数値。ここでは読み込み時点からTestを除外したデータで"
        "同じシミュレーションを再実行しており、上記が正となる)"
    )

    admitted_keys = {(t["symbol"], t["signal_date"]) for t in portfolio_result["admitted_trades"]}
    skip_reason = {(s["symbol"], s["signal_date"]): s["reason"] for s in portfolio_result["skipped_trades"]}

    for t in trades_all:
        key = (t["symbol"], t["signal_date"])
        if key in admitted_keys:
            t["_status"] = "admitted"
        elif key in skip_reason:
            t["_status"] = skip_reason[key]  # "no_slot" / "cash_insufficient"
        else:
            t["_status"] = "unknown"

    unknown_count = sum(1 for t in trades_all if t["_status"] == "unknown")
    print(f"(状態不明トレード: {unknown_count}件。0でなければsimulate_portfolioとの突き合わせに漏れがある)")

    # =====================================================================
    # シナリオA/B/C: 採用ルールと資金制約の寄与分離
    # =====================================================================
    print("\n=== シナリオA/B/C ===")
    scenario_c_stats = compute_detailed_stats(trades_all)  # 全シグナルを制約なしで採用(均等サイズ想定)
    scenario_a_admitted = simulate_capacity_only(trades_all, max_concurrent=config.BACKTEST_MAX_CONCURRENT_POSITIONS)
    scenario_a_stats = compute_detailed_stats(scenario_a_admitted)
    scenario_b_admitted = [t for t in trades_all if t["_status"] == "admitted"]
    scenario_b_stats = compute_detailed_stats(scenario_b_admitted)
    scenario_b_portfolio_metrics = compute_portfolio_metrics(portfolio_result["nav_series"], portfolio_result["num_years"])

    print(f"C(全シグナル・制約なし): n={scenario_c_stats['num_trades']} PF={scenario_c_stats['profit_factor']} EV%={scenario_c_stats['expected_value_pct']} 勝率={scenario_c_stats['win_rate']}")
    print(f"A(3枠制約のみ・資金無限): n={scenario_a_stats['num_trades']} PF={scenario_a_stats['profit_factor']} EV%={scenario_a_stats['expected_value_pct']} 勝率={scenario_a_stats['win_rate']}")
    print(f"B(現実的資金制約): n={scenario_b_stats['num_trades']} PF={scenario_b_stats['profit_factor']} EV%={scenario_b_stats['expected_value_pct']} 勝率={scenario_b_stats['win_rate']}")
    print(f"B portfolio metrics: {scenario_b_portfolio_metrics}")

    # =====================================================================
    # ランキング方式比較
    # =====================================================================
    print("\n=== ランキング方式比較 ===")
    ranking_methods = {
        "1_score_desc": lambda t: -(t.get("signal_score") or 0),
        "2_score_asc": lambda t: (t.get("signal_score") or 0),
        "4_volume_ratio_desc": lambda t: -(t.get("signal_volume_ratio") or 0),
        "5_breakout_desc": lambda t: -(t.get("signal_breakout_pct") or -999),
        "6_breakout_asc": lambda t: (t.get("signal_breakout_pct") or 999),
        "7_atr_pct_asc": lambda t: (t.get("_atr_pct") if t.get("_atr_pct") is not None else 1e9),
        "8_return5d_asc": lambda t: (t.get("_return_5d") if t.get("_return_5d") is not None else 1e9),
    }
    ranking_results = {}
    for name, fn in ranking_methods.items():
        res = simulate_portfolio(
            trades_all, price_frames, max_concurrent=config.BACKTEST_MAX_CONCURRENT_POSITIONS,
            cash_reuse_timing="next_day", ranking_key_fn=fn,
        )
        pm = compute_portfolio_metrics(res["nav_series"], res["num_years"])
        stats = compute_detailed_stats(res["admitted_trades"])
        ranking_results[name] = {**pm, "pf": stats["profit_factor"], "ev_pct": stats["expected_value_pct"],
                                  "win_rate": stats["win_rate"], "n": stats["num_trades"]}
        print(f"{name}: n={stats['num_trades']} PF={stats['profit_factor']} EV%={stats['expected_value_pct']} CAGR%={pm['cagr_pct']} maxDD%={pm['max_drawdown_pct']}")

    # ランダム1000回(方式3/9は同一の候補集合に対する一様ランダム選択のため同じ実験で兼ねる)
    random_cagrs, random_maxdds, random_pfs, random_evs = [], [], [], []
    for seed in range(N_RANDOM_TRIALS):
        res = simulate_portfolio(
            trades_all, price_frames, max_concurrent=config.BACKTEST_MAX_CONCURRENT_POSITIONS,
            cash_reuse_timing="next_day", ranking_key="random", random_seed=seed,
        )
        pm = compute_portfolio_metrics(res["nav_series"], res["num_years"])
        stats = compute_detailed_stats(res["admitted_trades"])
        random_cagrs.append(pm["cagr_pct"])
        random_maxdds.append(pm["max_drawdown_pct"])
        if stats["profit_factor"] is not None:
            random_pfs.append(stats["profit_factor"])
        random_evs.append(stats["expected_value_pct"])

    def pct_rank(value: float, population: list[float]) -> float:
        return round(sum(1 for v in population if v <= value) / len(population) * 100, 1)

    random_summary = {
        "cagr_median": round(statistics.median(random_cagrs), 2),
        "cagr_p5": round(statistics.quantiles(random_cagrs, n=20)[0], 2),
        "cagr_p95": round(statistics.quantiles(random_cagrs, n=20)[18], 2),
        "maxdd_median": round(statistics.median(random_maxdds), 2),
        "maxdd_p5": round(statistics.quantiles(random_maxdds, n=20)[0], 2),
        "maxdd_p95": round(statistics.quantiles(random_maxdds, n=20)[18], 2),
        "pf_median": round(statistics.median(random_pfs), 3) if random_pfs else None,
        "ev_median": round(statistics.median(random_evs), 3),
        "current_score_desc_cagr_percentile": pct_rank(ranking_results["1_score_desc"]["cagr_pct"], random_cagrs),
        "current_score_desc_maxdd_percentile": pct_rank(ranking_results["1_score_desc"]["max_drawdown_pct"], random_maxdds),
    }
    print(f"ランダム{N_RANDOM_TRIALS}回: {random_summary}")

    # =====================================================================
    # 同日内ペア比較
    # =====================================================================
    print("\n=== 同日内ペア比較(entry_date単位) ===")
    by_entry: dict[str, list[dict]] = defaultdict(list)
    for t in trades_all:
        by_entry[t["entry_date"]].append(t)

    day_admitted_avg, day_rejected_avg = [], []
    day_admit_beats_reject = 0
    day_got_best = 0
    day_got_worst = 0
    competitive_days = 0
    for day, day_trades in by_entry.items():
        if len(day_trades) < 2:
            continue  # 競合が無い日は「見送り」自体が発生しないため対象外
        admitted_today = [t for t in day_trades if t["_status"] == "admitted"]
        rejected_today = [t for t in day_trades if t["_status"] in ("no_slot", "cash_insufficient")]
        if not admitted_today or not rejected_today:
            continue
        competitive_days += 1
        a_avg = statistics.mean(t["pnl_pct"] for t in admitted_today)
        r_avg = statistics.mean(t["pnl_pct"] for t in rejected_today)
        day_admitted_avg.append(a_avg)
        day_rejected_avg.append(r_avg)
        if a_avg > r_avg:
            day_admit_beats_reject += 1
        best_pnl = max(t["pnl_pct"] for t in day_trades)
        worst_pnl = min(t["pnl_pct"] for t in day_trades)
        if any(t["pnl_pct"] == best_pnl for t in admitted_today):
            day_got_best += 1
        if any(t["pnl_pct"] == worst_pnl for t in admitted_today):
            day_got_worst += 1

    same_day_summary = {
        "competitive_days": competitive_days,
        "admitted_dayavg_of_dayavg": round(statistics.mean(day_admitted_avg), 3) if day_admitted_avg else None,
        "rejected_dayavg_of_dayavg": round(statistics.mean(day_rejected_avg), 3) if day_rejected_avg else None,
        "pct_days_admitted_beats_rejected": round(day_admit_beats_reject / competitive_days * 100, 1) if competitive_days else None,
        "pct_days_admitted_got_best": round(day_got_best / competitive_days * 100, 1) if competitive_days else None,
        "pct_days_admitted_got_worst": round(day_got_worst / competitive_days * 100, 1) if competitive_days else None,
    }
    print(f"競合日数: {competitive_days}")
    print(f"日次平均の平均(採用): {same_day_summary['admitted_dayavg_of_dayavg']}%  (見送り): {same_day_summary['rejected_dayavg_of_dayavg']}%")
    print(f"採用が見送りを上回った日の割合: {same_day_summary['pct_days_admitted_beats_rejected']}%")
    print(f"最良候補を採用できた日の割合: {same_day_summary['pct_days_admitted_got_best']}%")
    print(f"最悪候補を採用してしまった日の割合: {same_day_summary['pct_days_admitted_got_worst']}%")

    # 採用順位別・同日候補数別(qualifying=score>=45の候補のみが母集団)
    rank_within_day: dict[tuple, int] = {}
    for day, day_trades in by_entry.items():
        ranked = sorted(day_trades, key=lambda t: -(t.get("signal_score") or 0))
        for rank, t in enumerate(ranked, start=1):
            rank_within_day[(t["symbol"], t["signal_date"])] = rank
    rank_stats: dict[int, list[float]] = defaultdict(list)
    count_stats: dict[int, list[float]] = defaultdict(list)  # candidate-count -> admitted trades' pnl
    for t in trades_all:
        key = (t["symbol"], t["signal_date"])
        rank_stats[rank_within_day[key]].append(t["pnl_pct"] if t["_status"] == "admitted" else None)
    for day, day_trades in by_entry.items():
        n_candidates = len(day_trades)
        for t in day_trades:
            if t["_status"] == "admitted":
                count_stats[n_candidates].append(t["pnl_pct"])

    print("採用順位別admitted pnl平均: " + str({
        r: round(statistics.mean([v for v in vs if v is not None]), 3)
        for r, vs in sorted(rank_stats.items()) if any(v is not None for v in vs)
    }))

    # =====================================================================
    # 特徴量分布比較(採用 vs 見送り、効果量つき)
    # =====================================================================
    print("\n=== 特徴量分布比較 ===")
    admitted_df = pd.DataFrame([{**{f"_{c}": t.get(f"_{c}") for c in RANK_FEATURES}, "n_same_day": len(by_entry[t["entry_date"]])}
                                 for t in trades_all if t["_status"] == "admitted"])
    rejected_df = pd.DataFrame([{**{f"_{c}": t.get(f"_{c}") for c in RANK_FEATURES}, "n_same_day": len(by_entry[t["entry_date"]])}
                                 for t in trades_all if t["_status"] in ("no_slot", "cash_insufficient")])
    distribution_comparison = {}
    for col in RANK_FEATURES + ["n_same_day"]:
        colname = col if col == "n_same_day" else f"_{col}"
        a_desc = describe_group(admitted_df[colname])
        r_desc = describe_group(rejected_df[colname])
        d = cohens_d(admitted_df[colname], rejected_df[colname])
        distribution_comparison[col] = {"admitted": a_desc, "rejected": r_desc, "cohens_d": d}
        print(f"{col}: admitted={a_desc} rejected={r_desc} cohens_d={d}")

    # =====================================================================
    # Train内 時系列安定性(半年ごと)
    # =====================================================================
    print("\n=== Train内 半年ごとの安定性 ===")
    trades_train = [t for t in trades_all if train_start <= t["signal_date"] < train_end]
    periods = sorted({t["signal_date"][:4] + ("H1" if t["signal_date"][5:7] <= "06" else "H2") for t in trades_train})
    temporal_stats = {}
    for p in periods:
        year, half = p[:4], p[4:]
        if half == "H1":
            sub = [t for t in trades_train if t["signal_date"][:4] == year and t["signal_date"][5:7] <= "06"]
        else:
            sub = [t for t in trades_train if t["signal_date"][:4] == year and t["signal_date"][5:7] > "06"]
        stats = compute_detailed_stats(sub)
        temporal_stats[p] = stats
        print(f"{p}: n={stats['num_trades']} PF={stats['profit_factor']} EV%={stats['expected_value_pct']} 勝率={stats['win_rate']}")

    # =====================================================================
    # レポート出力
    # =====================================================================
    report_path = config.REPORTS_DIR / f"anti_selection_analysis_{pd.Timestamp.today().date().isoformat()}.md"
    write_report(
        report_path, build_run, feat_df, trades_all, portfolio_result, n_no_slot, n_cash,
        scenario_a_stats, scenario_b_stats, scenario_b_portfolio_metrics, scenario_c_stats,
        ranking_results, random_summary, same_day_summary, rank_stats, count_stats,
        distribution_comparison, temporal_stats, n_exit_past_test,
    )
    print(f"\nレポート出力: {report_path}")


def write_report(path, build_run, feat_df, trades_all, portfolio_result, n_no_slot, n_cash,
                  scenario_a_stats, scenario_b_stats, scenario_b_portfolio_metrics, scenario_c_stats,
                  ranking_results, random_summary, same_day_summary, rank_stats, count_stats,
                  distribution_comparison, temporal_stats, n_exit_past_test) -> None:
    lines = [f"# Phase2 Step1 監査・逆選択分析 ({pd.Timestamp.today().date().isoformat()})", ""]
    lines.append(f"build_run_id: {build_run['build_run_id']}")
    lines.append("")
    lines.append("## 対象データ区分(このレポート全体に共通)")
    lines.append("")
    lines.append(
        f"- 対象期間: Train {build_run['train_start']}〜{build_run['train_end']} + "
        f"Validation {build_run['val_start']}〜{build_run['val_end']}(=Train+Validation全体)\n"
        f"- Test期間({build_run['test_start']}〜{build_run['end_date']})は、features/labelsの"
        f"読み込みSQLの段階(`WHERE signal_date < test_start`)で除外しており、以降のいかなる"
        f"集計処理にも一切使われていない(スクリプト内でassertにより機械的に検証済み)。\n"
        f"- 候補集合の定義: 一次スクリーニング条件(price/market_cap/avg_volume)を満たした"
        f"universe candidateのうち、technical_score_v1>=45で信号化されたもの"
        f"(=backtest.engine.simulate_symbolの出力トレード)。信号化されなかった"
        f"(スコア未達 or 同一銘柄内クールダウン中の)universe candidateは、シナリオ比較・"
        f"ランキング比較の対象には含まれていない(未解決事項として末尾に明記)。"
    )
    lines.append("")
    lines.append(
        f"**透明性の注記**: signal_date(Train/Validation)のうち{n_exit_past_test}件は、"
        f"最大保有{config.BACKTEST_MAX_HOLDING_DAYS}営業日後のexit_dateがTest開始日以降になる"
        "(ラベル設計上、シグナル自体はTrain/Valの正当なデータ点であり、結果確定に将来数日〜"
        "数週間分の価格データを要するのは一般的なラベル設計であるため)。price_framesは"
        "test_start未満に切り詰めているため、これらのトレードの一部はポートフォリオNAV上で"
        "保有中スロットが凍結される(軽微な近似。Test期間の候補自体の成績評価には使っていない)。"
    )
    lines.append("")

    lines.append("## 2. 件数整合性")
    lines.append("")
    lines.append(f"- features(Train+Valのみ): {len(feat_df)}件")
    lines.append(f"- うちcandidate_flag=1(universe candidate): {int((feat_df['candidate_flag']==1).sum())}件")
    lines.append(f"- シグナル(score>=45、Train+Valのみ): {len(trades_all)}件")
    lines.append(f"- 採用: {portfolio_result['trades_taken']}件 / 見送り(枠不足): {n_no_slot}件 / 見送り(現金不足): {n_cash}件")
    lines.append(
        "- 前回報告の428/1320(採用/枠不足見送り)は、全期間(559/1559)からTest期間分を除いた"
        "近似値だった。今回はTest除外済みデータで同一シミュレーションを再実行しており、"
        "上記の数値が正とする値である。"
    )
    lines.append("")

    lines.append("## 9. シナリオA/B/C(採用ルールと資金制約の寄与分離)")
    lines.append("")
    lines.append("| シナリオ | 説明 | n | PF | EV(%) | 勝率(%) |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| C | 全シグナル・制約なし(均等サイズ想定) | {scenario_c_stats['num_trades']} | {scenario_c_stats['profit_factor']} | {scenario_c_stats['expected_value_pct']} | {scenario_c_stats['win_rate']} |")
    lines.append(f"| A | 3枠制約のみ・資金無限(スコア降順採用) | {scenario_a_stats['num_trades']} | {scenario_a_stats['profit_factor']} | {scenario_a_stats['expected_value_pct']} | {scenario_a_stats['win_rate']} |")
    lines.append(f"| B | 現実的資金制約(現行) | {scenario_b_stats['num_trades']} | {scenario_b_stats['profit_factor']} | {scenario_b_stats['expected_value_pct']} | {scenario_b_stats['win_rate']} |")
    lines.append("")
    lines.append(f"B(現行)のポートフォリオ指標: {scenario_b_portfolio_metrics}")
    lines.append("")
    lines.append(
        "解釈: C→Aで悪化していれば「3枠の順番待ち(タイミング)」自体が逆選択を生んでいる。"
        "A→Bでさらに悪化していれば、現金制約が追加でダメージを与えている。"
        "C自体が既に悪ければ、そもそもスコアによる銘柄選択自体に優位性が無い。"
    )
    lines.append("")

    lines.append("## 8. ランキング方式比較")
    lines.append("")
    lines.append("| 方式 | n | PF | EV(%) | 勝率(%) | CAGR(%) | maxDD(%) |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, r in ranking_results.items():
        lines.append(f"| {name} | {r['n']} | {r['pf']} | {r['ev_pct']} | {r['win_rate']} | {r['cagr_pct']} | {r['max_drawdown_pct']} |")
    lines.append("")
    lines.append(f"ランダム(方式3/9兼用、{N_RANDOM_TRIALS}回試行): {random_summary}")
    lines.append("")
    lines.append(
        f"現行のスコア降順(方式1)のCAGRは、ランダム{N_RANDOM_TRIALS}回試行の分布のうち"
        f"{random_summary['current_score_desc_cagr_percentile']}パーセンタイルに位置する"
        "(50%前後ならランダムと有意差なし、高いほどスコアに順位付け能力があることを示す)。"
    )
    lines.append("")

    lines.append("## 6. 同日内ペア比較")
    lines.append("")
    for k, v in same_day_summary.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("採用順位別 admitted pnl平均(%): " + str({
        r: round(statistics.mean([v for v in vs if v is not None]), 3)
        for r, vs in sorted(rank_stats.items()) if any(v is not None for v in vs)
    }))
    lines.append("")
    lines.append("同日候補数別 admitted pnl平均(%)(件数の多い日に支配されないよう、候補数ごとに独立集計): " + str({
        n: round(statistics.mean(vs), 3) for n, vs in sorted(count_stats.items())
    }))
    lines.append("")

    lines.append("## 7. 特徴量分布比較(採用 vs 見送り)")
    lines.append("")
    lines.append("| 特徴量 | 採用n | 採用平均 | 採用中央値 | 採用Q1 | 採用Q3 | 見送りn | 見送り平均 | 見送り中央値 | 見送りQ1 | 見送りQ3 | Cohen's d |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for col, d in distribution_comparison.items():
        a, r = d["admitted"], d["rejected"]
        lines.append(
            f"| {col} | {a['n']} | {a['mean']} | {a['median']} | {a['q1']} | {a['q3']} | "
            f"{r['n']} | {r['mean']} | {r['median']} | {r['q1']} | {r['q3']} | {d['cohens_d']} |"
        )
    lines.append("")
    lines.append(
        "Cohen's dが正 = 採用された候補の方が値が大きい。|d|>=0.2で小、>=0.5で中、>=0.8で大の目安"
        "(あくまで目安であり、分布の非正規性を考慮した簡易指標である点に注意)。"
    )
    lines.append("")

    lines.append("## 10. Train内 半年ごとの時系列安定性")
    lines.append("")
    lines.append("| 期間 | n | PF | EV(%) | 勝率(%) |")
    lines.append("|---|---|---|---|---|")
    for p, s in temporal_stats.items():
        lines.append(f"| {p} | {s['num_trades']} | {s['profit_factor']} | {s['expected_value_pct']} | {s['win_rate']} |")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
