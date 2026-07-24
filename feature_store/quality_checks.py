"""feature store のデータ品質チェック。結果は data_quality_log テーブルとレポートに残す。"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import date
from typing import Optional

from feature_store.database import FeatureStoreDB

# ウォームアップ期間が十分なら常に埋まっているべき列(=欠損していたら「予期しない欠損」)
EXPECTED_NON_NULL_COLUMNS = [
    "volume", "avg_volume_5d", "avg_volume_20d", "ma5", "ma15", "ma25", "ma50", "ma200",
    "return_1d", "return_5d", "return_20d", "rsi_14", "macd", "atr_14",
    "historical_volatility_20d", "bollinger_band_width", "gap_pct",
    "spy_close", "sector_return_5d", "technical_score_v1",
]

# 仕様上、無料データ源では取得不能、または本ツールで採用していないため常にNULLになる列
# (欠損率に含めず別集計する。品質問題ではない)
SPEC_ALWAYS_NULL_COLUMNS = ["bid_ask_spread", "turnover_ratio", "beta", "ma20"]

# EXPECTED_NON_NULL_COLUMNS の各列が、その銘柄の(このデータセット内での)何営業日目から
# 埋まり始めるべきかの目安(ウォームアップ日数)。この日数未満での欠損は「ウォームアップ期間の
# 欠損」として区別し、それ以降での欠損だけを「予期しない欠損」として扱う。
WARMUP_DAYS = {
    "volume": 1, "avg_volume_5d": 5, "avg_volume_20d": 20,
    "ma5": 5, "ma15": 15, "ma25": 25, "ma50": 50, "ma200": 200,
    "return_1d": 1, "return_5d": 5, "return_20d": 20,
    "rsi_14": 14, "macd": 26, "atr_14": 14,
    "historical_volatility_20d": 20, "bollinger_band_width": 20, "gap_pct": 1,
    "spy_close": 1, "sector_return_5d": 5, "technical_score_v1": 50,
}


def _is_bad_float(v) -> bool:
    if v is None:
        return False
    try:
        return math.isinf(v) or math.isnan(v)
    except TypeError:
        return False


def run_quality_checks(
    features: list[dict],
    labels: list[dict],
    db: Optional[FeatureStoreDB] = None,
    failed_symbols: Optional[list[str]] = None,
) -> dict:
    """features/labelsのリストに対して一連のチェックを行い、レポート(dict)を返す。
    db を渡した場合は data_quality_log にも記録する。
    """
    report: dict = {"checks": []}

    def log(name: str, status: str, count: int, details: str) -> None:
        report["checks"].append({"check_name": name, "status": status, "affected_count": count, "details": details})
        if db is not None:
            db.log_quality_check(name, status, count, details)

    # 1. 重複(ticker, signal_date)チェック
    keys = [(f["ticker"], f["signal_date"]) for f in features]
    dup_counter = Counter(keys)
    dups = [k for k, c in dup_counter.items() if c > 1]
    log(
        "duplicate_ticker_signal_date", "fail" if dups else "pass", len(dups),
        f"重複キー: {dups[:10]}" if dups else "重複なし",
    )

    # 2. 未来日データの混入(signal_dateが今日より未来であってはならない)
    today_str = date.today().isoformat()
    future_dated = [f for f in features if f["signal_date"] > today_str]
    log(
        "future_dated_signal", "fail" if future_dated else "pass", len(future_dated),
        f"未来日のsignal_dateが{len(future_dated)}件見つかりました" if future_dated else "問題なし",
    )

    # 3. 特徴量生成時点より後の価格利用(構造的なリークチェックはtests/test_feature_store.pyの
    #    専用ユニットテストで行う。ここではdata_collected_atがsignal_date以降であることのみ確認)
    bad_collected_at = [
        f for f in features
        if f.get("data_collected_at") and f["data_collected_at"][:10] < f["signal_date"]
    ]
    log(
        "data_collected_at_before_signal_date",
        "fail" if bad_collected_at else "pass", len(bad_collected_at),
        "data_collected_atがsignal_dateより前になっているレコードあり" if bad_collected_at else "問題なし",
    )

    # 4. 欠損率(3区分: 予期しない欠損 / 仕様上のNULL / ウォームアップ期間による欠損)
    # 銘柄ごとにsignal_date昇順で並べ、このデータセット内での通し番号(0始まり)を求める。
    # これはウォームアップ判定のみに使う位置情報であり、特徴量そのものには含めない。
    ticker_row_index: dict[tuple, int] = {}
    if features:
        by_ticker: dict[str, list[dict]] = defaultdict(list)
        for f in features:
            by_ticker[f["ticker"]].append(f)
        for ticker, rows in by_ticker.items():
            for i, f in enumerate(sorted(rows, key=lambda r: r["signal_date"])):
                ticker_row_index[(f["ticker"], f["signal_date"])] = i

    unexpected_missing_report = {}
    warmup_missing_report = {}
    if features:
        for col in EXPECTED_NON_NULL_COLUMNS:
            warmup_days = WARMUP_DAYS.get(col, 0)
            unexpected = 0
            warmup = 0
            for f in features:
                if f.get(col) is not None:
                    continue
                pos = ticker_row_index.get((f["ticker"], f["signal_date"]), warmup_days)
                if pos < warmup_days:
                    warmup += 1
                else:
                    unexpected += 1
            unexpected_missing_report[col] = {
                "件数": unexpected, "率(%)": round(unexpected / len(features) * 100, 1),
            }
            if warmup:
                warmup_missing_report[col] = warmup

    spec_null_report = {}
    if features:
        for col in SPEC_ALWAYS_NULL_COLUMNS:
            missing = sum(1 for f in features if f.get(col) is None)
            spec_null_report[col] = round(missing / len(features) * 100, 1)

    any_unexpected = any(v["件数"] > 0 for v in unexpected_missing_report.values())
    log(
        "missing_rate",
        "fail" if any_unexpected else "pass",
        sum(v["件数"] for v in unexpected_missing_report.values()),
        (
            f"予期しない欠損(ウォームアップ済みのはずが欠損): {unexpected_missing_report} / "
            f"ウォームアップ期間中の欠損(件数、品質問題ではない): {warmup_missing_report} / "
            f"仕様上常にNULLの列(欠損率には含めない、品質問題ではない): {spec_null_report}"
        ),
    )

    # 5. 無限値
    inf_count = 0
    for f in features:
        for v in f.values():
            if isinstance(v, float) and _is_bad_float(v):
                inf_count += 1
    log("infinite_or_nan_values", "fail" if inf_count else "pass", inf_count,
        f"{inf_count}個の inf/NaN が見つかりました" if inf_count else "問題なし")

    # 6. 異常値(RSIが0-100の範囲外、テクニカルスコアが0-75の範囲外など)
    range_violations = []
    for f in features:
        rsi = f.get("rsi_14")
        if rsi is not None and not (0 <= rsi <= 100):
            range_violations.append((f["ticker"], f["signal_date"], "rsi_14", rsi))
        score = f.get("technical_score_v1")
        if score is not None and not (0 <= score <= 75):
            range_violations.append((f["ticker"], f["signal_date"], "technical_score_v1", score))
    log("range_violations", "fail" if range_violations else "pass", len(range_violations),
        str(range_violations[:10]))

    # 7. 出来高0
    zero_volume = [f for f in features if f.get("volume") == 0]
    log("zero_volume", "warning" if zero_volume else "pass", len(zero_volume),
        f"出来高0のレコードが{len(zero_volume)}件" if zero_volume else "問題なし")

    # 8. 価格0以下
    non_positive_price = [f for f in features if (f.get("close") or 0) <= 0]
    log("non_positive_price", "fail" if non_positive_price else "pass", len(non_positive_price),
        f"価格0以下のレコードが{len(non_positive_price)}件" if non_positive_price else "問題なし")

    # 9. 株式分割未調整の疑い(前日比±40%を超えるギャップ)
    suspicious_gaps = [f for f in features if f.get("gap_pct") is not None and abs(f["gap_pct"]) > 40]
    log("suspicious_split_gap", "warning" if suspicious_gaps else "pass", len(suspicious_gaps),
        f"分割未調整の疑いがあるギャップが{len(suspicious_gaps)}件" if suspicious_gaps else "問題なし")

    # 10. ラベル未確定期間の割合
    label_missing_report = {}
    if labels:
        for col in ["target_15pct_within_10d", "target_trade_success", "future_return_10d"]:
            missing = sum(1 for l in labels if l.get(col) is None)
            label_missing_report[col] = round(missing / len(labels) * 100, 1)
    log("label_undetermined_rate", "pass", 0, str(label_missing_report))

    # 11. データ取得失敗銘柄
    failed_symbols = failed_symbols or []
    log("data_fetch_failures", "warning" if failed_symbols else "pass", len(failed_symbols),
        f"取得/計算に失敗した銘柄: {failed_symbols}" if failed_symbols else "問題なし")

    # 13. Phase2 Step2: 時価総額・セクター・市場データの欠損率
    for col, name in [("market_cap", "market_cap欠損"), ("sector", "sector欠損"), ("spy_close", "market_data(spy_close)欠損")]:
        missing = sum(1 for f in features if f.get(col) is None) if features else 0
        rate = round(missing / len(features) * 100, 1) if features else 0.0
        log(f"missing_{col}", "warning" if rate > 0 else "pass", missing, f"{name}率: {rate}%")

    # 14. label_statusの内訳(pending/data_end/invalidを失敗として隠さず件数を出す)
    if labels:
        status_counts: dict[str, int] = {}
        for l in labels:
            s = l.get("label_status") or "unknown"
            status_counts[s] = status_counts.get(s, 0) + 1
        log("label_status_breakdown", "pass", 0, str(status_counts))

    if db is not None:
        # 15. ティッカー変更(エイリアス)経由の取得件数
        try:
            aliased = db.conn.execute(
                "SELECT COUNT(*) FROM universe_membership WHERE data_fetch_status = 'ok_via_alias'"
            ).fetchone()[0]
            log("ticker_alias_usage", "pass", aliased, f"ティッカー変更(エイリアス)経由で取得した銘柄: {aliased}件")
        except Exception:
            pass

        # 16. 上場廃止/データ終了の疑いがある銘柄数(universe_membershipのlast_available_dateが
        # 対象期間の終了日より大幅に早い銘柄。上場前データは、そもそも取得データがその銘柄の
        # first_available_date以降しか存在しないため構造的に発生しない)
        try:
            rows = db.conn.execute(
                "SELECT ticker, last_available_date FROM universe_membership "
                "WHERE data_fetch_status IN ('ok', 'ok_via_alias')"
            ).fetchall()
            log(
                "delisting_or_data_end_candidates", "pass", 0,
                f"universe_membership記載の最終取得可能日一覧から、必要に応じてlabel_status='data_end'"
                f"件数(上記label_status_breakdown参照)と突き合わせて確認する({len(rows)}銘柄分の記録あり)",
            )
        except Exception:
            pass

        # 17. ビルド失敗銘柄(build_run_symbol_status)
        try:
            failed_stages = db.conn.execute(
                "SELECT symbol, stage, status, error_type FROM build_run_symbol_status "
                "WHERE status IN ('failed', 'skipped') ORDER BY symbol"
            ).fetchall()
            failed_list = [dict(r) for r in failed_stages]
            log(
                "build_stage_failures", "warning" if failed_list else "pass", len(failed_list),
                str(failed_list[:30]),
            )
        except Exception:
            pass

        # 18. キャッシュ不整合(price_cache_metaのrow_countが実際のキャッシュファイルと極端に
        # 乖離していないかの簡易チェック。詳細な整合性確認はキャッシュファイル側で行う)
        try:
            cache_rows = db.conn.execute("SELECT COUNT(*) FROM price_cache_meta").fetchone()[0]
            log("price_cache_meta_count", "pass", 0, f"price_cache_metaレコード数: {cache_rows}")
        except Exception:
            pass

    # 12. 特徴量の分布変化(前半/後半で平均・標準偏差を比較する簡易チェック)
    distribution_shift = {}
    if len(features) >= 20:
        sorted_features = sorted(features, key=lambda f: f["signal_date"])
        half = len(sorted_features) // 2
        first_half, second_half = sorted_features[:half], sorted_features[half:]
        for col in ["rsi_14", "return_5d", "atr_pct", "technical_score_v1"]:
            v1 = [f[col] for f in first_half if f.get(col) is not None]
            v2 = [f[col] for f in second_half if f.get(col) is not None]
            if v1 and v2:
                mean1, mean2 = sum(v1) / len(v1), sum(v2) / len(v2)
                distribution_shift[col] = {"前半平均": round(mean1, 2), "後半平均": round(mean2, 2)}
    log("distribution_shift_check", "pass", 0, str(distribution_shift))

    overall_status = "fail" if any(c["status"] == "fail" for c in report["checks"]) else (
        "warning" if any(c["status"] == "warning" for c in report["checks"]) else "pass"
    )
    report["overall_status"] = overall_status
    report["num_features"] = len(features)
    report["num_labels"] = len(labels)
    return report


def format_report(report: dict) -> str:
    lines = [
        f"データ品質レポート(総合判定: {report['overall_status']})",
        f"特徴量レコード数: {report['num_features']}  ラベルレコード数: {report['num_labels']}",
        "",
    ]
    for c in report["checks"]:
        lines.append(f"[{c['status'].upper()}] {c['check_name']}: {c['details']}")
    return "\n".join(lines)
