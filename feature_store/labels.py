"""future label(将来の結果)の計算。features.py とは完全に独立したモジュール。

重要な原則:
- ここで計算する値は全て「シグナル発生日より後」の価格データのみを使う。
- 全てのラベルは「シグナル翌営業日の始値」を基準(エントリー価格)とする(target_15pct_within_10d
  の定義に合わせ、他のラベルもこれで統一する)。
- 指定した将来ウィンドウの価格データが十分に存在しない場合は、0/1で確定させずNone(不明)を返す
  (「未到達」と「データ不足で判定不能」を混同しない)。
- target_trade_success / hit_stop_atr_1_5_before_plus_15 は backtest.engine.price_trade_at_signal
  (本番のバックテストエンジンと全く同じATRストップ/利確ロジック)をそのまま再利用する。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from analysis.scoring import score_technical
from analysis.technical import row_to_indicators
from backtest.engine import price_trade_at_signal

LABEL_VERSION = "v2.0.0-phase1"


def _full_window_available(entry_idx: int, days: int, n: int) -> bool:
    return (entry_idx + days - 1) <= (n - 1)


def compute_labels_for_signal(
    df: pd.DataFrame,
    signal_idx: int,
    symbol: str,
) -> Optional[dict]:
    """1件のシグナル(signal_idx)について、将来ラベルの辞書を返す。
    翌営業日のデータが存在しない(データ末尾)場合はNoneを返す(ラベル自体が確定不能)。
    """
    n = len(df)
    entry_idx = signal_idx + 1
    if entry_idx >= n:
        return None

    entry_price = float(df.iloc[entry_idx]["open"])
    signal_date = df.iloc[signal_idx]["date"].strftime("%Y-%m-%d")

    def future_return(days: int) -> Optional[float]:
        idx = entry_idx + days - 1
        if idx > n - 1:
            return None
        return (float(df.iloc[idx]["close"]) / entry_price - 1) * 100

    def future_max_return(days: int) -> Optional[float]:
        if not _full_window_available(entry_idx, days, n):
            return None
        window = df.iloc[entry_idx : entry_idx + days]
        return (float(window["high"].max()) / entry_price - 1) * 100

    def future_min_return(days: int) -> Optional[float]:
        if not _full_window_available(entry_idx, days, n):
            return None
        window = df.iloc[entry_idx : entry_idx + days]
        return (float(window["low"].min()) / entry_price - 1) * 100

    def days_to_pct(threshold_pct: float, horizon: int) -> Optional[int]:
        """到達すれば経過営業日数(エントリー日=1)、ウィンドウ内に未到達ならNone。"""
        end_idx = min(entry_idx + horizon - 1, n - 1)
        for offset, idx in enumerate(range(entry_idx, end_idx + 1), start=1):
            if float(df.iloc[idx]["high"]) >= entry_price * (1 + threshold_pct / 100):
                return offset
        return None

    def hit_pct_within(threshold_pct: float, horizon: int) -> Optional[int]:
        """到達すれば1、ウィンドウ全体のデータがあり未到達なら0、データ不足ならNone。"""
        if not _full_window_available(entry_idx, horizon, n):
            return None
        return 1 if days_to_pct(threshold_pct, horizon) is not None else 0

    future_return_1d = future_return(1)
    future_return_3d = future_return(3)
    future_return_5d = future_return(5)
    future_return_10d = future_return(10)
    future_return_15d = future_return(15)

    future_max_return_5d = future_max_return(5)
    future_max_return_10d = future_max_return(10)
    future_max_return_15d = future_max_return(15)
    future_min_return_5d = future_min_return(5)
    future_min_return_10d = future_min_return(10)
    future_min_return_15d = future_min_return(15)

    hit_plus_5pct_10d = hit_pct_within(5, 10)
    hit_plus_10pct_10d = hit_pct_within(10, 10)
    hit_plus_15pct_10d = hit_pct_within(15, 10)
    hit_plus_15pct_15d = hit_pct_within(15, 15)

    # 到達日数系は15営業日の固定ウィンドウで検索する。Noneは「15営業日以内に未到達」と
    # 「データが15営業日分そろう前に打ち切られた」の両方を区別せず返す簡易仕様
    # (hit_plus_*系のように厳密にNone/0/1を切り分けたい場合はそちらを参照する)。
    days_to_plus_5pct = days_to_pct(5, 15)
    days_to_plus_10pct = days_to_pct(10, 15)
    days_to_plus_15pct = days_to_pct(15, 15)

    # 主要ラベル1: 翌営業日始値基準、10営業日以内に高値が+15%へ到達したか
    target_15pct_within_10d = hit_plus_15pct_10d

    # 主要ラベル2・hit_stop系: 本番と全く同じATRストップ/利確ロジックを再利用する
    indicators = row_to_indicators(df, signal_idx)
    tech = score_technical(indicators)
    signal_score = tech["subtotal"]

    # exit_reason=="data_end"は「保有期間を使い切る前にデータが尽きた」(上場廃止・データ末尾)
    # ことを意味し、実際の結果が不明なケース。0/1で確定させずNone(未確定)として扱う
    # (stop_loss/take_profitヒットや、正当な期間満了holding_period_limitは確定値として扱ってよい)。
    trade_10d = price_trade_at_signal(
        symbol, df, signal_idx, indicators, signal_score, max_holding_days=10,
        entry_mode="next_open", slippage_pct=0.2, commission_pct=0.1, same_day_priority="stop_first",
    )
    target_trade_success = None
    if trade_10d is not None and trade_10d.get("exit_reason") != "data_end":
        target_trade_success = 1 if trade_10d["outcome"] == "win" else 0

    trade_15d = price_trade_at_signal(
        symbol, df, signal_idx, indicators, signal_score, max_holding_days=15,
        entry_mode="next_open", slippage_pct=0.2, commission_pct=0.1, same_day_priority="stop_first",
    )
    hit_stop_atr_1_5_before_plus_15 = None
    if trade_15d is not None and trade_15d.get("exit_reason") != "data_end":
        hit_stop_atr_1_5_before_plus_15 = 1 if trade_15d["outcome"] == "loss" else 0

    return {
        "ticker": symbol,
        "signal_date": signal_date,
        "future_return_1d": future_return_1d,
        "future_return_3d": future_return_3d,
        "future_return_5d": future_return_5d,
        "future_return_10d": future_return_10d,
        "future_return_15d": future_return_15d,
        "future_max_return_5d": future_max_return_5d,
        "future_max_return_10d": future_max_return_10d,
        "future_max_return_15d": future_max_return_15d,
        "future_min_return_5d": future_min_return_5d,
        "future_min_return_10d": future_min_return_10d,
        "future_min_return_15d": future_min_return_15d,
        "hit_plus_5pct_10d": hit_plus_5pct_10d,
        "hit_plus_10pct_10d": hit_plus_10pct_10d,
        "hit_plus_15pct_10d": hit_plus_15pct_10d,
        "hit_plus_15pct_15d": hit_plus_15pct_15d,
        "hit_stop_atr_1_5_before_plus_15": hit_stop_atr_1_5_before_plus_15,
        "days_to_plus_5pct": days_to_plus_5pct,
        "days_to_plus_10pct": days_to_plus_10pct,
        "days_to_plus_15pct": days_to_plus_15pct,
        "target_15pct_within_10d": target_15pct_within_10d,
        "target_trade_success": target_trade_success,
        "label_computed_at": datetime.now(timezone.utc).isoformat(),
        "label_version": LABEL_VERSION,
    }
