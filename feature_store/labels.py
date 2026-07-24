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

LABEL_VERSION = "v2.0.1-phase2step2"


def _full_window_available(entry_idx: int, days: int, n: int) -> bool:
    return (entry_idx + days - 1) <= (n - 1)


def _classify_incomplete_window(df: pd.DataFrame, n: int, as_of_date: Optional[str]) -> str:
    """将来ウィンドウが不足している行を'pending'(時間が経てば確定し得る)と
    'data_end'(銘柄側のデータがそこで終了しており今後も確定しない可能性が高い=上場廃止等)
    に分類する。as_of_dateが無ければ判別できないため常に'pending'とする(安全側デフォルト)。
    """
    if as_of_date is None or n == 0:
        return "pending"
    last_date = df.iloc[n - 1]["date"]
    last_date_str = last_date.strftime("%Y-%m-%d") if hasattr(last_date, "strftime") else str(last_date)
    try:
        gap_days = (pd.Timestamp(as_of_date) - pd.Timestamp(last_date_str)).days
    except (TypeError, ValueError):
        return "pending"
    return "pending" if gap_days <= 7 else "data_end"


def compute_labels_for_signal(
    df: pd.DataFrame,
    signal_idx: int,
    symbol: str,
    as_of_date: Optional[str] = None,
) -> dict:
    """1件のシグナル(signal_idx)について、将来ラベルの辞書を返す(常に辞書を返す。Noneは返さない)。

    as_of_date: ビルド実行時点で価格データを取得した基準日(通常はfetch_end/today)。
    翌営業日のデータが無い(entry_idx>=n)場合、またはウィンドウが不足している場合、
    その原因が「まだ時間が経っていないだけ(pending)」なのか「銘柄側のデータがそこで
    終了している(data_end、上場廃止等)」なのかを、この日付と比較して判別する。
    いずれの場合も0/1へ確定させたラベルは返さない(label_statusで区別する)。
    """
    n = len(df)
    entry_idx = signal_idx + 1
    signal_date = df.iloc[signal_idx]["date"].strftime("%Y-%m-%d")

    if entry_idx >= n:
        status = _classify_incomplete_window(df, n, as_of_date)
        return {
            "ticker": symbol, "signal_date": signal_date,
            "future_return_1d": None, "future_return_3d": None, "future_return_5d": None,
            "future_return_10d": None, "future_return_15d": None,
            "future_max_return_5d": None, "future_max_return_10d": None, "future_max_return_15d": None,
            "future_min_return_5d": None, "future_min_return_10d": None, "future_min_return_15d": None,
            "hit_plus_5pct_10d": None, "hit_plus_10pct_10d": None,
            "hit_plus_15pct_10d": None, "hit_plus_15pct_15d": None,
            "hit_stop_atr_1_5_before_plus_15": None,
            "days_to_plus_5pct": None, "days_to_plus_10pct": None, "days_to_plus_15pct": None,
            "target_15pct_within_10d": None, "target_5pct_within_10d": None, "target_10pct_within_10d": None,
            "target_trade_success": None, "target_trade_pnl_pct": None,
            "days_to_target": None, "exit_reason": None,
            "label_status": "invalid" if status == "pending" else status,  # 翌日データが全く無いのは構造的に無効
            "label_computed_at": datetime.now(timezone.utc).isoformat(),
            "label_version": LABEL_VERSION,
        }

    entry_price = float(df.iloc[entry_idx]["open"])

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
    # Phase2 Step2追加: 同じ10営業日ウィンドウでの+5%/+10%版(命名の一貫性のためhit_plus_*の別名)
    target_5pct_within_10d = hit_plus_5pct_10d
    target_10pct_within_10d = hit_plus_10pct_10d
    # target_15pct_within_10dと同じ10日ウィンドウでの到達日数(days_to_plus_15pctは15日ウィンドウ
    # のため horizon が異なる。target指標との対応を厳密にするためここで別途計算する)。
    days_to_target = days_to_pct(15, 10)

    label_status = "confirmed" if target_15pct_within_10d is not None else _classify_incomplete_window(df, n, as_of_date)

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
    target_trade_pnl_pct = None
    exit_reason = trade_10d.get("exit_reason") if trade_10d is not None else None
    if trade_10d is not None and trade_10d.get("exit_reason") != "data_end":
        target_trade_success = 1 if trade_10d["outcome"] == "win" else 0
        target_trade_pnl_pct = trade_10d["pnl_pct"]  # PF/期待値の実測計算用(スリッページ・手数料込み)

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
        "target_5pct_within_10d": target_5pct_within_10d,
        "target_10pct_within_10d": target_10pct_within_10d,
        "target_trade_success": target_trade_success,
        "target_trade_pnl_pct": target_trade_pnl_pct,
        "days_to_target": days_to_target,
        "exit_reason": exit_reason,
        "label_status": label_status,
        "label_computed_at": datetime.now(timezone.utc).isoformat(),
        "label_version": LABEL_VERSION,
    }
