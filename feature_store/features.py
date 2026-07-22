"""point-in-time特徴量の計算。

重要な原則: ここで計算する全ての値は、signal_date時点で実際に知り得た情報だけを使う
(その日の終値・それ以前の価格データ・その日までの市場/セクターデータのみ)。
未来の価格データは一切参照しない。ラベル(将来の結果)はlabels.pyで完全に別工程として計算する。

analysis.technical.compute_indicator_frame() で計算済みの指標(sma5/15/25/50, rsi14, macd,
atr14, bollinger, volume_ratio, new_20d_high, breakout_pct)を再利用し、それ以外の追加指標
(ma10/75/100/200, 複数期間リターン, ROC, ヒストリカルボラティリティ, ローソク足特徴量, ギャップ)
はこのモジュールで追加計算する。
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from analysis.scoring import score_technical
from analysis.technical import compute_indicator_frame, row_to_indicators

FEATURE_VERSION = "v2.0.0-phase1"


def compute_feature_frame(price_history: list[dict], min_rows: int = 200) -> Optional[pd.DataFrame]:
    """1銘柄分の価格履歴から、特徴量計算に必要な全列を追加したDataFrameを返す。

    min_rows=200としているのは、ma200等の長期指標を計算するのに十分なデータ量を
    要求するため(本番パイプラインのcompute_indicator_frameのmin_rows=30より厳しい)。
    """
    df = compute_indicator_frame(price_history, min_rows=min_rows)
    if df is None:
        return None

    close = df["close"]
    volume = df["volume"]

    # --- 追加の移動平均 ---
    df["ma10"] = close.rolling(10, min_periods=10).mean()
    df["ma75"] = close.rolling(75, min_periods=75).mean()
    df["ma100"] = close.rolling(100, min_periods=100).mean()
    df["ma200"] = close.rolling(200, min_periods=200).mean()

    df["price_vs_ma5_pct"] = (close - df["sma5"]) / df["sma5"] * 100
    df["price_vs_ma15_pct"] = (close - df["sma15"]) / df["sma15"] * 100
    df["price_vs_ma25_pct"] = (close - df["sma25"]) / df["sma25"] * 100
    df["price_vs_ma50_pct"] = (close - df["sma50"]) / df["sma50"] * 100
    df["price_vs_ma200_pct"] = (close - df["ma200"]) / df["ma200"] * 100

    df["ma5_slope"] = df["sma5"].diff(5)
    df["ma15_slope"] = df["sma15"].diff(5)
    df["ma25_slope"] = df["sma25"].diff(5)
    df["ma50_slope"] = df["sma50"].diff(5)
    df["ma200_slope"] = df["ma200"].diff(5)

    df["ma5_above_ma15"] = (df["sma5"] > df["sma15"]).astype("Int64")
    df["ma15_above_ma25"] = (df["sma15"] > df["sma25"]).astype("Int64")
    df["ma25_above_ma50"] = (df["sma25"] > df["sma50"]).astype("Int64")

    # --- 出来高 ---
    df["avg_volume_5d"] = volume.rolling(5, min_periods=5).mean()
    df["avg_volume_25d"] = volume.rolling(25, min_periods=25).mean()
    df["volume_ratio_5d"] = volume / df["avg_volume_5d"]
    df["volume_ratio_20d"] = df["volume_ratio"]  # compute_indicator_frameで既に20日ベースを計算済み
    df["dollar_volume"] = close * volume
    df["turnover_ratio"] = None  # 発行済株式数が必要でyfinanceの無料取得が不安定なため常にNULL

    # --- モメンタム(複数期間リターン) ---
    df["return_1d"] = close.pct_change(1) * 100
    df["return_3d"] = close.pct_change(3) * 100
    df["return_5d"] = close.pct_change(5) * 100
    df["return_10d"] = close.pct_change(10) * 100
    df["return_20d"] = close.pct_change(20) * 100

    df["rsi_7"] = _rsi(close, 7)
    # 注意: analysis.technicalのrsi14はウォームアップ不足分を中立値50でfillnaする仕様
    # (本番スコアリング用の意図的な設計)だが、point-in-time特徴量ではそれが「実際のRSI=50」
    # なのか「データ不足で計算不能」なのかを区別できなくなり、欠損チェックもすり抜けてしまう。
    # そのためfeature storeでは同じ計算をfillnaなしでここで再計算し、ウォームアップ期間は
    # NULLのまま残す(quality_checksが正しく「欠損」として検知できるようにする)。
    df["rsi_14"] = _rsi(close, 14)

    df["roc_5"] = close.pct_change(5) * 100
    df["roc_10"] = close.pct_change(10) * 100
    df["roc_20"] = close.pct_change(20) * 100

    # --- ブレイクアウト ---
    df["highest_close_10d"] = close.rolling(10, min_periods=10).max()
    df["highest_close_20d"] = close.rolling(20, min_periods=20).max()
    df["highest_high_20d"] = df["high"].rolling(20, min_periods=20).max()
    prev9_close_high = close.shift(1).rolling(9, min_periods=9).max()
    prev19_close_high = close.shift(1).rolling(19, min_periods=19).max()
    df["breakout_close_10d_pct"] = (close - prev9_close_high) / prev9_close_high * 100
    df["breakout_close_20d_pct"] = (close - prev19_close_high) / prev19_close_high * 100
    df["distance_from_20d_high_pct"] = (close - df["highest_close_20d"]) / df["highest_close_20d"] * 100
    df["days_since_20d_high"] = _days_since_rolling_high(close, window=20)
    df["new_high_60d_flag"] = (close >= close.rolling(60, min_periods=60).max()).astype("Int64")
    df["new_high_252d_flag"] = (close >= close.rolling(252, min_periods=1).max()).astype("Int64")

    # --- ボラティリティ ---
    df["atr_pct"] = df["atr14"] / close * 100
    log_returns = np.log(close / close.shift(1))
    df["historical_volatility_10d"] = log_returns.rolling(10, min_periods=10).std() * math.sqrt(252) * 100
    df["historical_volatility_20d"] = log_returns.rolling(20, min_periods=20).std() * math.sqrt(252) * 100
    df["bollinger_band_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"] * 100
    df["distance_upper_band_pct"] = (close - df["bb_upper"]) / df["bb_upper"] * 100
    prev_close = close.shift(1)
    df["gap_pct"] = (df["open"] - prev_close) / prev_close * 100
    df["average_gap_20d"] = df["gap_pct"].abs().rolling(20, min_periods=20).mean()
    df["beta"] = None  # 市場ベータの計算はPhase2以降(SPYとの回帰が必要なため)に持ち越し

    # --- ローソク足 ---
    high, low, open_ = df["high"], df["low"], df["open"]
    candle_range = (high - low).replace(0, np.nan)
    df["body_pct"] = (close - open_).abs() / candle_range * 100
    df["upper_wick_pct"] = (high - df[["open", "close"]].max(axis=1)) / candle_range * 100
    df["lower_wick_pct"] = (df[["open", "close"]].min(axis=1) - low) / candle_range * 100
    df["close_location_value"] = ((close - low) - (high - close)) / candle_range
    prev_high, prev_low = high.shift(1), low.shift(1)
    df["inside_bar_flag"] = ((high <= prev_high) & (low >= prev_low)).astype("Int64")
    df["outside_bar_flag"] = ((high > prev_high) & (low < prev_low)).astype("Int64")

    return df


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """analysis.technical._rsiと同じ計算式だが、ウォームアップ不足分をNULL(NaN)のまま残す
    (本番pipelineのfillna(50)は意図的な仕様だが、point-in-time特徴量では欠損を隠してしまうため採用しない)。"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, np.where(avg_gain > 0, 100.0, 50.0))
    return pd.Series(rsi, index=close.index)


def _days_since_rolling_high(close: pd.Series, window: int) -> pd.Series:
    result = []
    for i in range(len(close)):
        if i < window - 1:
            result.append(None)
            continue
        window_slice = close.iloc[i - window + 1 : i + 1]
        max_idx = window_slice.values.argmax()
        result.append(int(window - 1 - max_idx))
    return pd.Series(result, index=close.index)


def _safe(v) -> Optional[float]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return float(v)


def build_feature_row(
    df: pd.DataFrame,
    idx: int,
    ticker: str,
    exchange: Optional[str],
    sector: Optional[str],
    industry: Optional[str],
    market_cap: Optional[float],
    candidate_source: str,
    market_features: dict,
    sector_features: dict,
    universe_included_flag: int = 1,
    candidate_flag: int = 1,
    candidate_reason: str = "ok",
    universe_version: Optional[str] = None,
) -> dict:
    """DataFrameの特定行(idx)から、feature_store.features テーブル1行分の辞書を作る。

    universe_included_flag/candidate_flag/candidate_reason/universe_version は
    呼び出し側(feature_store.universe等)で判定した結果をそのまま渡す想定。
    """
    row = df.iloc[idx]
    indicators = row_to_indicators(df, idx)
    tech = score_technical(indicators)
    breakdown = tech["breakdown"]

    signal_date = row["date"].strftime("%Y-%m-%d")
    price = float(row["close"])

    stock_vs_sector_5d = None
    stock_vs_sector_20d = None
    if sector_features.get("sector_return_5d") is not None and _safe(row.get("return_5d")) is not None:
        stock_vs_sector_5d = _safe(row.get("return_5d")) - sector_features["sector_return_5d"]
    if sector_features.get("sector_return_20d") is not None and _safe(row.get("return_20d")) is not None:
        stock_vs_sector_20d = _safe(row.get("return_20d")) - sector_features["sector_return_20d"]

    feature = {
        "ticker": ticker,
        "signal_date": signal_date,
        "exchange": exchange,
        "sector": sector,
        "industry": industry,
        "market_cap": market_cap,
        "price": price,
        "candidate_source": candidate_source,
        "data_collected_at": datetime.now(timezone.utc).isoformat(),
        "feature_version": FEATURE_VERSION,

        "universe_included_flag": universe_included_flag,
        "candidate_flag": candidate_flag,
        "candidate_reason": candidate_reason,
        "universe_version": universe_version,

        "volume": _safe(row.get("volume")),
        "avg_volume_5d": _safe(row.get("avg_volume_5d")),
        "avg_volume_20d": _safe(row.get("avg_volume20")),
        "avg_volume_25d": _safe(row.get("avg_volume_25d")),
        "volume_ratio_5d": _safe(row.get("volume_ratio_5d")),
        "volume_ratio_20d": _safe(row.get("volume_ratio_20d")),
        "dollar_volume": _safe(row.get("dollar_volume")),
        "turnover_ratio": None,
        "turnover_ratio_available": 0,  # 発行済株式数の無料取得が不安定なため仕様上常に取得不能
        "bid_ask_spread": None,
        "bid_ask_spread_available": 0,  # 無料データ源では仕様上常に取得不能(0埋めではなくNULL+フラグ)

        "ma5": _safe(row.get("sma5")), "ma10": _safe(row.get("ma10")), "ma15": _safe(row.get("sma15")),
        "ma20": None,  # 20日線は本ツールでは未使用(15日/25日を採用)のため常にNULL
        "ma25": _safe(row.get("sma25")), "ma50": _safe(row.get("sma50")),
        "ma75": _safe(row.get("ma75")), "ma100": _safe(row.get("ma100")), "ma200": _safe(row.get("ma200")),
        "price_vs_ma5_pct": _safe(row.get("price_vs_ma5_pct")),
        "price_vs_ma15_pct": _safe(row.get("price_vs_ma15_pct")),
        "price_vs_ma25_pct": _safe(row.get("price_vs_ma25_pct")),
        "price_vs_ma50_pct": _safe(row.get("price_vs_ma50_pct")),
        "price_vs_ma200_pct": _safe(row.get("price_vs_ma200_pct")),
        "ma5_slope": _safe(row.get("ma5_slope")), "ma15_slope": _safe(row.get("ma15_slope")),
        "ma25_slope": _safe(row.get("ma25_slope")), "ma50_slope": _safe(row.get("ma50_slope")),
        "ma200_slope": _safe(row.get("ma200_slope")),
        "ma5_above_ma15": _safe(row.get("ma5_above_ma15")),
        "ma15_above_ma25": _safe(row.get("ma15_above_ma25")),
        "ma25_above_ma50": _safe(row.get("ma25_above_ma50")),

        "return_1d": _safe(row.get("return_1d")), "return_3d": _safe(row.get("return_3d")),
        "return_5d": _safe(row.get("return_5d")), "return_10d": _safe(row.get("return_10d")),
        "return_20d": _safe(row.get("return_20d")),
        "rsi_7": _safe(row.get("rsi_7")), "rsi_14": _safe(row.get("rsi_14")),
        "macd": _safe(row.get("macd")), "macd_signal": _safe(row.get("macd_signal")),
        "macd_histogram": _safe(row.get("macd_hist")),
        "roc_5": _safe(row.get("roc_5")), "roc_10": _safe(row.get("roc_10")), "roc_20": _safe(row.get("roc_20")),

        "highest_close_10d": _safe(row.get("highest_close_10d")),
        "highest_close_20d": _safe(row.get("highest_close_20d")),
        "highest_high_20d": _safe(row.get("highest_high_20d")),
        "breakout_close_10d_pct": _safe(row.get("breakout_close_10d_pct")),
        "breakout_close_20d_pct": _safe(row.get("breakout_close_20d_pct")),
        "distance_from_20d_high_pct": _safe(row.get("distance_from_20d_high_pct")),
        "days_since_20d_high": _safe(row.get("days_since_20d_high")),
        "new_high_20d_flag": 1 if row.get("new_20d_high") else 0,
        "new_high_60d_flag": _safe(row.get("new_high_60d_flag")),
        "new_high_252d_flag": _safe(row.get("new_high_252d_flag")),

        "atr_14": _safe(row.get("atr14")), "atr_pct": _safe(row.get("atr_pct")),
        "historical_volatility_10d": _safe(row.get("historical_volatility_10d")),
        "historical_volatility_20d": _safe(row.get("historical_volatility_20d")),
        "bollinger_band_width": _safe(row.get("bollinger_band_width")),
        "distance_upper_band_pct": _safe(row.get("distance_upper_band_pct")),
        "gap_pct": _safe(row.get("gap_pct")), "average_gap_20d": _safe(row.get("average_gap_20d")),
        "beta": None,
        "beta_available": 0,  # SPYとの回帰計算はPhase2以降に持ち越しのため仕様上常に取得不能

        "open": _safe(row.get("open")), "high": _safe(row.get("high")),
        "low": _safe(row.get("low")), "close": _safe(row.get("close")),
        "body_pct": _safe(row.get("body_pct")), "upper_wick_pct": _safe(row.get("upper_wick_pct")),
        "lower_wick_pct": _safe(row.get("lower_wick_pct")),
        "close_location_value": _safe(row.get("close_location_value")),
        "inside_bar_flag": _safe(row.get("inside_bar_flag")), "outside_bar_flag": _safe(row.get("outside_bar_flag")),

        **market_features,

        "sector_etf": sector_features.get("sector_etf"),
        "sector_return_5d": sector_features.get("sector_return_5d"),
        "sector_return_20d": sector_features.get("sector_return_20d"),
        "stock_vs_sector_5d": stock_vs_sector_5d,
        "stock_vs_sector_20d": stock_vs_sector_20d,
        "sector_rank": None,  # 候補ユニバース全体を集計する段階(build_feature_store.py)で付与する

        "technical_score_v1": tech["subtotal"],
        "volume_condition": 1 if breakdown.get("volume_surge") else 0,
        "ma_condition": 1 if breakdown.get("ma5_gt_ma15") else 0,
        "ma25_condition": 1 if breakdown.get("ma25_breakout") else 0,
        "breakout_condition": 1 if breakdown.get("new_20d_high") else 0,
        "rsi_condition": 1 if breakdown.get("rsi_50_70") else 0,
    }
    return feature
