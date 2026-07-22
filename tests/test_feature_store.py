"""feature store(Version2 Phase1)のユニットテスト。

最重要: 未来情報リークテスト(test_no_future_leakage)。
「シグナル日以降のデータを追加しても、シグナル日時点の特徴量が変化しない」ことを検証する。
もし変化してしまえば、それは特徴量計算のどこかで未来のデータを参照していることを意味する。

実行:
    python -m unittest tests.test_feature_store -v
"""
from __future__ import annotations

import sys
import unittest
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_store.database import FeatureStoreDB
from feature_store.features import build_feature_row, compute_feature_frame
from feature_store.labels import compute_labels_for_signal

# 特徴量計算では使わないが build_feature_row の引数として必要な、市場/セクター特徴量のダミー値
DUMMY_MARKET_FEATURES = {
    "spy_close": None, "spy_return_1d": None, "spy_return_5d": None,
    "spy_above_ma20": None, "spy_above_ma50": None, "spy_above_ma200": None,
    "spy_ma20_slope": None, "spy_ma200_slope": None,
    "qqq_return_5d": None, "iwm_return_5d": None,
    "vix_close": None, "vix_return_5d": None, "market_regime": "neutral",
}
DUMMY_SECTOR_FEATURES = {"sector_etf": None, "sector_return_5d": None, "sector_return_20d": None}


def make_synthetic_history(n: int = 300, seed: int = 42) -> list[dict]:
    """ネットワーク不要の決定的な合成OHLCVデータを作る(乱高下・出来高変動を含む)。"""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp("2026-01-01"), periods=n)
    close = 100 + np.cumsum(rng.normal(0.05, 1.2, n))
    close = np.maximum(close, 5.0)  # 価格が非正にならないようにする
    high = close + np.abs(rng.normal(0.4, 0.3, n))
    low = close - np.abs(rng.normal(0.4, 0.3, n))
    open_ = close + rng.normal(0, 0.3, n)
    volume = rng.integers(300_000, 900_000, n).astype(float)

    return [
        {
            "date": d.strftime("%Y-%m-%d"),
            "open": float(o), "high": float(h), "low": float(l), "close": float(c), "volume": float(v),
        }
        for d, o, h, l, c, v in zip(dates, open_, high, low, close, volume)
    ]


class TestFeatureStoreLeakage(unittest.TestCase):
    def test_no_future_leakage(self):
        """特徴量は、シグナル日より後のデータを追加しても変化してはならない。"""
        full_history = make_synthetic_history(300)
        signal_idx_in_full = 250  # 十分な過去データ(ma200等)がある位置

        df_full = compute_feature_frame(full_history, min_rows=200)
        self.assertIsNotNone(df_full)

        # シグナル日までで打ち切ったデータだけを使う(未来の行が存在しない状態)
        truncated_history = full_history[: signal_idx_in_full + 1]
        df_truncated = compute_feature_frame(truncated_history, min_rows=200)
        self.assertIsNotNone(df_truncated)

        signal_date = df_full.iloc[signal_idx_in_full]["date"]
        self.assertEqual(df_truncated.iloc[-1]["date"], signal_date, "打ち切りデータの最終行がシグナル日と一致しない")

        row_full = build_feature_row(
            df_full, signal_idx_in_full, "TEST", "NASDAQ", "Technology", "Software", 1e10,
            "approx_universe", DUMMY_MARKET_FEATURES, DUMMY_SECTOR_FEATURES,
        )
        row_truncated = build_feature_row(
            df_truncated, len(df_truncated) - 1, "TEST", "NASDAQ", "Technology", "Software", 1e10,
            "approx_universe", DUMMY_MARKET_FEATURES, DUMMY_SECTOR_FEATURES,
        )

        # data_collected_at は実行時刻が入るため比較対象から除外する
        row_full.pop("data_collected_at", None)
        row_truncated.pop("data_collected_at", None)

        for key in row_full:
            v_full = row_full[key]
            v_trunc = row_truncated[key]
            if isinstance(v_full, float) and isinstance(v_trunc, float):
                self.assertAlmostEqual(
                    v_full, v_trunc, places=8,
                    msg=f"未来データ追加で特徴量'{key}'が変化した(リークの疑い): {v_full} != {v_trunc}",
                )
            else:
                self.assertEqual(
                    v_full, v_trunc,
                    msg=f"未来データ追加で特徴量'{key}'が変化した(リークの疑い): {v_full!r} != {v_trunc!r}",
                )

    def test_labels_require_future_data_only(self):
        """ラベルはシグナル日より後のデータが無ければNoneを返す(データ末尾付近で確認)。"""
        history = make_synthetic_history(300)
        df = compute_feature_frame(history, min_rows=200)
        label_at_end = compute_labels_for_signal(df, len(df) - 1, "TEST")
        self.assertIsNone(label_at_end, "翌営業日データが無いのにラベルが計算されてしまっている")

    def test_labels_undetermined_vs_not_reached_distinction(self):
        """将来データが全く無いケースでラベルが確定してしまわないか(境界值テスト)。"""
        history = make_synthetic_history(300)
        df = compute_feature_frame(history, min_rows=200)
        # 残り5日分しかデータが無い地点(15日先が必要なラベルは全てNoneになるべき)
        idx = len(df) - 6
        label = compute_labels_for_signal(df, idx, "TEST")
        self.assertIsNotNone(label)
        self.assertIsNone(label["hit_plus_15pct_15d"], "15日分のデータが無いのに hit_plus_15pct_15d が確定している")
        self.assertIsNone(label["future_return_15d"], "15日分のデータが無いのに future_return_15d が確定している")

    def test_feature_label_schema_separation(self):
        """features辞書にラベル由来のキーが、labels辞書に特徴量由来のキーが混入していないこと。"""
        history = make_synthetic_history(300)
        df = compute_feature_frame(history, min_rows=200)
        idx = 250
        feature = build_feature_row(
            df, idx, "TEST", "NASDAQ", "Technology", "Software", 1e10,
            "approx_universe", DUMMY_MARKET_FEATURES, DUMMY_SECTOR_FEATURES,
        )
        label = compute_labels_for_signal(df, idx, "TEST")

        forbidden_in_features = {"future_return_1d", "target_15pct_within_10d", "target_trade_success"}
        forbidden_in_labels = {"ma5", "rsi_14", "technical_score_v1", "atr_14"}

        self.assertTrue(forbidden_in_features.isdisjoint(feature.keys()), "featuresに未来ラベルが混入している")
        self.assertTrue(forbidden_in_labels.isdisjoint(label.keys()), "labelsに特徴量が混入している")

    def test_no_duplicate_keys_on_upsert(self):
        """同一(ticker, signal_date)への2回目のupsertが例外にならず、重複行を作らないこと。"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_feature_store.db"
            db = FeatureStoreDB(db_path)
            history = make_synthetic_history(300)
            df = compute_feature_frame(history, min_rows=200)
            feature = build_feature_row(
                df, 250, "TEST", "NASDAQ", "Technology", "Software", 1e10,
                "approx_universe", DUMMY_MARKET_FEATURES, DUMMY_SECTOR_FEATURES,
            )
            db.upsert_feature(feature)
            db.upsert_feature(feature)  # 2回目(重複)
            self.assertEqual(db.count_features(), 1, "同一キーのupsertで重複行ができてしまっている")
            db.close()


if __name__ == "__main__":
    unittest.main()
