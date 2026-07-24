"""feature_store.labels の境界条件ユニットテスト(Phase1追加要件)。

対象:
  1. 10営業日目にちょうど+15%へ到達
  2. 11営業日目に+15%へ到達(10日ラベルではカウントされない)
  3. 利確と損切りが同日に到達(stop_first優先の確認)
  4. 期間途中でデータが尽きる(上場廃止・データ末尾)場合、0/1で確定させずNoneを返す
  5. ATRが算出不能な場合、ATR依存ラベルはNoneになる
  6. 株式分割等による未調整ギャップを品質チェックが検知する(安全網の確認)
  7. 直近すぎて判定できないラベルは0ではなくNone(pending)になる

実行:
    python -m unittest tests.test_label_boundaries -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

import config
from analysis.technical import row_to_indicators
from feature_store.features import compute_feature_frame
from feature_store.labels import compute_labels_for_signal
from feature_store.quality_checks import run_quality_checks
from tests.test_feature_store import make_synthetic_history

SIGNAL_IDX = 230  # ma200等のウォームアップ(200行)を満たしつつ、後ろに十分な日数を残す位置


def _copy_history(history: list[dict]) -> list[dict]:
    return [dict(day) for day in history]


class TestLabelBoundaries(unittest.TestCase):
    def test_exact_plus15pct_on_day10_counts_for_10d_label(self):
        history = _copy_history(make_synthetic_history(n=260, seed=101))
        entry_idx = SIGNAL_IDX + 1
        entry_open = 100.0
        target = entry_open * 1.15  # ちょうど+15%

        history[entry_idx]["open"] = entry_open
        # offset1〜9(day1〜9): 未到達に保つ
        for offset in range(0, 9):
            idx = entry_idx + offset
            history[idx].update({"high": entry_open * 1.05, "low": entry_open * 0.97, "close": entry_open})
        # offset10(day10, idx=entry_idx+9): ちょうど到達
        history[entry_idx + 9].update({"high": target, "low": entry_open * 0.97, "close": entry_open})

        df = compute_feature_frame(history, min_rows=200)
        self.assertIsNotNone(df)
        label = compute_labels_for_signal(df, SIGNAL_IDX, "TEST")
        self.assertIsNotNone(label)
        self.assertEqual(label["hit_plus_15pct_10d"], 1, "ちょうど10営業日目の到達がカウントされていない")
        self.assertEqual(label["target_15pct_within_10d"], 1)
        self.assertEqual(label["days_to_plus_15pct"], 10)

    def test_plus15pct_on_day11_does_not_count_for_10d_label(self):
        history = _copy_history(make_synthetic_history(n=260, seed=102))
        entry_idx = SIGNAL_IDX + 1
        entry_open = 100.0
        target = entry_open * 1.15

        history[entry_idx]["open"] = entry_open
        # offset1〜10(day1〜10): 未到達に保つ
        for offset in range(0, 10):
            idx = entry_idx + offset
            history[idx].update({"high": entry_open * 1.05, "low": entry_open * 0.97, "close": entry_open})
        # offset11(day11, idx=entry_idx+10): ここで初めて到達
        history[entry_idx + 10].update({"high": target, "low": entry_open * 0.97, "close": entry_open})

        df = compute_feature_frame(history, min_rows=200)
        self.assertIsNotNone(df)
        label = compute_labels_for_signal(df, SIGNAL_IDX, "TEST")
        self.assertIsNotNone(label)
        self.assertEqual(label["hit_plus_15pct_10d"], 0, "11営業日目の到達が誤って10日ラベルにカウントされている")
        self.assertEqual(label["hit_plus_15pct_15d"], 1, "15日ラベルでは到達しているはず")
        self.assertEqual(label["days_to_plus_15pct"], 11)

    def test_stop_and_target_hit_same_day_prioritizes_stop(self):
        history = _copy_history(make_synthetic_history(n=260, seed=103))
        df0 = compute_feature_frame(history, min_rows=200)
        self.assertIsNotNone(df0)
        indicators = row_to_indicators(df0, SIGNAL_IDX)
        atr = indicators["atr14"]
        self.assertIsNotNone(atr)
        self.assertGreater(atr, 0)

        entry_idx = SIGNAL_IDX + 1
        entry_open_raw = float(df0.iloc[entry_idx]["open"])
        cost_buy = 1 + 0.2 / 100 + 0.1 / 100  # labels.pyのslippage_pct=0.2, commission_pct=0.1と一致させる
        entry_price = entry_open_raw * cost_buy
        stop_loss = entry_price - config.STOP_LOSS_ATR_MULTIPLIER * atr
        take_profit = entry_price * (1 + config.TAKE_PROFIT_2_TARGET_RETURN)

        # エントリー当日(offset1)に、損切り価格・利確価格の両方に触れさせる
        history[entry_idx].update({
            "open": entry_open_raw,
            "high": take_profit + 1.0,
            "low": stop_loss - 1.0,
            "close": entry_open_raw,
        })

        df1 = compute_feature_frame(history, min_rows=200)
        self.assertIsNotNone(df1)
        label = compute_labels_for_signal(df1, SIGNAL_IDX, "TEST")
        self.assertIsNotNone(label)
        self.assertEqual(
            label["target_trade_success"], 0,
            "同日に損切り・利確の両方へ到達した場合はstop_first(保守的)によりlossとなるべき",
        )

    def test_data_ends_before_holding_period_completes_returns_none_not_zero_or_one(self):
        """上場廃止・データ末尾のシミュレーション: 保有期間を使い切る前にデータが尽きた場合、
        target_trade_success/hit_plus_15pct_10dはNone(未確定)であるべきで、0/1に確定させてはならない。
        """
        full_history = _copy_history(make_synthetic_history(n=260, seed=104))
        df0 = compute_feature_frame(full_history, min_rows=200)
        self.assertIsNotNone(df0)
        indicators = row_to_indicators(df0, SIGNAL_IDX)
        atr = indicators["atr14"]
        entry_idx = SIGNAL_IDX + 1
        entry_open_raw = float(df0.iloc[entry_idx]["open"])
        cost_buy = 1 + 0.2 / 100 + 0.1 / 100
        entry_price = entry_open_raw * cost_buy
        stop_loss = entry_price - config.STOP_LOSS_ATR_MULTIPLIER * atr
        take_profit = entry_price * (1 + config.TAKE_PROFIT_2_TARGET_RETURN)
        margin = min(entry_price - stop_loss, take_profit - entry_price) * 0.3
        low_floor = stop_loss + margin
        high_ceiling = take_profit - margin

        # データが entry_idx から5日分(offset1〜5)しか存在しない状態を作る(上場廃止/データ末尾を模擬)。
        # max_holding_days=10なので、本来の保有期間を使い切る前にデータが尽きる。
        truncated_history = full_history[: entry_idx + 5]
        for i in range(entry_idx, len(truncated_history)):
            truncated_history[i].update({
                "open": entry_open_raw, "high": high_ceiling, "low": low_floor, "close": entry_open_raw,
            })

        df1 = compute_feature_frame(truncated_history, min_rows=200)
        self.assertIsNotNone(df1)
        label = compute_labels_for_signal(df1, SIGNAL_IDX, "TEST")
        self.assertIsNotNone(label)
        self.assertIsNone(
            label["target_trade_success"],
            "データが保有期間の途中で尽きたのに、target_trade_successが0/1に確定してしまっている",
        )
        self.assertIsNone(
            label["hit_stop_atr_1_5_before_plus_15"],
            "データが保有期間の途中で尽きたのに、hit_stop_atr_1_5_before_plus_15が0/1に確定してしまっている",
        )
        self.assertIsNone(
            label["hit_plus_15pct_10d"],
            "データ不足(ウィンドウ未完成)なのに、hit_plus_15pct_10dが0/1に確定してしまっている",
        )

    def test_atr_uncomputable_returns_none_for_atr_dependent_labels(self):
        history = _copy_history(make_synthetic_history(n=260, seed=105))
        df = compute_feature_frame(history, min_rows=200)
        self.assertIsNotNone(df)
        # ATR算出不能な状態を強制的に作る(実データでは出来高ゼロ日連続等で起こり得る)
        df.loc[df.index[SIGNAL_IDX], "atr14"] = 0.0

        label = compute_labels_for_signal(df, SIGNAL_IDX, "TEST")
        self.assertIsNotNone(label)
        self.assertIsNone(label["target_trade_success"], "ATR算出不能なのにtarget_trade_successが確定している")
        self.assertIsNone(
            label["hit_stop_atr_1_5_before_plus_15"],
            "ATR算出不能なのにhit_stop_atr_1_5_before_plus_15が確定している",
        )
        # ATRに依存しない価格到達型ラベルは影響を受けず、通常通り計算されるべき
        self.assertIn(label["hit_plus_15pct_10d"], (0, 1))

    def test_recent_undetermined_label_is_none_not_zero(self):
        """直近すぎて将来ウィンドウが確定できないデータは、0ではなくNone(pending)であるべき。"""
        history = _copy_history(make_synthetic_history(n=260, seed=106))
        df = compute_feature_frame(history, min_rows=200)
        self.assertIsNotNone(df)
        recent_idx = len(df) - 3  # 直近(翌営業日は存在するが、将来ウィンドウが全く足りない)
        label = compute_labels_for_signal(df, recent_idx, "TEST")
        self.assertIsNotNone(label)
        self.assertIsNone(label["hit_plus_15pct_10d"], "直近データなのにhit_plus_15pct_10dが確定してしまっている")
        self.assertIsNone(label["target_15pct_within_10d"], "直近データなのにtarget_15pct_within_10dが確定してしまっている")
        self.assertIsNone(label["target_trade_success"], "直近データなのにtarget_trade_successが確定してしまっている")
        self.assertEqual(label["label_status"], "pending", "as_of_date未指定時は安全側でpendingになるべき")

    def test_label_status_confirmed_when_full_window_available(self):
        """将来ウィンドウが十分にある通常のシグナルはlabel_status='confirmed'であるべき。"""
        history = _copy_history(make_synthetic_history(n=260, seed=107))
        df = compute_feature_frame(history, min_rows=200)
        self.assertIsNotNone(df)
        label = compute_labels_for_signal(df, SIGNAL_IDX, "TEST")
        self.assertIsNotNone(label)
        self.assertEqual(label["label_status"], "confirmed")
        self.assertIn(label["target_15pct_within_10d"], (0, 1))

    def test_label_status_pending_vs_data_end_via_as_of_date(self):
        """as_of_dateを指定した場合、直近すぎるだけ(pending)と、データが大きく手前で
        終わっている(data_end、上場廃止等を想定)を正しく区別できること。"""
        history = _copy_history(make_synthetic_history(n=260, seed=108))
        df = compute_feature_frame(history, min_rows=200)
        self.assertIsNotNone(df)
        recent_idx = len(df) - 3
        last_date = df.iloc[-1]["date"].strftime("%Y-%m-%d")

        # as_of_dateがデータの最終日とほぼ同じ(=単に直近すぎるだけ) → pending
        label_pending = compute_labels_for_signal(df, recent_idx, "TEST", as_of_date=last_date)
        self.assertEqual(label_pending["label_status"], "pending")

        # as_of_dateがデータの最終日よりずっと後(=その後データが更新されていない=上場廃止等) → data_end
        far_future = (df.iloc[-1]["date"] + pd.Timedelta(days=60)).strftime("%Y-%m-%d")
        label_data_end = compute_labels_for_signal(df, recent_idx, "TEST", as_of_date=far_future)
        self.assertEqual(label_data_end["label_status"], "data_end")
        self.assertIsNone(label_data_end["target_15pct_within_10d"], "data_endなのに0/1へ確定してしまっている")

    def test_unadjusted_split_like_gap_is_flagged_by_quality_check(self):
        """株式分割が未調整のまま混入した場合を模した大きなギャップを、品質チェックが検知すること。"""
        base_feature = {
            "ticker": "TEST", "signal_date": "2025-01-10", "data_collected_at": "2025-01-10T00:00:00+00:00",
            "close": 100.0, "volume": 1_000_000, "rsi_14": 50.0, "technical_score_v1": 40.0, "gap_pct": -50.0,
        }
        report = run_quality_checks([base_feature], [])
        gap_check = next(c for c in report["checks"] if c["check_name"] == "suspicious_split_gap")
        self.assertEqual(gap_check["status"], "warning", "未調整分割の疑いがある大きなギャップが検知されていない")


if __name__ == "__main__":
    unittest.main()
