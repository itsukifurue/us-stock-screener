"""backtest.portfolio.simulate_portfolio の整合性監査用ユニットテスト。

人工的な価格データ・トレードを使い、手計算で期待できるNAV(資産評価額)の推移と
プログラムの出力が一致することを確認する。あわせて、独立実装
(backtest/portfolio_reference.py)とも突き合わせる。

実行:
    python -m unittest tests.test_portfolio -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from backtest.portfolio import simulate_portfolio
from backtest.portfolio_reference import simulate_portfolio_reference


def make_price_frame(dates: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(dates), "close": closes})


def make_trade(symbol, entry_date, exit_date, entry_price, exit_price, signal_score=50) -> dict:
    pnl_pct = (exit_price - entry_price) / entry_price * 100
    return {
        "symbol": symbol,
        "signal_date": entry_date,
        "signal_score": signal_score,
        "entry_date": entry_date,
        "entry_price": entry_price,
        "exit_date": exit_date,
        "exit_price": exit_price,
        "pnl_pct": round(pnl_pct, 4),
        "holding_days": 1,
        "outcome": "win" if exit_price > entry_price else "loss",
    }


class TestPortfolioAudit(unittest.TestCase):
    def test_a_single_winning_trade(self):
        """1回だけ利益になるケース: 100→110で+10%。全額投資・手数料なしなのでNAVもそのまま+10%。"""
        trades = [make_trade("WIN", "2024-01-02", "2024-01-03", 100.0, 110.0)]
        price_frames = {"WIN": make_price_frame(["2024-01-02", "2024-01-03"], [100.0, 110.0])}

        result = simulate_portfolio(trades, price_frames, max_concurrent=1, initial_capital=1.0, commission_pct=0.0)

        # 手計算: Day1(エントリー日)は値動きゼロ(closeもentry_priceと同じ100)なのでNAV=1.0のまま。
        # Day2(エグジット日)にexit_price=110で手仕舞い → NAV=1.0*1.10=1.10
        expected_nav = [1.0, 1.0, 1.10]
        for exp, actual in zip(expected_nav, result["nav_series"]):
            self.assertAlmostEqual(exp, actual, places=6)
        self.assertEqual(result["trades_taken"], 1)
        self.assertEqual(result["trades_skipped"], 0)

    def test_b_single_losing_trade(self):
        """1回だけ損失になるケース: 100→94で-6%。"""
        trades = [make_trade("LOSE", "2024-01-02", "2024-01-03", 100.0, 94.0)]
        price_frames = {"LOSE": make_price_frame(["2024-01-02", "2024-01-03"], [100.0, 94.0])}

        result = simulate_portfolio(trades, price_frames, max_concurrent=1, initial_capital=1.0, commission_pct=0.0)

        expected_nav = [1.0, 1.0, 0.94]
        for exp, actual in zip(expected_nav, result["nav_series"]):
            self.assertAlmostEqual(exp, actual, places=6)

    def test_c_three_concurrent_positions(self):
        """3銘柄を同時保有するケース: 等金額(1/3ずつ)投資し、+10%/+10%/-10%の平均で+3.333%。"""
        trades = [
            make_trade("A", "2024-01-02", "2024-01-03", 100.0, 110.0, signal_score=70),
            make_trade("B", "2024-01-02", "2024-01-03", 50.0, 55.0, signal_score=60),
            make_trade("C", "2024-01-02", "2024-01-03", 200.0, 180.0, signal_score=50),
        ]
        # 各銘柄ともエントリー日の終値をentry_priceと同額(値動きゼロ)にしておくと、
        # 3件均等配分が厳密に1/3ずつになり手計算しやすい。
        price_frames = {
            "A": make_price_frame(["2024-01-02", "2024-01-03"], [100.0, 110.0]),
            "B": make_price_frame(["2024-01-02", "2024-01-03"], [50.0, 55.0]),
            "C": make_price_frame(["2024-01-02", "2024-01-03"], [200.0, 180.0]),
        }

        result = simulate_portfolio(trades, price_frames, max_concurrent=3, initial_capital=1.0, commission_pct=0.0)

        # 手計算: Day1は3銘柄とも値動きゼロなのでNAV=1.0のまま。
        # Day2decoded: (+10% + +10% + -10%) / 3 = +3.333...%
        expected_final = 1.0 * (1 + (0.10 + 0.10 - 0.10) / 3)
        self.assertAlmostEqual(result["nav_series"][0], 1.0, places=6)
        self.assertAlmostEqual(result["nav_series"][1], 1.0, places=6)
        self.assertAlmostEqual(result["nav_series"][2], expected_final, places=6)
        self.assertEqual(result["trades_taken"], 3)
        self.assertEqual(result["trades_skipped"], 0)

    def test_d_same_day_exit_and_reentry(self):
        """同日決済が発生するケース: Xが決済される同じ日にYが新規エントリーする(枠の再利用)。"""
        trades = [
            make_trade("X", "2024-01-02", "2024-01-03", 100.0, 105.0, signal_score=50),  # +5%
            make_trade("Y", "2024-01-03", "2024-01-04", 80.0, 88.0, signal_score=60),    # +10%
        ]
        price_frames = {
            "X": make_price_frame(["2024-01-02", "2024-01-03"], [100.0, 105.0]),
            "Y": make_price_frame(["2024-01-03", "2024-01-04"], [80.0, 88.0]),
        }

        result = simulate_portfolio(trades, price_frames, max_concurrent=1, initial_capital=1.0, commission_pct=0.0)

        # 手計算: Day1(X エントリー、値動きゼロ)→NAV=1.0
        # Day2(Xを決済して+5%確定 → 同日中にYへ全額再エントリー、Yも値動きゼロの日なのでNAV変化なし)→NAV=1.05
        # Day3(Yを決済して+10%)→NAV=1.05*1.10=1.155
        expected_nav = [1.0, 1.0, 1.05, 1.155]
        for exp, actual in zip(expected_nav, result["nav_series"]):
            self.assertAlmostEqual(exp, actual, places=6)
        self.assertEqual(result["trades_taken"], 2, "同日決済されたスロットが同日中に再利用されていない(バグの疑い)")
        self.assertEqual(result["trades_skipped"], 0)

    def test_e_skipped_when_no_free_slot(self):
        """現金/枠が不足して新規エントリーを見送るケース(max_concurrent=1で2件が完全に重複)。"""
        trades = [
            make_trade("P", "2024-01-02", "2024-01-05", 100.0, 110.0, signal_score=70),
            make_trade("Q", "2024-01-03", "2024-01-04", 50.0, 55.0, signal_score=60),  # Pの保有中に発生
        ]
        price_frames = {
            "P": make_price_frame(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"], [100.0, 102.0, 104.0, 110.0]
            ),
            "Q": make_price_frame(["2024-01-03", "2024-01-04"], [50.0, 55.0]),
        }

        result = simulate_portfolio(trades, price_frames, max_concurrent=1, initial_capital=1.0, commission_pct=0.0)

        self.assertEqual(result["trades_taken"], 1)
        self.assertEqual(result["trades_skipped"], 1, "枠がフルの時に新規エントリーを見送れていない")
        self.assertAlmostEqual(result["nav_series"][-1], 1.10, places=6, msg="見送られたQの損益がNAVに混入している")

    def test_f_same_day_round_trip_trade_does_not_lock_slot_forever(self):
        """エントリー当日中に決済が確定する(entry_date == exit_date)トレードが発生した場合、
        スロットが永久に埋まったままにならないことを確認する(実際に発生した重大バグの再発防止)。
        """
        trades = [
            make_trade("SAMEDAY", "2024-01-02", "2024-01-02", 100.0, 108.0, signal_score=70),  # 即日+8%で決済
            make_trade("NEXT", "2024-01-03", "2024-01-04", 50.0, 55.0, signal_score=60),
        ]
        price_frames = {
            "SAMEDAY": make_price_frame(["2024-01-02"], [108.0]),
            "NEXT": make_price_frame(["2024-01-03", "2024-01-04"], [50.0, 55.0]),
        }

        result = simulate_portfolio(trades, price_frames, max_concurrent=1, initial_capital=1.0, commission_pct=0.0)

        # SAMEDAYが即日決済されてスロットが解放されなければ、NEXTは永久に見送られてしまう。
        self.assertEqual(result["trades_taken"], 2, "同日決済トレードの後、スロットが解放されていない(バグの再発)")
        self.assertEqual(result["trades_skipped"], 0)
        # 手計算: Day1で+8%確定(1.08)、Day2は値動きなし、Day3で+10%確定(1.08*1.10=1.188)
        self.assertAlmostEqual(result["nav_series"][-1], 1.08 * 1.10, places=6)

    def test_e2_cash_clamped_when_existing_position_grows(self):
        """既存ポジションが大きく値上がりし、空き枠はあっても現金が目標配分額に届かず
        少額投資に制限される(現金不足によるクランプ)ケース。"""
        # Rが10倍に値上がりする間ずっと保有し続け、その途中でSに新規エントリーしようとすると
        # (nav_now/2という目標額に対して)現金が足りず、現金の範囲内(クランプ)でしか投資できない。
        dates_r = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        closes_r = [10.0, 100.0, 100.0, 100.0]
        trades = [
            make_trade("R", "2024-01-02", "2024-01-05", 10.0, 100.0, signal_score=70),
            make_trade("S", "2024-01-04", "2024-01-05", 20.0, 22.0, signal_score=60),
        ]
        price_frames = {
            "R": make_price_frame(dates_r, closes_r),
            "S": make_price_frame(["2024-01-04", "2024-01-05"], [20.0, 22.0]),
        }

        result = simulate_portfolio(trades, price_frames, max_concurrent=2, initial_capital=1.0, commission_pct=0.0)

        # Rにはmax_concurrent=2なので初日にnav(1.0)の半分=0.5を投資、残り現金0.5。
        # Rが10倍(10→100)に値上がりする間、現金0.5は使われず据え置かれる。
        # 2024-01-04時点のnav_now = cash(0.5) + R評価額(0.05株×100=5.0) = 5.5
        # 目標配分 = 5.5/2 = 2.75 だが、現金は0.5しか無いので投資額は0.5にクランプされるはず。
        entry_logs = [e for e in result["log"] if e["action"] == "entry" and e["symbol"] == "S"]
        self.assertEqual(len(entry_logs), 1, "Sへの新規エントリーが記録されていない")
        self.assertAlmostEqual(entry_logs[0]["invest_amount"], 0.5, places=6,
                                msg="現金不足によるクランプが働いていない(目標配分額をそのまま投資してしまっている)")

    def test_cross_check_against_reference_implementation(self):
        """独立に実装したportfolio_reference.pyと、上記の複雑なケース(同日決済+クランプ)で
        NAV系列が一致することを確認する。"""
        trades = [
            make_trade("X", "2024-01-02", "2024-01-03", 100.0, 105.0, signal_score=50),
            make_trade("Y", "2024-01-03", "2024-01-04", 80.0, 88.0, signal_score=60),
            make_trade("R", "2024-01-02", "2024-01-06", 10.0, 100.0, signal_score=70),
            make_trade("S", "2024-01-04", "2024-01-06", 20.0, 22.0, signal_score=40),
        ]
        price_frames = {
            "X": make_price_frame(["2024-01-02", "2024-01-03"], [100.0, 105.0]),
            "Y": make_price_frame(["2024-01-03", "2024-01-04"], [80.0, 88.0]),
            "R": make_price_frame(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"],
                [10.0, 50.0, 100.0, 100.0, 100.0],
            ),
            "S": make_price_frame(["2024-01-04", "2024-01-05", "2024-01-06"], [20.0, 21.0, 22.0]),
        }

        result_main = simulate_portfolio(trades, price_frames, max_concurrent=2, initial_capital=1.0, commission_pct=0.1)
        result_ref = simulate_portfolio_reference(trades, price_frames, max_concurrent=2, initial_capital=1.0, commission_pct=0.1)

        self.assertEqual(len(result_main["nav_series"]), len(result_ref["nav_series"]))
        for a, b in zip(result_main["nav_series"], result_ref["nav_series"]):
            self.assertAlmostEqual(a, b, places=6, msg="独立実装とNAV系列が一致しない")


if __name__ == "__main__":
    unittest.main()
