# Feature Store データ品質レポート (2026-07-22)

```
データ品質レポート(総合判定: pass)
特徴量レコード数: 1255  ラベルレコード数: 1255

[PASS] duplicate_ticker_signal_date: 重複なし
[PASS] future_dated_signal: 問題なし
[PASS] data_collected_at_before_signal_date: 問題なし
[PASS] missing_rate: 予期しない欠損(ウォームアップ済みのはずが欠損): {'volume': {'件数': 0, '率(%)': 0.0}, 'avg_volume_5d': {'件数': 0, '率(%)': 0.0}, 'avg_volume_20d': {'件数': 0, '率(%)': 0.0}, 'ma5': {'件数': 0, '率(%)': 0.0}, 'ma15': {'件数': 0, '率(%)': 0.0}, 'ma25': {'件数': 0, '率(%)': 0.0}, 'ma50': {'件数': 0, '率(%)': 0.0}, 'ma200': {'件数': 0, '率(%)': 0.0}, 'return_1d': {'件数': 0, '率(%)': 0.0}, 'return_5d': {'件数': 0, '率(%)': 0.0}, 'return_20d': {'件数': 0, '率(%)': 0.0}, 'rsi_14': {'件数': 0, '率(%)': 0.0}, 'macd': {'件数': 0, '率(%)': 0.0}, 'atr_14': {'件数': 0, '率(%)': 0.0}, 'historical_volatility_20d': {'件数': 0, '率(%)': 0.0}, 'bollinger_band_width': {'件数': 0, '率(%)': 0.0}, 'gap_pct': {'件数': 0, '率(%)': 0.0}, 'spy_close': {'件数': 0, '率(%)': 0.0}, 'sector_return_5d': {'件数': 0, '率(%)': 0.0}, 'technical_score_v1': {'件数': 0, '率(%)': 0.0}} / ウォームアップ期間中の欠損(件数、品質問題ではない): {} / 仕様上常にNULLの列(欠損率には含めない、品質問題ではない): {'bid_ask_spread': 100.0, 'turnover_ratio': 100.0, 'beta': 100.0, 'ma20': 100.0}
[PASS] infinite_or_nan_values: 問題なし
[PASS] range_violations: []
[PASS] zero_volume: 問題なし
[PASS] non_positive_price: 問題なし
[PASS] suspicious_split_gap: 問題なし
[PASS] label_undetermined_rate: {'target_15pct_within_10d': 0.0, 'target_trade_success': 0.0, 'future_return_10d': 0.0}
[PASS] data_fetch_failures: 問題なし
[PASS] distribution_shift_check: {'rsi_14': {'前半平均': 56.76, '後半平均': 50.94}, 'return_5d': {'前半平均': 1.17, '後半平均': 0.66}, 'atr_pct': {'前半平均': 3.11, '後半平均': 3.42}, 'technical_score_v1': {'前半平均': 27.06, '後半平均': 19.77}}
```
