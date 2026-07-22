# Feature Store Phase2 Step1 データ品質レポート (2026-07-22)

build_run_id: phase2_step1_2026-07-22_20e200c9

```
データ品質レポート(総合判定: warning)
特徴量レコード数: 54365  ラベルレコード数: 54365

[PASS] duplicate_ticker_signal_date: 重複なし
[PASS] future_dated_signal: 問題なし
[PASS] data_collected_at_before_signal_date: 問題なし
[PASS] missing_rate: 予期しない欠損(ウォームアップ済みのはずが欠損): {'volume': {'件数': 0, '率(%)': 0.0}, 'avg_volume_5d': {'件数': 0, '率(%)': 0.0}, 'avg_volume_20d': {'件数': 0, '率(%)': 0.0}, 'ma5': {'件数': 0, '率(%)': 0.0}, 'ma15': {'件数': 0, '率(%)': 0.0}, 'ma25': {'件数': 0, '率(%)': 0.0}, 'ma50': {'件数': 0, '率(%)': 0.0}, 'ma200': {'件数': 0, '率(%)': 0.0}, 'return_1d': {'件数': 0, '率(%)': 0.0}, 'return_5d': {'件数': 0, '率(%)': 0.0}, 'return_20d': {'件数': 0, '率(%)': 0.0}, 'rsi_14': {'件数': 0, '率(%)': 0.0}, 'macd': {'件数': 0, '率(%)': 0.0}, 'atr_14': {'件数': 0, '率(%)': 0.0}, 'historical_volatility_20d': {'件数': 0, '率(%)': 0.0}, 'bollinger_band_width': {'件数': 0, '率(%)': 0.0}, 'gap_pct': {'件数': 0, '率(%)': 0.0}, 'spy_close': {'件数': 0, '率(%)': 0.0}, 'sector_return_5d': {'件数': 0, '率(%)': 0.0}, 'technical_score_v1': {'件数': 0, '率(%)': 0.0}} / ウォームアップ期間中の欠損(件数、品質問題ではない): {'avg_volume_5d': 20, 'avg_volume_20d': 95, 'ma5': 20, 'ma15': 70, 'ma25': 120, 'ma50': 247, 'ma200': 1431, 'return_1d': 5, 'return_5d': 25, 'return_20d': 100, 'rsi_14': 70, 'atr_14': 65, 'historical_volatility_20d': 100, 'bollinger_band_width': 95, 'gap_pct': 5} / 仕様上常にNULLの列(欠損率には含めない、品質問題ではない): {'bid_ask_spread': 100.0, 'turnover_ratio': 100.0, 'beta': 100.0, 'ma20': 100.0}
[PASS] infinite_or_nan_values: 問題なし
[PASS] range_violations: []
[WARNING] zero_volume: 出来高0のレコードが1件
[PASS] non_positive_price: 問題なし
[WARNING] suspicious_split_gap: 分割未調整の疑いがあるギャップが6件
[PASS] label_undetermined_rate: {'target_15pct_within_10d': 0.0, 'target_trade_success': 0.1, 'future_return_10d': 0.0}
[WARNING] data_fetch_failures: 取得/計算に失敗した銘柄: ['SQ']
[PASS] distribution_shift_check: {'rsi_14': {'前半平均': 50.22, '後半平均': 52.08}, 'return_5d': {'前半平均': 0.12, '後半平均': 0.59}, 'atr_pct': {'前半平均': 4.46, '後半平均': 4.09}, 'technical_score_v1': {'前半平均': 21.09, '後半平均': 23.11}}
```
