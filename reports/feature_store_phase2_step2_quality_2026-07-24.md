# Feature Store Phase2 Step2 データ品質レポート (2026-07-24)

build_run_id: phase2_step2_universe_v2_1_2021-06-28_2026-06-27

## 選定除外銘柄

(なし)

## 取得失敗銘柄

[]

## エイリアス経由取得

['SQ->XYZ']

```
データ品質レポート(総合判定: warning)
特徴量レコード数: 0  ラベルレコード数: 0

[PASS] duplicate_ticker_signal_date: 重複なし
[PASS] future_dated_signal: 問題なし
[PASS] data_collected_at_before_signal_date: 問題なし
[PASS] missing_rate: 予期しない欠損(ウォームアップ済みのはずが欠損): {} / ウォームアップ期間中の欠損(件数、品質問題ではない): {} / 仕様上常にNULLの列(欠損率には含めない、品質問題ではない): {}
[PASS] infinite_or_nan_values: 問題なし
[PASS] range_violations: []
[PASS] zero_volume: 問題なし
[PASS] non_positive_price: 問題なし
[PASS] suspicious_split_gap: 問題なし
[PASS] label_undetermined_rate: {}
[PASS] data_fetch_failures: 問題なし
[PASS] missing_market_cap: market_cap欠損率: 0.0%
[PASS] missing_sector: sector欠損率: 0.0%
[PASS] missing_spy_close: market_data(spy_close)欠損率: 0.0%
[PASS] ticker_alias_usage: ティッカー変更(エイリアス)経由で取得した銘柄: 1件
[PASS] delisting_or_data_end_candidates: universe_membership記載の最終取得可能日一覧から、必要に応じてlabel_status='data_end'件数(上記label_status_breakdown参照)と突き合わせて確認する(275銘柄分の記録あり)
[WARNING] build_stage_failures: [{'symbol': 'K', 'stage': 'fetch_price', 'status': 'failed', 'error_type': 'empty_history'}, {'symbol': 'MRO', 'stage': 'fetch_price', 'status': 'failed', 'error_type': 'empty_history'}, {'symbol': 'WBA', 'stage': 'fetch_price', 'status': 'failed', 'error_type': 'empty_history'}]
[PASS] price_cache_meta_count: price_cache_metaレコード数: 249
[PASS] distribution_shift_check: {}
```
