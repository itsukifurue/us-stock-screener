"""Version2 Feature Set v1のFeature Freeze対応: featuresテーブルの全列のメタデータ。

各特徴量について 説明/単位/計算式/Provider/NULL可否/導入Version を記録し、
feature_metadataテーブルへ保存する。Phase3以降のモデル比較で「この特徴量は何か」を
コードを読まずに参照できるようにするためのドキュメントである。

実行:
    python -c "from feature_store.feature_metadata import seed_feature_metadata; from feature_store.database import FeatureStoreDB; import config; db=FeatureStoreDB(config.DATA_DIR/'feature_store.db'); seed_feature_metadata(db); db.close()"
または scripts/build_feature_store_phase2_step2.py の実行時に自動でseedされる。
"""
from __future__ import annotations

FEATURE_STORE_PROVIDER = "MarketDataProvider(yfinance経由)"
MARKET_PROVIDER = "MarketDataProvider(SPY/QQQ/IWM/VIX、yfinance経由)"
SECTOR_PROVIDER = "MarketDataProvider(SPDRセクターETF、yfinance経由)"
FUNDAMENTAL_PROVIDER = "FundamentalDataProvider(yfinance Ticker.info、非point-in-time)"
DERIVED = "features.py(自社計算、Providerからの生データを加工)"
RULE_V1 = "analysis.scoring.score_technical(Version1ルールベーススコア、比較用に保存)"
CROSS_SECTION = "features.compute_cross_sectional_percentiles(同日eligible_universe内で計算)"

# (feature_name, description, unit, formula, provider, nullable, introduced_version)
_METADATA: list[tuple[str, str, str, str, str, int, str]] = [
    ("ticker", "銘柄ティッカー", "-", "-", "-", 0, "phase1"),
    ("signal_date", "シグナル発生日(その日の終値までの情報で計算)", "date", "-", "-", 0, "phase1"),
    ("exchange", "上場取引所", "-", "-", FUNDAMENTAL_PROVIDER, 1, "phase1"),
    ("sector", "GICS類似セクター分類(yfinance、現在時点の分類を遡及適用・非point-in-time)", "-", "-", FUNDAMENTAL_PROVIDER, 1, "phase1"),
    ("industry", "業種(yfinance分類)", "-", "-", FUNDAMENTAL_PROVIDER, 1, "phase1"),
    ("market_cap", "時価総額(yfinance取得時点の値を遡及適用・非point-in-time)", "USD", "-", FUNDAMENTAL_PROVIDER, 1, "phase1"),
    ("price", "signal_date時点の終値", "USD", "close", FEATURE_STORE_PROVIDER, 0, "phase1"),
    ("candidate_source", "候補生成元ラベル", "-", "-", "-", 0, "phase1"),
    ("data_collected_at", "特徴量計算実行時刻(UTC ISO8601)", "datetime", "-", "-", 0, "phase1"),
    ("feature_version", "特徴量計算ロジックのバージョン文字列", "-", "-", "-", 0, "phase1"),
    ("universe_included_flag", "対象ウォッチリストに含まれる銘柄日か(1/0)", "flag", "-", DERIVED, 0, "phase1"),
    ("candidate_flag", "一次スクリーニング条件通過フラグ(=eligible_flagと同値)", "flag", "price>=5 and market_cap>=1e8 and avg_volume_20d>=5e5", DERIVED, 0, "phase1"),
    ("candidate_reason", "非候補の場合の理由文字列", "-", "-", DERIVED, 1, "phase1"),
    ("universe_version", "ユニバース定義のバージョン文字列", "-", "-", "-", 0, "phase1"),
    ("volume", "signal_date当日の出来高", "shares", "-", FEATURE_STORE_PROVIDER, 0, "phase1"),
    ("avg_volume_5d", "過去5営業日平均出来高", "shares", "volume.rolling(5).mean()", DERIVED, 1, "phase1"),
    ("avg_volume_20d", "過去20営業日平均出来高", "shares", "volume.rolling(20).mean()", DERIVED, 1, "phase1"),
    ("avg_volume_25d", "過去25営業日平均出来高", "shares", "volume.rolling(25).mean()", DERIVED, 1, "phase1"),
    ("volume_ratio_5d", "当日出来高 ÷ 過去5日平均出来高", "ratio", "volume / avg_volume_5d", DERIVED, 1, "phase1"),
    ("volume_ratio_20d", "当日出来高 ÷ 過去20日平均出来高", "ratio", "volume / avg_volume_20d", DERIVED, 1, "phase1"),
    ("dollar_volume", "当日ドル出来高(流動性の代理指標)", "USD", "close * volume", DERIVED, 0, "phase1"),
    ("turnover_ratio", "株式回転率(無料データ源では発行済株式数取得が不安定なため常にNULL)", "ratio", "N/A", "-", 1, "phase1"),
    ("turnover_ratio_available", "turnover_ratioが仕様上取得可能か(常に0)", "flag", "-", "-", 0, "phase1"),
    ("bid_ask_spread", "気配値スプレッド(無料データ源では取得不能のため常にNULL)", "USD", "N/A", "-", 1, "phase1"),
    ("bid_ask_spread_available", "bid_ask_spreadが仕様上取得可能か(常に0)", "flag", "-", "-", 0, "phase1"),
    ("ma5", "5日単純移動平均", "USD", "close.rolling(5).mean()", DERIVED, 1, "phase1"),
    ("ma10", "10日単純移動平均", "USD", "close.rolling(10).mean()", DERIVED, 1, "phase2_step1"),
    ("ma15", "15日単純移動平均", "USD", "close.rolling(15).mean()", DERIVED, 1, "phase1"),
    ("ma20", "20日単純移動平均(本ツールでは15/25日を採用するため常にNULL)", "USD", "N/A", "-", 1, "phase1"),
    ("ma25", "25日単純移動平均", "USD", "close.rolling(25).mean()", DERIVED, 1, "phase1"),
    ("ma50", "50日単純移動平均", "USD", "close.rolling(50).mean()", DERIVED, 1, "phase1"),
    ("ma75", "75日単純移動平均", "USD", "close.rolling(75).mean()", DERIVED, 1, "phase2_step1"),
    ("ma100", "100日単純移動平均", "USD", "close.rolling(100).mean()", DERIVED, 1, "phase2_step1"),
    ("ma200", "200日単純移動平均", "USD", "close.rolling(200).mean()", DERIVED, 1, "phase2_step1"),
    ("price_vs_ma5_pct", "終値のma5からの乖離率", "%", "(close-ma5)/ma5*100", DERIVED, 1, "phase1"),
    ("price_vs_ma15_pct", "終値のma15からの乖離率", "%", "(close-ma15)/ma15*100", DERIVED, 1, "phase1"),
    ("price_vs_ma25_pct", "終値のma25からの乖離率", "%", "(close-ma25)/ma25*100", DERIVED, 1, "phase1"),
    ("price_vs_ma50_pct", "終値のma50からの乖離率", "%", "(close-ma50)/ma50*100", DERIVED, 1, "phase1"),
    ("price_vs_ma200_pct", "終値のma200からの乖離率", "%", "(close-ma200)/ma200*100", DERIVED, 1, "phase2_step1"),
    ("ma5_slope", "ma5の5日前からの変化量", "USD", "ma5.diff(5)", DERIVED, 1, "phase1"),
    ("ma15_slope", "ma15の5日前からの変化量", "USD", "ma15.diff(5)", DERIVED, 1, "phase1"),
    ("ma25_slope", "ma25の5日前からの変化量", "USD", "ma25.diff(5)", DERIVED, 1, "phase1"),
    ("ma50_slope", "ma50の5日前からの変化量", "USD", "ma50.diff(5)", DERIVED, 1, "phase1"),
    ("ma200_slope", "ma200の5日前からの変化量", "USD", "ma200.diff(5)", DERIVED, 1, "phase2_step1"),
    ("ma5_above_ma15", "ma5がma15を上回っているか", "flag", "ma5 > ma15", DERIVED, 1, "phase1"),
    ("ma15_above_ma25", "ma15がma25を上回っているか", "flag", "ma15 > ma25", DERIVED, 1, "phase1"),
    ("ma25_above_ma50", "ma25がma50を上回っているか", "flag", "ma25 > ma50", DERIVED, 1, "phase1"),
    ("return_1d", "1日リターン", "%", "close.pct_change(1)*100", DERIVED, 1, "phase1"),
    ("return_3d", "3日リターン", "%", "close.pct_change(3)*100", DERIVED, 1, "phase1"),
    ("return_5d", "5日リターン", "%", "close.pct_change(5)*100", DERIVED, 1, "phase1"),
    ("return_10d", "10日リターン", "%", "close.pct_change(10)*100", DERIVED, 1, "phase1"),
    ("return_20d", "20日リターン", "%", "close.pct_change(20)*100", DERIVED, 1, "phase1"),
    ("rsi_7", "7日RSI(ウォームアップ不足時はNULL、中立値埋めなし)", "0-100", "Wilder RSI(period=7)", DERIVED, 1, "phase1"),
    ("rsi_14", "14日RSI(ウォームアップ不足時はNULL、中立値埋めなし)", "0-100", "Wilder RSI(period=14)", DERIVED, 1, "phase1"),
    ("macd", "MACD線(12,26)", "USD", "ema12-ema26", DERIVED, 1, "phase1"),
    ("macd_signal", "MACDシグナル線(9日EMA)", "USD", "macd.ewm(span=9).mean()", DERIVED, 1, "phase1"),
    ("macd_histogram", "MACDヒストグラム", "USD", "macd - macd_signal", DERIVED, 1, "phase1"),
    ("roc_5", "5日変化率(return_5dと同一定義)", "%", "close.pct_change(5)*100", DERIVED, 1, "phase1"),
    ("roc_10", "10日変化率", "%", "close.pct_change(10)*100", DERIVED, 1, "phase1"),
    ("roc_20", "20日変化率", "%", "close.pct_change(20)*100", DERIVED, 1, "phase1"),
    ("highest_close_10d", "過去10日終値の最高値", "USD", "close.rolling(10).max()", DERIVED, 1, "phase1"),
    ("highest_close_20d", "過去20日終値の最高値", "USD", "close.rolling(20).max()", DERIVED, 1, "phase1"),
    ("highest_high_20d", "過去20日高値の最高値", "USD", "high.rolling(20).max()", DERIVED, 1, "phase1"),
    ("breakout_close_10d_pct", "直近10日高値(当日除く)からのブレイクアウト率", "%", "(close-prev9_high)/prev9_high*100", DERIVED, 1, "phase1"),
    ("breakout_close_20d_pct", "直近20日高値(当日除く)からのブレイクアウト率(Version1のbreakout幅)", "%", "(close-prev19_high)/prev19_high*100", DERIVED, 1, "phase1"),
    ("distance_from_20d_high_pct", "20日高値からの下方乖離率", "%", "(close-highest_close_20d)/highest_close_20d*100", DERIVED, 1, "phase1"),
    ("days_since_20d_high", "20日高値からの経過営業日数", "days", "-", DERIVED, 1, "phase1"),
    ("new_high_20d_flag", "当日が20日高値更新か", "flag", "-", DERIVED, 0, "phase1"),
    ("new_high_60d_flag", "当日が60日高値更新か", "flag", "close >= close.rolling(60).max()", DERIVED, 1, "phase1"),
    ("new_high_252d_flag", "当日が252日(約1年)高値更新か", "flag", "close >= close.rolling(252, min_periods=1).max()", DERIVED, 0, "phase1"),
    ("atr_14", "14日ATR(Average True Range)", "USD", "Wilder ATR(period=14)", DERIVED, 1, "phase1"),
    ("atr_pct", "ATRの終値に対する比率(ボラティリティの代理指標)", "%", "atr_14/close*100", DERIVED, 1, "phase1"),
    ("historical_volatility_10d", "10日ヒストリカルボラティリティ(年率換算)", "%", "log_return.rolling(10).std()*sqrt(252)*100", DERIVED, 1, "phase1"),
    ("historical_volatility_20d", "20日ヒストリカルボラティリティ(年率換算)", "%", "log_return.rolling(20).std()*sqrt(252)*100", DERIVED, 1, "phase1"),
    ("bollinger_band_width", "ボリンジャーバンド幅", "%", "(bb_upper-bb_lower)/bb_mid*100", DERIVED, 1, "phase1"),
    ("distance_upper_band_pct", "終値のボリンジャー上限バンドからの乖離率", "%", "(close-bb_upper)/bb_upper*100", DERIVED, 1, "phase1"),
    ("gap_pct", "前日終値に対する当日始値ギャップ率", "%", "(open-prev_close)/prev_close*100", DERIVED, 1, "phase1"),
    ("average_gap_20d", "過去20日ギャップ率絶対値の平均", "%", "gap_pct.abs().rolling(20).mean()", DERIVED, 1, "phase1"),
    ("beta", "市場ベータ(SPYとの回帰。Phase2以降実装予定のため常にNULL)", "-", "N/A", "-", 1, "phase1"),
    ("beta_available", "betaが仕様上取得可能か(常に0)", "flag", "-", "-", 0, "phase1"),
    ("open", "当日始値(split調整済み、配当調整なし)", "USD", "-", FEATURE_STORE_PROVIDER, 0, "phase1"),
    ("high", "当日高値", "USD", "-", FEATURE_STORE_PROVIDER, 0, "phase1"),
    ("low", "当日安値", "USD", "-", FEATURE_STORE_PROVIDER, 0, "phase1"),
    ("close", "当日終値", "USD", "-", FEATURE_STORE_PROVIDER, 0, "phase1"),
    ("body_pct", "ローソク足の実体幅(高安レンジに対する比率)", "%", "abs(close-open)/(high-low)*100", DERIVED, 1, "phase1"),
    ("upper_wick_pct", "上ヒゲの長さ比率", "%", "(high-max(open,close))/(high-low)*100", DERIVED, 1, "phase1"),
    ("lower_wick_pct", "下ヒゲの長さ比率", "%", "(min(open,close)-low)/(high-low)*100", DERIVED, 1, "phase1"),
    ("close_location_value", "終値の高安レンジ内での位置(-1〜1)", "ratio", "((close-low)-(high-close))/(high-low)", DERIVED, 1, "phase1"),
    ("inside_bar_flag", "内包足(前日レンジに収まる)フラグ", "flag", "high<=prev_high and low>=prev_low", DERIVED, 1, "phase1"),
    ("outside_bar_flag", "包み足(前日レンジを包含)フラグ", "flag", "high>prev_high and low<prev_low", DERIVED, 1, "phase1"),
    ("spy_close", "SPY当日終値", "USD", "-", MARKET_PROVIDER, 1, "phase1"),
    ("spy_return_1d", "SPY 1日リターン", "%", "-", MARKET_PROVIDER, 1, "phase1"),
    ("spy_return_5d", "SPY 5日リターン", "%", "-", MARKET_PROVIDER, 1, "phase1"),
    ("spy_above_ma20", "SPYが20日線を上回っているか", "flag", "-", MARKET_PROVIDER, 1, "phase1"),
    ("spy_above_ma50", "SPYが50日線を上回っているか", "flag", "-", MARKET_PROVIDER, 1, "phase1"),
    ("spy_above_ma200", "SPYが200日線を上回っているか", "flag", "-", MARKET_PROVIDER, 1, "phase1"),
    ("spy_ma20_slope", "SPY 20日線の傾き", "USD", "-", MARKET_PROVIDER, 1, "phase1"),
    ("spy_ma200_slope", "SPY 200日線の傾き", "USD", "-", MARKET_PROVIDER, 1, "phase1"),
    ("qqq_return_5d", "QQQ 5日リターン", "%", "-", MARKET_PROVIDER, 1, "phase1"),
    ("iwm_return_5d", "IWM(小型株指数)5日リターン", "%", "-", MARKET_PROVIDER, 1, "phase1"),
    ("vix_close", "VIX当日終値", "index", "-", MARKET_PROVIDER, 1, "phase1"),
    ("vix_return_5d", "VIX 5日リターン", "%", "-", MARKET_PROVIDER, 1, "phase1"),
    ("market_regime", "市場環境分類(strong_bull/bull/neutral/bear/strong_bear)", "-", "feature_store.market_regime.classify_regime参照", DERIVED, 1, "phase1"),
    ("sector_etf", "対応するSPDRセクターETFティッカー", "-", "-", "-", 1, "phase1"),
    ("sector_return_5d", "セクターETFの5日リターン", "%", "-", SECTOR_PROVIDER, 1, "phase1"),
    ("sector_return_20d", "セクターETFの20日リターン", "%", "-", SECTOR_PROVIDER, 1, "phase1"),
    ("stock_vs_sector_5d", "個別銘柄5日リターン - セクター5日リターン", "%", "return_5d - sector_return_5d", DERIVED, 1, "phase1"),
    ("stock_vs_sector_20d", "個別銘柄20日リターン - セクター20日リターン", "%", "return_20d - sector_return_20d", DERIVED, 1, "phase1"),
    ("sector_rank", "同日候補内でのstock_vs_sector_20dの順位(0〜1正規化)", "0-1", "-", DERIVED, 1, "phase1"),
    ("technical_score_v1", "Version1ルールベーススコア(比較用、Step2以降は銘柄選択の主ルールとしては不採用)", "0-75", "analysis.scoring.score_technical参照", RULE_V1, 0, "phase1"),
    ("volume_condition", "Version1: 出来高急増条件", "flag", "-", RULE_V1, 0, "phase1"),
    ("ma_condition", "Version1: ma5>ma15条件", "flag", "-", RULE_V1, 0, "phase1"),
    ("ma25_condition", "Version1: ma25ブレイクアウト条件", "flag", "-", RULE_V1, 0, "phase1"),
    ("breakout_condition", "Version1: 20日高値更新条件", "flag", "-", RULE_V1, 0, "phase1"),
    ("rsi_condition", "Version1: RSI 50-70条件", "flag", "-", RULE_V1, 0, "phase1"),
    ("eligible_flag", "3層候補定義: 一次スクリーニング条件通過(=candidate_flagと同値)", "flag", "-", DERIVED, 0, "phase2_step2"),
    ("signal_v1_flag", "3層候補定義: technical_score_v1>=45(Version1シグナル条件)を満たすか", "flag", "score>=45", DERIVED, 0, "phase2_step2"),
    ("signal_reason", "signal_v1_flagの理由文字列(スコア値つき)", "-", "-", DERIVED, 0, "phase2_step2"),
    ("cross_section_rank", "同日eligible_universe内でのtechnical_score_v1順位(比較参考用)", "rank", "-", CROSS_SECTION, 1, "phase2_step2"),
    ("return_1d_pctrank", "同日eligible_universe内でのreturn_1dの市場横断percentile", "0-1", "-", CROSS_SECTION, 1, "phase2_step2"),
    ("return_5d_pctrank", "同日eligible_universe内でのreturn_5dの市場横断percentile", "0-1", "-", CROSS_SECTION, 1, "phase2_step2"),
    ("volume_ratio_5d_pctrank", "同日eligible_universe内でのvolume_ratio_5dの市場横断percentile", "0-1", "-", CROSS_SECTION, 1, "phase2_step2"),
    ("atr_pct_pctrank", "同日eligible_universe内でのatr_pctの市場横断percentile", "0-1", "-", CROSS_SECTION, 1, "phase2_step2"),
    ("rsi_14_pctrank", "同日eligible_universe内でのrsi_14の市場横断percentile", "0-1", "-", CROSS_SECTION, 1, "phase2_step2"),
    ("breakout_close_20d_pct_pctrank", "同日eligible_universe内でのbreakout_close_20d_pctの市場横断percentile", "0-1", "-", CROSS_SECTION, 1, "phase2_step2"),
    ("dollar_volume_pctrank", "同日eligible_universe内でのdollar_volumeの市場横断percentile", "0-1", "-", CROSS_SECTION, 1, "phase2_step2"),
    ("market_cap_pctrank", "同日eligible_universe内でのmarket_capの市場横断percentile", "0-1", "-", CROSS_SECTION, 1, "phase2_step2"),
    ("sector_return_5d_pctrank", "同一セクター内でのreturn_5dのpercentile", "0-1", "-", CROSS_SECTION, 1, "phase2_step2"),
    ("sector_volume_ratio_5d_pctrank", "同一セクター内でのvolume_ratio_5dのpercentile", "0-1", "-", CROSS_SECTION, 1, "phase2_step2"),
    ("cross_section_universe_size", "市場横断percentile計算時のランキング対象銘柄数", "count", "-", DERIVED, 1, "phase2_step2"),
    ("sector_universe_size", "セクター内percentile計算時のランキング対象銘柄数", "count", "-", DERIVED, 1, "phase2_step2"),
]


def get_feature_metadata_rows() -> list[dict]:
    return [
        {
            "feature_name": name, "description": desc, "unit": unit, "formula": formula,
            "provider": provider, "nullable": nullable, "introduced_version": version,
        }
        for name, desc, unit, formula, provider, nullable, version in _METADATA
    ]


def seed_feature_metadata(db) -> None:
    db.upsert_feature_metadata_bulk(get_feature_metadata_rows())
