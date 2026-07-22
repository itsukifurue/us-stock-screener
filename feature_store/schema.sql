-- Version 2 point-in-time feature store。data/feature_store.db に配置する。
-- 主キーは (ticker, signal_date)。features と labels は完全に別テーブル・別工程で生成し、
-- 特徴量生成時に未来情報が混入しないようにする(リーク防止のためのアーキテクチャ上の分離)。

CREATE TABLE IF NOT EXISTS features (
    ticker TEXT NOT NULL,
    signal_date TEXT NOT NULL,

    -- 識別情報
    exchange TEXT,
    sector TEXT,
    industry TEXT,
    market_cap REAL,
    price REAL,
    candidate_source TEXT,       -- 'most_actives' / 'biggest_gainers' / 'both' / 'approx_universe'
    data_collected_at TEXT,
    feature_version TEXT,

    -- ユニバース/候補区分(全営業日レコードの中から候補日だけを後から正確に抽出できるようにする)
    universe_included_flag INTEGER,   -- その日、銘柄が対象ユニバースに含まれていたか(1/0)
    candidate_flag INTEGER,           -- 一次スクリーニング条件を満たし候補となったか(1/0)
    candidate_reason TEXT,            -- 候補/非候補の理由(例: "ok" / "price<5" / "avg_volume<500000")
    universe_version TEXT,            -- ユニバース定義のバージョン(例: "phase1_fixed_watchlist_v1")

    -- 流動性情報
    volume REAL,
    avg_volume_5d REAL,
    avg_volume_20d REAL,
    avg_volume_25d REAL,
    volume_ratio_5d REAL,
    volume_ratio_20d REAL,
    dollar_volume REAL,
    turnover_ratio REAL,
    turnover_ratio_available INTEGER,  -- 0/1。仕様上取得不能な項目かどうかを明示する
    bid_ask_spread REAL,
    bid_ask_spread_available INTEGER,  -- 無料データ源では常に0(=取得不能)。0埋めではなくNULL+フラグで表現

    -- トレンド情報
    ma5 REAL, ma10 REAL, ma15 REAL, ma20 REAL, ma25 REAL, ma50 REAL, ma75 REAL, ma100 REAL, ma200 REAL,
    price_vs_ma5_pct REAL, price_vs_ma15_pct REAL, price_vs_ma25_pct REAL,
    price_vs_ma50_pct REAL, price_vs_ma200_pct REAL,
    ma5_slope REAL, ma15_slope REAL, ma25_slope REAL, ma50_slope REAL, ma200_slope REAL,
    ma5_above_ma15 INTEGER, ma15_above_ma25 INTEGER, ma25_above_ma50 INTEGER,

    -- モメンタム情報
    return_1d REAL, return_3d REAL, return_5d REAL, return_10d REAL, return_20d REAL,
    rsi_7 REAL, rsi_14 REAL, macd REAL, macd_signal REAL, macd_histogram REAL,
    roc_5 REAL, roc_10 REAL, roc_20 REAL,

    -- ブレイクアウト情報
    highest_close_10d REAL, highest_close_20d REAL, highest_high_20d REAL,
    breakout_close_10d_pct REAL, breakout_close_20d_pct REAL,
    distance_from_20d_high_pct REAL, days_since_20d_high INTEGER,
    new_high_20d_flag INTEGER, new_high_60d_flag INTEGER, new_high_252d_flag INTEGER,

    -- ボラティリティ情報
    atr_14 REAL, atr_pct REAL,
    historical_volatility_10d REAL, historical_volatility_20d REAL,
    bollinger_band_width REAL, distance_upper_band_pct REAL,
    gap_pct REAL, average_gap_20d REAL,
    beta REAL,
    beta_available INTEGER,     -- 無料データ源では常に0(=取得不能)。0埋めではなくNULL+フラグで表現

    -- ローソク足情報
    open REAL, high REAL, low REAL, close REAL,
    body_pct REAL, upper_wick_pct REAL, lower_wick_pct REAL, close_location_value REAL,
    inside_bar_flag INTEGER, outside_bar_flag INTEGER,

    -- 市場環境(SPY/QQQ/IWM/VIXベース)
    spy_close REAL, spy_return_1d REAL, spy_return_5d REAL,
    spy_above_ma20 INTEGER, spy_above_ma50 INTEGER, spy_above_ma200 INTEGER,
    spy_ma20_slope REAL, spy_ma200_slope REAL,
    qqq_return_5d REAL, iwm_return_5d REAL,
    vix_close REAL, vix_return_5d REAL,
    market_regime TEXT,          -- strong_bull/bull/neutral/bear/strong_bear (feature_store/market_regime.py参照)

    -- セクター相対強度
    sector_etf TEXT,
    sector_return_5d REAL, sector_return_20d REAL,
    stock_vs_sector_5d REAL, stock_vs_sector_20d REAL,
    sector_rank REAL,            -- その日の候補ユニバース内でのセクター相対強度順位(0〜1に正規化)

    -- 既存ルール情報(Version1のスコアとの比較用、要件通り破棄せず保存)
    technical_score_v1 REAL,
    volume_condition INTEGER, ma_condition INTEGER, ma25_condition INTEGER,
    breakout_condition INTEGER, rsi_condition INTEGER,

    PRIMARY KEY (ticker, signal_date)
);

-- ラベル(将来の結果)。features とは完全に別テーブル・別スクリプト(labels.py)で生成する。
CREATE TABLE IF NOT EXISTS labels (
    ticker TEXT NOT NULL,
    signal_date TEXT NOT NULL,

    future_return_1d REAL, future_return_3d REAL, future_return_5d REAL,
    future_return_10d REAL, future_return_15d REAL,
    future_max_return_5d REAL, future_max_return_10d REAL, future_max_return_15d REAL,
    future_min_return_5d REAL, future_min_return_10d REAL, future_min_return_15d REAL,

    hit_plus_5pct_10d INTEGER, hit_plus_10pct_10d INTEGER,
    hit_plus_15pct_10d INTEGER, hit_plus_15pct_15d INTEGER,
    hit_stop_atr_1_5_before_plus_15 INTEGER,

    days_to_plus_5pct INTEGER, days_to_plus_10pct INTEGER, days_to_plus_15pct INTEGER,

    -- 主要ラベル1: シグナル翌営業日始値基準で、10営業日以内に高値が+15%到達したか
    target_15pct_within_10d INTEGER,

    -- 主要ラベル2: 現実的な取引ルール(翌日始値エントリー・コスト込み・ATR損切り/利確)で
    -- 損切り到達前に利確到達したか(同日両到達は損切り優先の保守的判定)
    target_trade_success INTEGER,

    label_computed_at TEXT,
    label_version TEXT,

    PRIMARY KEY (ticker, signal_date)
);

-- データ品質チェックの結果ログ
CREATE TABLE IF NOT EXISTS data_quality_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT,
    check_name TEXT,
    status TEXT,          -- 'pass' / 'fail' / 'warning'
    affected_count INTEGER,
    details TEXT
);
