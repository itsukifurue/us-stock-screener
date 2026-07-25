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

    -- 既存ルール情報(Version1のスコアとの比較用。Step2以降は銘柄選択の主ルールとしては
    -- 使わず、比較対象の1特徴量として保存するのみ。要件通り破棄しない)
    technical_score_v1 REAL,
    volume_condition INTEGER, ma_condition INTEGER, ma25_condition INTEGER,
    breakout_condition INTEGER, rsi_condition INTEGER,

    -- Phase2 Step2: 3層候補定義(daily_universe/eligible_universe/signal_candidates)を
    -- featuresテーブル上のフラグとして表現する(物理的に別テーブルへ分割はしない)。
    -- eligible_flag は既存の candidate_flag と同じ値(一次スクリーニング条件通過)を指すが、
    -- Step2の用語に合わせて列名を追加した(candidate_flagは後方互換のため残す)。
    eligible_flag INTEGER,            -- 一次スクリーニング条件(price/market_cap/avg_volume)通過(=candidate_flagと同値)
    signal_v1_flag INTEGER,           -- technical_score_v1>=45(Version1のシグナル条件)を満たすか
    signal_reason TEXT,               -- 例: "score>=45" / "score<45(35.0)"
    cross_section_rank REAL,          -- その日のeligible_universe内でのtechnical_score_v1順位(1が最良、比較参考用)

    -- Phase2 Step2: 市場横断percentile特徴量(その営業日に実在するeligible_universeの銘柄だけを
    -- 使って計算する。未来日・全期間ランキングは一切使わない。0=最低、1=最高)
    return_1d_pctrank REAL,
    return_5d_pctrank REAL,
    volume_ratio_5d_pctrank REAL,
    atr_pct_pctrank REAL,
    rsi_14_pctrank REAL,
    breakout_close_20d_pct_pctrank REAL,
    dollar_volume_pctrank REAL,
    market_cap_pctrank REAL,
    sector_return_5d_pctrank REAL,       -- 同一セクター内でのreturn_5dのpercentile
    sector_volume_ratio_5d_pctrank REAL, -- 同一セクター内でのvolume_ratio_5dのpercentile
    cross_section_universe_size INTEGER, -- 上記percentile計算時のランキング対象銘柄数(市場全体)
    sector_universe_size INTEGER,        -- セクター内percentile計算時のランキング対象銘柄数

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
    target_trade_pnl_pct REAL,  -- 上記トレードの実損益%(スリッページ・手数料込み)。PF/期待値の実測計算用

    -- Phase2 Step2追加: target_15pct_within_10dと同じ定義の別閾値版(hit_plus_*pct_10dの別名。
    -- Phase3での命名一貫性のために明示的な列として持たせる)
    target_5pct_within_10d INTEGER,
    target_10pct_within_10d INTEGER,
    days_to_target INTEGER,     -- target_15pct_within_10d到達までの営業日数(未到達/未確定はNULL)
    exit_reason TEXT,           -- target_trade_success算出に使ったトレードのexit_reason
                                 -- ('stop_loss'/'take_profit'/'holding_period_limit'/'data_end'/NULL)

    -- label_status: 'confirmed'(将来ウィンドウが十分にあり確定)/'pending'(直近すぎてまだ確定できない、
    -- データが増えれば将来確定し得る)/'data_end'(銘柄側のデータがそこで終了しており今後も確定しない
    -- 可能性が高い=上場廃止等)/'invalid'(翌営業日データが無い等、構造的に計算不能)。
    -- pending/data_endを失敗(0)として扱ってはならない。
    label_status TEXT,

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

-- ================= Phase 2: 2層ユニバース設計 =================

-- ビルド実行1回分のメタ情報(再現性のため対象銘柄・期間・バージョンを記録する)
CREATE TABLE IF NOT EXISTS build_runs (
    build_run_id TEXT PRIMARY KEY,
    run_at TEXT,
    symbols TEXT,             -- カンマ区切り
    start_date TEXT,
    end_date TEXT,
    universe_version TEXT,
    feature_version TEXT,
    label_version TEXT,
    train_start TEXT, train_end TEXT,
    val_start TEXT, val_end TEXT,
    test_start TEXT, test_end TEXT,
    notes TEXT
);

-- 各ビルドで対象とした銘柄と、その銘柄がデータ取得できた実際の期間(上場日/データ開始日の制約を含む)
CREATE TABLE IF NOT EXISTS universe_membership (
    ticker TEXT NOT NULL,
    build_run_id TEXT NOT NULL,
    first_available_date TEXT,   -- データソース上で取得できた最初の日付(上場日の近似)
    last_available_date TEXT,
    sector TEXT,
    exchange TEXT,
    data_fetch_status TEXT,      -- 'ok' / 'failed' / 'insufficient_history'
    PRIMARY KEY (ticker, build_run_id)
);

-- Daily universe layer: 対象銘柄・対象期間の全営業日(候補にならなかった日も含む)の軽量スナップショット。
-- featuresテーブル(全指標入り)より軽量にし、将来Step2/3で対象銘柄数が増えた際にも
-- 全銘柄×全営業日を安価に把握できるようにする。
CREATE TABLE IF NOT EXISTS daily_universe (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    build_run_id TEXT NOT NULL,
    close REAL,
    volume REAL,
    dollar_volume REAL,
    universe_included_flag INTEGER,
    candidate_flag INTEGER,
    candidate_reason TEXT,
    universe_version TEXT,
    PRIMARY KEY (ticker, date, build_run_id)
);

-- Candidate snapshot layer: その営業日に「一次スクリーニング条件(price/market_cap/avg_volume)」を
-- 満たした銘柄=universe candidate のみを対象とする(注意: これはVersion1のシグナル条件
-- technical_score_v1>=45 を満たしたという意味ではない。一次スクリーニング条件を満たした
-- universe candidateの中に、スコア未達で一度もシグナル化されないものも大量に含まれる)。
-- Version1の採用ルール(スコア上位・最大同時保有数)を後から適用した結果も一緒に保存し、
-- 「候補全体」「実際に採用された候補」「枠不足/資金不足で見送られた候補」を正確に比較できるようにする。
CREATE TABLE IF NOT EXISTS candidate_snapshots (
    ticker TEXT NOT NULL,
    signal_date TEXT NOT NULL,
    build_run_id TEXT NOT NULL,
    candidate_source TEXT,
    candidate_rank REAL,             -- その日のuniverse candidate内でのtechnical_score_v1順位(1が最良)
    candidate_reason TEXT,
    technical_score_v1 REAL,
    signaled_flag INTEGER,           -- technical_score_v1>=45でVersion1シグナルとして信号化されたか
    selected_by_v1_flag INTEGER,     -- Version1の採用ルール(ポートフォリオシミュレーター)で実際に採用されたか
    selected_rank_v1 REAL,           -- 信号化された候補内での、採用ルールのランキングキー(score)順位
    selection_capacity INTEGER,      -- その日のmax_concurrent(空き枠数の目安、Version1既定は3)
    rejected_due_to_capacity_flag INTEGER,  -- 信号化されたが枠不足or資金不足で見送られたか(理由の合算)
    rejected_reason TEXT,             -- 見送り理由の内訳: NULL(見送りなし)/'no_slot'(3枠が全て埋まっていた)/
                                       -- 'cash_insufficient'(枠はあったが投資可能な現金が無かった)
    feature_version TEXT,
    universe_version TEXT,
    PRIMARY KEY (ticker, signal_date, build_run_id)
);

-- ================= Phase 2 Step2 =================

-- ティッカー変更管理(例: SQ→XYZ)。取得失敗を単純な"failed"で終わらせず、
-- 既知の改称であることを記録する。価格系列の自動接続は行わない(危険なため)。
CREATE TABLE IF NOT EXISTS ticker_aliases (
    old_ticker TEXT NOT NULL,
    new_ticker TEXT NOT NULL,
    effective_date TEXT,        -- 改称が発生したとされる日付(判明している場合)
    company_name TEXT,
    reason TEXT,                 -- 例: "ticker_rename" / "merger" / "spinoff"
    source TEXT,                 -- 情報源(例: "manual_research_2026-07-22")
    manual_override_flag INTEGER,  -- 1なら手動で確認・登録した情報(自動検出ではない)
    price_series_spliced_flag INTEGER,  -- 1なら新旧の価格系列を接続済み、0なら別系列のまま
    notes TEXT,
    PRIMARY KEY (old_ticker, new_ticker)
);

-- ビルドの再開可能性: 銘柄×工程(stage)単位で進捗を記録する。
-- 再実行時はstatus='completed'の(build_run_id, symbol, stage)をスキップし、
-- pending/failedのみ処理することで、途中停止からの再開を可能にする。
CREATE TABLE IF NOT EXISTS build_run_symbol_status (
    build_run_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    stage TEXT NOT NULL,        -- 'fetch_price' / 'compute_features' / 'compute_labels' / 'save_db'
    status TEXT NOT NULL,       -- 'pending' / 'running' / 'completed' / 'failed' / 'skipped'
    started_at TEXT,
    completed_at TEXT,
    rows_written INTEGER,
    error_type TEXT,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    PRIMARY KEY (build_run_id, symbol, stage)
);

-- 特徴量メタデータ(Version2 Feature Set v1のFeature Freeze対象。Phase3でのモデル比較時に
-- 各特徴量の意味・単位・計算式・取得元・NULL可否・導入バージョンを参照できるようにする)。
CREATE TABLE IF NOT EXISTS feature_metadata (
    feature_name TEXT PRIMARY KEY,
    description TEXT,
    unit TEXT,
    formula TEXT,
    provider TEXT,
    nullable INTEGER,        -- 1: NULLになり得る(ウォームアップ不足・仕様上取得不能等) / 0: 常に値を持つ
    introduced_version TEXT  -- 例: "phase1" / "phase2_step1" / "phase2_step2"
);

-- ================= Phase 3A: モデル実験の再現性記録 =================

CREATE TABLE IF NOT EXISTS model_experiments (
    experiment_id TEXT PRIMARY KEY,
    run_at TEXT,
    git_commit TEXT,
    feature_set_version TEXT,        -- 例: "v1"(Version2 Feature Set v1)
    universe_variant TEXT,            -- "A_eligible_universe" / "B_signal_v1_subset"
    target_variable TEXT,             -- "target_trade_success" / "target_15pct_within_10d"
    start_date TEXT, end_date TEXT,
    train_start TEXT, train_end TEXT, val_start TEXT, val_end TEXT, test_start TEXT, test_end TEXT,
    n_train INTEGER, n_val INTEGER,
    n_features INTEGER,
    feature_whitelist TEXT,           -- JSON配列(モデルへ渡した全列名)
    preprocessing_summary TEXT,       -- JSON(補完方式・カテゴリ処理・スケーリング等)
    model_type TEXT,                  -- "dummy" / "logreg_l2" / "logreg_l1" / "logreg_elasticnet"
    model_params TEXT,                -- JSON(C, l1_ratio, class_weight等)
    embargo_days INTEGER,
    n_folds INTEGER,
    random_seed INTEGER,
    library_versions TEXT,            -- JSON(sklearn/pandas/numpy等のバージョン)
    cooldown_days INTEGER,            -- 0=cooldownなし(全eligible営業日)、5/10=適用あり
    notes TEXT
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    split_name TEXT,          -- "fold1".."foldN" / "validation"
    metric_name TEXT,         -- "roc_auc" / "pr_auc" / "log_loss" / "brier" / "precision" / ...
    metric_value REAL,
    metric_detail TEXT,       -- JSON(top-K別内訳等、単一値に収まらない場合)
    FOREIGN KEY (experiment_id) REFERENCES model_experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS model_coefficients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    fold_name TEXT,           -- "fold1".."foldN" / "validation"(最終学習)
    feature_name TEXT,
    coefficient REAL,
    odds_ratio REAL,
    abs_rank INTEGER,
    FOREIGN KEY (experiment_id) REFERENCES model_experiments(experiment_id)
);

-- 価格キャッシュのメタデータ(実データは data/price_cache/ 配下に銘柄別ファイルで保存し、
-- Gitには含めない。ここにはメタ情報のみを記録する)。
CREATE TABLE IF NOT EXISTS price_cache_meta (
    symbol TEXT NOT NULL,
    provider TEXT NOT NULL,
    cache_version TEXT NOT NULL,   -- キャッシュのバージョン。形式変更時はこれを上げて新旧混在を防ぐ
    fetch_start TEXT,
    fetch_end TEXT,
    fetched_at TEXT,
    row_count INTEGER,
    adjustment_mode TEXT,          -- 例: "split_adjusted_no_dividend"(api/yfinance_clientの仕様固定)
    data_hash TEXT,                 -- キャッシュファイル内容のハッシュ(整合性確認用)
    provider_version TEXT,
    PRIMARY KEY (symbol, provider, cache_version)
);
