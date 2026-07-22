-- 銘柄マスタ
CREATE TABLE IF NOT EXISTS stocks (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    exchange TEXT,
    sector TEXT,
    industry TEXT,
    country TEXT,
    is_etf INTEGER DEFAULT 0,
    market_cap REAL,
    updated_at TEXT
);

-- 過去株価(追記のみ・既存日付はスキップ)
CREATE TABLE IF NOT EXISTS daily_prices (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_daily_prices_symbol ON daily_prices(symbol);

-- FMPリクエスト回数トラッキング(日次予算ガード用)
CREATE TABLE IF NOT EXISTS api_usage (
    date TEXT PRIMARY KEY,
    request_count INTEGER NOT NULL DEFAULT 0
);

-- スクリーニング実行履歴
CREATE TABLE IF NOT EXISTS screening_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    candidate_count INTEGER,
    status TEXT DEFAULT 'running'
);

-- 各ステージの候補銘柄とスコア
CREATE TABLE IF NOT EXISTS candidates (
    run_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    stage TEXT NOT NULL,
    technical_score REAL,
    ai_score REAL,
    total_score REAL,
    rank INTEGER,
    PRIMARY KEY (run_id, symbol),
    FOREIGN KEY (run_id) REFERENCES screening_runs(id)
);

-- 最終分析結果(Top3レポートの元データ)
CREATE TABLE IF NOT EXISTS analysis_results (
    run_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    company_name TEXT,
    current_price REAL,
    entry_price REAL,
    stop_loss REAL,
    take_profit_1 REAL,
    take_profit_2 REAL,
    expected_return_pct REAL,
    win_rate REAL,
    expected_value REAL,
    ai_comment TEXT,
    news_summary TEXT,
    risk_factors TEXT,
    total_score REAL,
    created_at TEXT,
    PRIMARY KEY (run_id, symbol),
    FOREIGN KEY (run_id) REFERENCES screening_runs(id)
);

-- ニュースキャッシュ(API節約用)
CREATE TABLE IF NOT EXISTS news_cache (
    symbol TEXT NOT NULL,
    published_date TEXT,
    title TEXT,
    summary TEXT,
    url TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (symbol, url)
);

-- 財務諸表キャッシュ(API節約用)
CREATE TABLE IF NOT EXISTS financials_cache (
    symbol TEXT NOT NULL,
    period TEXT NOT NULL,
    statement_type TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    data_json TEXT NOT NULL,
    PRIMARY KEY (symbol, period, statement_type)
);

-- バックテスト結果(次フェーズで実装・今回はテーブルのみ用意)
CREATE TABLE IF NOT EXISTS backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    entry_date TEXT,
    entry_price REAL,
    exit_date TEXT,
    exit_price REAL,
    pnl_pct REAL,
    holding_days INTEGER,
    outcome TEXT,
    run_date TEXT
);
