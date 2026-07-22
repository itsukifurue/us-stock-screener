"""環境変数・しきい値・スコア配点・API予算を一元管理する設定モジュール。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# ---------- API keys ----------
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
FMP_BASE_URL = os.getenv("FMP_BASE_URL", "https://financialmodelingprep.com/stable")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")

# ---------- API budget ----------
# 注意: FMP stable APIはquote/profile/historical-price-eodが1銘柄ずつしか取得できないため、
# 「差分キャッシュ」で減らせるのは1リクエストあたりのデータ量のみで、リクエスト回数そのものは
# 毎日ほぼ SCREENER_CANDIDATE_LIMIT に比例する(候補銘柄数を増やすと日次消費量も増える)。
MAX_DAILY_REQUESTS = int(os.getenv("MAX_DAILY_REQUESTS", "230"))

# ---------- paths ----------
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"
DB_PATH = DATA_DIR / "screener.db"

# ---------- 一次スクリーニング条件 ----------
SCREENER_CANDIDATE_LIMIT = int(os.getenv("SCREENER_CANDIDATE_LIMIT", "100"))
SCREEN_MIN_PRICE = 5.0
SCREEN_MIN_AVG_VOLUME = 500_000
SCREEN_MIN_MARKET_CAP = 100_000_000
SCREEN_COUNTRY = "US"

# ---------- パイプライン段階のサイズ ----------
TOP_N_TECHNICAL = 10  # テクニカルスコア上位、詳細分析(AI)に進める数
TOP_N_FINAL = 3        # 最終レポートに出す数

# 過去株価の取得範囲(52週高値計算などに必要な日数 + 余裕)
HISTORICAL_LOOKBACK_DAYS = 400

# ---------- スコア配点 ----------
# 要件定義書の例をそのまま採用(各項目の合計は120点になり「100点満点」という記載と
# 数値上は一致しない)。全条件を満たすボーナスを許容する設計とみなし、
# 最終スコアは SCORE_MAX で100点にキャップして表示する。
SCORE_WEIGHTS = {
    "volume_surge": 20,      # 出来高急増
    "ma5_gt_ma15": 15,       # 5MA > 15MA
    "ma25_breakout": 15,     # 25MA突破
    "new_20d_high": 15,      # 20日高値更新
    "rsi_50_70": 10,         # RSI 50〜70
    "news_positive": 15,     # ニュース好材料
    "earnings_good": 10,     # 決算良好
    "ai_evaluation": 20,     # AI評価
}
SCORE_MAX = 100

# ---------- エントリー/損切り/利確ヒューリスティック ----------
# MVPでは統計的裏付け(バックテスト)がまだ無いため、ATRベースの簡易ルールで算出する。
# バックテストエンジン実装後、この定数群と win_rate/expected_value の算出方法を差し替える想定。
STOP_LOSS_ATR_MULTIPLIER = 1.5
TAKE_PROFIT_1_ATR_MULTIPLIER = 1.0
TAKE_PROFIT_2_TARGET_RETURN = 0.15  # 目標: 2週間以内+15%

# ---------- ニュース/財務のキャッシュ有効期間 ----------
NEWS_CACHE_HOURS = 20  # 1日1回の実行を想定し、ほぼ「当日は再取得しない」相当
