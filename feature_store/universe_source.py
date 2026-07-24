"""Version2 Phase2 Step2: 200銘柄前後のユニバース候補リストと、選定パイプラインの各段階。

【重要な制約】FMP無料プランではS&P500/NASDAQ100等の指数構成銘柄を動的に取得できない
(company-screener/stock-listは402で拒否される。Phase1で確認済み)ため、ここでの
"universe source"は、著名な株価指数(S&P500・NASDAQ100等)の構成銘柄として広く知られている
銘柄・主要セクターETFの主要構成銘柄として広く知られている銘柄を手動でカテゴリ分けした
静的リストである。動的な指数構成銘柄APIではない。

このため、**現在時点で存在が確認できる銘柄だけを過去へ遡って使うことによる生存者バイアスが
残る**(上場廃止・指数除外・buyoutされた銘柄は含まれない)。この制約は生成物のREADME/レポートに
必ず明記すること。

選定パイプラインは以下の段階に分離する(各関数が1段階に対応):
  1. RAW_UNIVERSE_SOURCES (このモジュール) … カテゴリ別の候補シンボル(手動収集の静的リスト)
  2. normalize_symbol() … 表記ゆれの正規化(yfinance形式へ)
  3. is_likely_etf_or_fund() … ETF/ファンドらしき銘柄の除外(簡易ヒューリスティック)
  4. build_candidate_universe() … 1〜3を統合し、重複除去した候補銘柄一覧と除外理由一覧を返す
  5. データ取得可否確認は scripts/build_feature_store_phase2_step2.py 側で実際にfetchして判定する
     (このモジュールでは判定しない。ネットワークアクセスを分離するため)。
"""
from __future__ import annotations

from backtest.watchlist import PHASE2_WATCHLIST

# ================= カテゴリ別 手動収集シンボル(著名指数構成銘柄として広く知られているもの) =================
# 各カテゴリのコメントは「主に何の多様性を狙って追加したか」を示す(検証可能性のため)。

MEGA_LARGE_CAP_ADDL = [  # 既存46銘柄に無い、セクター分散のための大型株
    "ORCL", "CRM", "ADBE", "INTC", "IBM", "CSCO", "QCOM", "TXN", "AVGO",
    "MA", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW",
    "HD", "MCD", "NKE", "SBUX", "LOW", "TGT", "COST", "PG", "PEP", "CL", "MO", "PM",
    "LIN", "HON", "MMM", "GE", "RTX", "LMT", "NOC", "ADP", "INTU", "NOW",
    "AMAT", "LRCX", "KLAC", "MU",
]

HEALTHCARE_ADDL = [  # ヘルスケアセクターの多様性(大型製薬・医療機器・マネージドケア)
    "PFE", "MRK", "ABBV", "BMY", "GILD", "MDT", "DHR", "TMO", "CVS", "CI", "HUM",
    "ZTS", "REGN", "VRTX", "BIIB", "ISRG",
]

FINANCIALS_ADDL = [  # 金融セクターの多様性(銀行・保険)
    "PNC", "USB", "TFC", "COF", "AIG", "MET", "PRU", "TRV", "ALL", "PGR",
]

ENERGY_ADDL = [  # エネルギーセクター(既存46銘柄はXOM/CVXのみのため拡張)
    "SLB", "OXY", "DVN", "MRO", "HAL", "BKR", "VLO", "MPC", "PSX", "WMB", "KMI", "EOG", "COP",
]

INDUSTRIALS_ADDL = [  # 資本財セクターの多様性
    "UPS", "FDX", "DE", "EMR", "ETN", "ITW", "PH", "ROK", "DOV", "CMI", "PCAR",
]

MATERIALS = [  # 素材セクター(既存46銘柄に無いセクター)
    "APD", "ECL", "NUE", "STLD", "DOW", "LYB", "FCX", "NEM", "MOS", "CF", "ALB",
]

UTILITIES_ADDL = [  # 公益セクターの多様性(既存46銘柄はNEEのみ、低ボラティリティの代表)
    "DUK", "SO", "D", "AEP", "EXC", "XEL", "ED", "WEC", "ES",
]

REAL_ESTATE = [  # REITセクターの多様性(既存46銘柄はOのみ)
    "PLD", "AMT", "CCI", "EQIX", "SPG", "PSA", "AVB", "EQR", "DLR",
]

CONSUMER_STAPLES_ADDL = [  # 生活必需品セクターの多様性・低ボラティリティ
    "WBA", "KR", "SYY", "GIS", "K", "HSY", "STZ", "KMB", "CLX", "CHD", "HRL",
]

CONSUMER_DISCRETIONARY_MID = [  # 一般消費財の中型株・景気敏感株
    "RCL", "CCL", "NCLH", "MGM", "WYNN", "LVS", "YUM", "CMG", "DPZ",
    "ORLY", "AZO", "ROST", "TJX", "BBY", "EBAY", "ETSY",
]

MID_CAP_DIVERSE = [  # 中型株の多様性(複数セクター)
    "DECK", "ULTA", "CROX", "POOL", "GPC", "EXPD", "JBHT", "CHRW",
    "DAL", "UAL", "LUV", "HAS", "WHR", "KEY", "RF", "CFG", "HBAN", "ZION", "MTB", "IP", "PKG",
]

UNPROFITABLE_GROWTH = [  # 赤字企業・高成長株(テーマの偏りを避けるため複数業種から選定)
    "CVNA", "CHWY", "W", "PTON", "AFRM", "UPST", "OPEN", "FUBO",
]

RECENT_IPO_OR_SPINOFF = [  # 上場時期の短い銘柄(既存のARM/CART/RDDTに加えて拡張)
    "BIRK", "KVYO", "CAVA", "ONON", "GEHC", "KVUE", "VLTO", "SOLV",
]

RAW_UNIVERSE_SOURCES: dict[str, list[str]] = {
    "phase2_step1_existing": list(PHASE2_WATCHLIST),
    "mega_large_cap_addl": MEGA_LARGE_CAP_ADDL,
    "healthcare_addl": HEALTHCARE_ADDL,
    "financials_addl": FINANCIALS_ADDL,
    "energy_addl": ENERGY_ADDL,
    "industrials_addl": INDUSTRIALS_ADDL,
    "materials": MATERIALS,
    "utilities_addl": UTILITIES_ADDL,
    "real_estate": REAL_ESTATE,
    "consumer_staples_addl": CONSUMER_STAPLES_ADDL,
    "consumer_discretionary_mid": CONSUMER_DISCRETIONARY_MID,
    "mid_cap_diverse": MID_CAP_DIVERSE,
    "unprofitable_growth": UNPROFITABLE_GROWTH,
    "recent_ipo_or_spinoff": RECENT_IPO_OR_SPINOFF,
}

# yfinance形式への表記正規化(クラス株のドット表記など)
SYMBOL_NORMALIZATION_MAP = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
}

# 既知のETF/ファンド(誤って紛れ込んだ場合の除外用。今回のRAW_UNIVERSE_SOURCESには
# 意図的にETFを含めていないが、将来ソースを追加する際の安全網として残す)
KNOWN_ETF_SYMBOLS = {
    "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "XLK", "XLF", "XLV", "XLY", "XLP",
    "XLE", "XLI", "XLB", "XLRE", "XLU", "XLC",
}


def normalize_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    return SYMBOL_NORMALIZATION_MAP.get(s, s)


def is_likely_etf_or_fund(symbol: str) -> bool:
    return symbol in KNOWN_ETF_SYMBOLS


def build_candidate_universe() -> tuple[list[str], list[dict]]:
    """RAW_UNIVERSE_SOURCESを正規化・ETF除外・重複除去し、
    (最終候補シンボル一覧, 除外理由一覧)を返す。
    データ取得可否の確認(ネットワークアクセス)はここでは行わない。
    """
    seen: dict[str, str] = {}  # normalized symbol -> 採用元カテゴリ(最初に見つかったもの)
    excluded: list[dict] = []
    accepted: list[str] = []

    for category, symbols in RAW_UNIVERSE_SOURCES.items():
        for raw in symbols:
            norm = normalize_symbol(raw)
            if is_likely_etf_or_fund(norm):
                excluded.append({"symbol": raw, "normalized": norm, "category": category, "reason": "etf_or_fund"})
                continue
            if norm in seen:
                excluded.append({
                    "symbol": raw, "normalized": norm, "category": category,
                    "reason": f"duplicate(既に{seen[norm]}で採用済み)",
                })
                continue
            seen[norm] = category
            accepted.append(norm)

    return accepted, excluded
