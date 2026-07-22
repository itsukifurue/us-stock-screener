"""バックテスト対象銘柄リスト。

FMPの無料プランでは「過去のある日の値動き上位ランキング」が取得できないため、
本番パイプライン(stage1)と同じ銘柄選定を過去にさかのぼって再現することはできない。
代わりに、セクターや時価総額規模の異なる銘柄を手動で選定した固定リストでバックテストする。
"""

DEFAULT_WATCHLIST = [
    # 大型テック
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "NFLX",
    # 決済・フィンテック
    "PYPL", "SQ", "SOFI", "COIN",
    # 自動車・輸送
    "F", "RIVN", "FWRD",
    # 消費財・通信
    "UTZ", "T",
    # 値動きが大きい銘柄・小型株
    "PLTR", "SNAP", "ROKU", "DKNG", "U", "SHOP", "MARA", "RIOT", "CLSK", "SOUN", "JOBY", "PATH",
]

# Version2 Phase2用に拡張したウォッチリスト。DEFAULT_WATCHLIST(Version1で使用、変更しない)を
# 全て含みつつ、以下を追加する:
#   - DEFAULT_WATCHLISTに無いセクター(ヘルスケア・金融・エネルギー・資本財・生活必需品・
#     REIT・公益)の大型株を追加し、セクターの偏りを減らす。
#   - 上場時期が大きく異なる銘柄(JNJ等の数十年上場 〜 ARM/CART/RDDT等の2023-2024年IPO)を
#     混在させ、ウォームアップ不足(直近上場)のケースをデータセットに含める。
PHASE2_EXTRA_SYMBOLS = [
    # ヘルスケア
    "JNJ", "UNH", "LLY",
    # 金融
    "JPM", "V",
    # エネルギー
    "XOM", "CVX",
    # 資本財・輸送
    "CAT", "BA",
    # 生活必需品
    "KO", "WMT",
    # REIT
    "O",
    # 公益
    "NEE",
    # 2023〜2024年IPO(上場時期が浅い銘柄。ウォームアップ不足ケースの確認用)
    "ARM", "CART", "RDDT",
]

PHASE2_WATCHLIST = DEFAULT_WATCHLIST + PHASE2_EXTRA_SYMBOLS
