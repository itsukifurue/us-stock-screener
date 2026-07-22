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
