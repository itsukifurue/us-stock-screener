"""FMP API + yfinance + Anthropic APIの疎通確認スクリプト。

実際のAPIキーを .env に設定してから実行する:
    python scripts/test_api_connection.py

無料プランでは company-screener・stock-list・news系・historical-price-eod/full(一部銘柄除く)が
使えないことを実キーで確認済み(2026-07-21)。過去株価だけはFMPではなくyfinance(無料・APIキー不要)
から取得する設計にしたため、このスクリプトもyfinanceの疎通を確認する。
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from db.database import Database
from api.fmp_client import FMPClient
from api.yfinance_client import fetch_historical_prices

TEST_SYMBOL = "AAPL"
TEST_SYMBOLS = ["AAPL", "MSFT"]
# FMPでは有料プラン限定だった小型株の例。yfinanceで取れることを確認する
SMALL_CAP_TEST_SYMBOL = "JOBY"


def check(name: str, fn) -> bool:
    try:
        result = fn()
        preview = str(result)[:200]
        print(f"[OK] {name}: {preview}")
        return True
    except Exception as e:
        print(f"[NG] {name}: {e}")
        return False


def main() -> None:
    if not config.FMP_API_KEY:
        print("FMP_API_KEY が .env に設定されていません。.env.example を .env にコピーして設定してください。")
        sys.exit(1)

    db = Database(config.DB_PATH)
    client = FMPClient(db)

    results = []
    results.append(check("1. most-actives (候補銘柄生成)", lambda: client.most_actives()[:3]))
    results.append(check("2. biggest-gainers (候補銘柄生成)", lambda: client.biggest_gainers()[:3]))
    results.append(check("3. Quote", lambda: client.batch_quote(TEST_SYMBOLS)))
    results.append(check("4. Company Profile", lambda: client.company_profile(TEST_SYMBOL)))
    results.append(check("5. Financial Statements", lambda: client.income_statement(TEST_SYMBOL, limit=1)))

    to_date = date.today().isoformat()
    from_date = (date.today() - timedelta(days=30)).isoformat()
    results.append(
        check("6. yfinance 過去株価(大型株AAPL)", lambda: fetch_historical_prices(TEST_SYMBOL, from_date, to_date)[-1:])
    )
    results.append(
        check(
            "7. yfinance 過去株価(小型株JOBY, FMPでは有料限定だった銘柄)",
            lambda: fetch_historical_prices(SMALL_CAP_TEST_SYMBOL, from_date, to_date)[-1:],
        )
    )

    print(f"\nFMP本日リクエスト使用数: {db.get_today_usage()} / {config.MAX_DAILY_REQUESTS}")

    if config.ANTHROPIC_API_KEY:
        try:
            from api.claude_client import ClaudeClient

            cc = ClaudeClient()
            reply = cc.ping()
            print(f"[OK] 8. Anthropic API: {reply[:200]}")
            results.append(True)
        except Exception as e:
            print(f"[NG] 8. Anthropic API: {e}")
            results.append(False)
    else:
        print("[SKIP] 8. Anthropic API: ANTHROPIC_API_KEY が未設定です。")

    db.close()

    if not all(results):
        print("\n一部のエンドポイントが失敗しました。")
        sys.exit(1)
    print("\n全ての疎通確認に成功しました。python main.py --dry-run → python main.py に進んでください。")


if __name__ == "__main__":
    main()
