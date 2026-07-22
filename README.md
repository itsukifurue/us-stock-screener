# 米国株AIスクリーニングツール (MVP)

米国株市場全体から、2週間以内に+15%の上昇が期待できる銘柄を毎日スクリーニングし、
テクニカル分析 + Claude(Anthropic API)によるAI分析でTop3銘柄をレポートするツール。

## Version 1 の結論(ルールベース・タグ: `v1-rule-based-backtest`)

徹底したバックテスト監査(エンジン・ポートフォリオシミュレーター・独立検証実装・ユニット
テスト・コスト分解・アブレーション・採用ルール比較・ポジションサイズ比較)の結果、以下が確認された。

- 現実条件(翌営業日始値エントリー・スリッページ0.2%・手数料0.1%)では、**全候補ベースでPF 0.979**
- **期待値 -0.078%/トレード**(全候補ベース)
- ポートフォリオに**実際に採用されたトレードはPF 0.828**(候補全体より明確に悪い = 逆選択)
- 現行の「テクニカルスコア順」採用ルールでは、空き枠が埋まった際に**逆選択**が発生している
  (採用済みトレードの成績が、候補全体の平均より悪い)
- NAVの1/3を1銘柄へ配分する(最大3銘柄同時保有)方式は、この逆選択・薄い優位性を**複利で大きく増幅し**、
  損失を悪化させている
- ATRベースの固定リスク0.5%/トレードのポジションサイズが最も損失を抑えたが、**それでもCAGRはマイナス**
- 以上より、**現行のスコアベース採用ルールのまま実弾運用は行わない**
- Version 1 は「失敗」ではなく、**厳密な検証基盤(バックテストエンジン・監査済みポートフォリオ
  シミュレーター・ユニットテスト一式)が完成した版**として保存する。Version 2 ではこの基盤を使い、
  スコアの微調整ではなく、特徴量とラベルを大量蓄積した上での統計的・機械学習的なランキングモデル
  構築に方針転換する。

詳細は `reports/backtest_decomposition_*.md` / `reports/cagr_decomposition_*.md` を参照。

FMP(Financial Modeling Prep)無料プラン(250リクエスト/日)を前提に、
「一次スクリーニング→絞り込み→詳細分析」の二段階方式でAPI消費を抑える設計。
過去株価のみyfinance(Yahoo Finance、無料・APIキー不要)を使う(理由は下記「設計上の注意点」参照)。

## セットアップ

1. Python 3.11以上を推奨(このリポジトリは仮想環境 `.venv` を同梱していないので各自作成する)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. `.env.example` を `.env` にコピーし、FMPとAnthropicのAPIキーを設定する

```bash
copy .env.example .env
```

`.env` の中身:

```
FMP_API_KEY=あなたのFMP APIキー
FMP_BASE_URL=https://financialmodelingprep.com/stable
ANTHROPIC_API_KEY=あなたのAnthropic APIキー
ANTHROPIC_MODEL=claude-sonnet-5
MAX_DAILY_REQUESTS=230
SCREENER_CANDIDATE_LIMIT=100
```

3. API疎通確認(実際にAPIを叩いて6エンドポイント+Anthropicが動くか確認する)

```bash
python scripts/test_api_connection.py
```

`[NG]` が出た場合は `api/fmp_client.py` の `ENDPOINTS` 辞書のパスを、実際に動くパスに修正する
(FMPは公式ドキュメントサイトが自動アクセスをブロックしており、事前に完全な検証ができなかったため)。

4. ロジックのみ確認したい場合(APIを一切呼ばない)

```bash
python main.py --dry-run
```

5. 本番実行

```bash
python main.py
```

`reports/YYYY-MM-DD.md` と `reports/YYYY-MM-DD.json` にTop3銘柄のレポートが出力される。

## アーキテクチャ

```
main.py                        # オーケストレーション + CLI
config.py                      # しきい値・配点・API予算の一元管理
db/
  schema.sql, database.py      # SQLite(銘柄・過去株価・スコア・分析結果を蓄積)
api/
  fmp_client.py                # FMP APIクライアント(予算ガード付き)
  yfinance_client.py           # Yahoo Financeから過去株価を取得(無料・APIキー不要)
  claude_client.py             # Anthropic APIクライアント(JSON構造化分析)
analysis/
  technical.py                 # SMA/RSI/MACD/ATR/ボリンジャー等(pandasで自前実装)
  scoring.py                   # テクニカルスコア+AIスコアの合算
pipeline/
  stage1_screener.py           # 値動き上位ランキング(most-actives/biggest-gainers) → 候補銘柄
  stage2_enrichment.py         # 過去株価取得(差分キャッシュ)→テクニカル計算→出来高フィルタ→Top10
  stage3_ai_ranking.py         # プロフィール/財務取得→ETF/時価総額/国フィルタ→Claude分析→Top3
report/
  report_generator.py          # Markdown/JSON日次レポート
```

## 設計上の注意点(MVP時点の既知の制約)

- **無料プランではCompany Screener/News系が使えない(実キーで確認済み、2026-07-21)**:
  `company-screener`・`stock-list`・`news/*` は無料プランだと402(Restricted Endpoint)になる。
  そのため候補銘柄の入口は要件定義書のCompany Screenerではなく、無料プランで動作確認済みの
  `most-actives`(出来高上位)・`biggest-gainers`(値上がり上位)ランキングに変更している。
  これらのレスポンスには出来高・時価総額が含まれないため、一次スクリーニング条件は
  パイプライン全体に分散して適用する: 株価フィルタ=stage1、出来高フィルタ=stage2(過去株価から
  算出したavg_volume20を使用)、ETF/時価総額/国フィルタ=stage3(プロフィール取得時)。
  ニュースが一切取得できないため、Claude分析はプロフィール・財務・テクニカル情報のみで行う
  (`news_positive`はほぼ常にfalseになる)。将来的に有料プランへ切り替えた場合に備えて
  `api/fmp_client.py`には`screen_stocks()`/`news()`メソッドをそのまま残してある。
- **過去株価はFMPではなくyfinanceを使う**: `historical-price-eod/full`はエンドポイント自体は
  200を返すが、実際には**銘柄ごとに**402(Special Endpoint)で拒否されることが判明した。
  most-actives/biggest-gainersで集めた候補56銘柄で検証したところ、AAPL・NVDA・Tなど有名な
  大企業株以外の44銘柄(MU・WBDのような中堅株を含む)が拒否され、実質「小型株を含めて
  スイングトレード候補を探す」という目的を果たせなかった。そのため過去株価だけ
  `api/yfinance_client.py`(Yahoo Finance、無料・APIキー不要、レート制限以外の制約なし)に
  切り替えている。企業概要・財務・候補銘柄生成は引き続きFMPを使う(小型株でも動作確認済み)。
  `api/fmp_client.py`の`historical_prices()`は有料プランに切り替えた場合の参考実装として
  残してあるだけで、現在のパイプラインからは呼ばれない。
- **勝率・期待値は暫定値**: バックテストエンジンが未実装のため、`analysis_results` の
  `win_rate` / `expected_value` は総合スコアから導いたヒューリスティックです。次フェーズで
  `backtest_results` テーブルを使った本物の統計値に置き換える設計にしてあります。
- **スコア配点**: 要件定義書の配点(出来高急増+20など)をそのまま採用していますが、
  合計すると120点になり「100点満点」という記載と数値上は一致しません。全条件達成の
  ボーナスを許容する設計とみなし、`config.SCORE_MAX`(100点)で最終スコアをキャップしています。
- **API予算ガード**: `db.api_usage` テーブルで当日のFMPリクエスト数を記録し、
  `MAX_DAILY_REQUESTS` に達すると以降の呼び出しを中断してその時点までの結果でレポートを出します。
- **FMP日次リクエスト消費量は少ない**: 過去株価がyfinance(無料・無制限)に移ったため、FMPを
  消費するのは most-actives・biggest-gainers取得(2) + Top10×2(profile/財務)(最大20)程度で、
  1日あたり合計20〜25リクエスト程度に収まる見込み(250/日の予算に対してかなり余裕がある)。
- **GitHub Actions実行時はSQLiteキャッシュが毎回リセットされる**: Actionsは毎回まっさらな
  checkoutから始まる(`data/`はgitignore対象で永続化していない)ため、差分キャッシュの恩恵は
  ローカル実行時のみ。Actions上では毎日候補銘柄分の過去株価をyfinanceからフルで取得し直す形に
  なるが、yfinanceは無料・実質無制限なので実害はない。

## 毎営業日オープン前の自動実行(GitHub Actions)

`.github/workflows/daily.yml` が平日12:00 UTC(日本時間21:00、米国市場オープン22:30〜23:30より前)に
GitHub側のサーバーで自動実行する。**このパソコンの電源やログイン状態に関係なく動く。**
`FMP_API_KEY`・`ANTHROPIC_API_KEY` はリポジトリの Settings → Secrets and variables → Actions に
登録済み(コードには含まれない)。手動実行やログ確認は https://github.com/itsukifurue/us-stock-screener/actions から。

以前はWindowsタスクスケジューラ + `run_daily.ps1` を使っていたが、パソコンの電源が必要という制約が
あったためGitHub Actionsに移行し、ローカルのタスク登録は削除済み。`run_daily.ps1` はローカルで
手動実行しつつ結果をpushしたい場合のために残してある(必須ではない)。

## Webダッシュボード(他のパソコンから閲覧)

`app.py`(Streamlit)が `reports/*.json` を読み込んでパスワード保護つきで表示する。
GitHubにpushされた最新のレポートを、Streamlit Community Cloud経由でどのパソコンからでも閲覧できる。

- リポジトリ: `https://github.com/itsukifurue/us-stock-screener`(公開リポジトリ。秘密情報は`.env`のみでgit管理外)
- パスワードは Streamlit Cloud側の「Secrets」に `APP_PASSWORD = "..."` として設定(コードには含まれない)
- `run_daily.ps1` が新しいレポートをpushすると、Streamlit Cloudが自動で再デプロイして最新内容を表示する

## バックテスト

```bash
python scripts/run_backtest.py
```

`backtest/watchlist.py` の固定銘柄リスト(30銘柄)について、過去`config.BACKTEST_YEARS`年分(既定5年)を
yfinanceで取得し、テクニカルスコアが`config.BACKTEST_MIN_TECHNICAL_SCORE`点以上になった日にエントリー、
ATRベースの損切り/利確(本番stage3と同じ式)に触れるか`config.BACKTEST_MAX_HOLDING_DAYS`日(既定10営業日)
経過したら手仕舞い、という単純なルールでシミュレーションする。

**重要な制約**: 本番パイプラインが使う「当日の値動き上位ランキング」による銘柄選定と、Claudeによる
AI評価は、過去の特定日について無料では再現できない。そのためバックテストは**テクニカル条件のみ**を
検証するものであり、本番の実績をそのまま表すものではない。

結果は `data/screener.db` の `backtest_results`(個別トレード)・`backtest_summary`(集計)テーブルと、
`reports/backtest_YYYY-MM-DD.md` に保存される。最大ドローダウンは「1銘柄に集中投資していた場合の
最悪ケース」の簡易値(詳細はレポート内の注記を参照)。

直近の実行結果(2026-07-22、5年・30銘柄): トレード1574件、勝率44.41%、プロフィットファクター1.23、
期待値+0.76%/トレード。地合いによって結果は変動するため、定期的に再実行して確認することを推奨。

## 今後のロードマップ

- [x] バックテストエンジン(テクニカル条件のみの簡易版。過去5年・勝率・PF・最大DD・Sharpe Ratio等)
- [ ] Discord / LINE / Slack通知、メール配信
- [ ] TradingViewチャート画像表示・リンク生成
- [ ] ポートフォリオ管理・売買履歴管理
- [ ] スコア自動学習による最適化
- [ ] 条件ごとのバックテスト比較
