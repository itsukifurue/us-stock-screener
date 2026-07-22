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

## Version 2: Feature Store(Phase 1 — データ基盤のみ、モデル学習はまだ)

Version 1の監査結果(スコアベースの採用ルールに逆選択があり、優位性が確認できない)を受けて、
スコアの微調整ではなく、**特徴量と将来ラベルを大量蓄積し、後から統計・機械学習でランキングモデルを
構築するためのデータ基盤**を作る方針に転換した。Phase 1は「基盤が正しく動くこと」の証明が目的で、
モデル学習はまだ行わない。

### 実行方法

```bash
python scripts/build_feature_store.py --symbols AAPL,MSFT,NVDA,TSLA,AMD --years 1
python -m unittest tests.test_feature_store -v
```

結果は `data/feature_store.db`(`features`・`labels`・`data_quality_log` の3テーブル)と
`reports/feature_store_quality_YYYY-MM-DD.md` に保存される(DB本体はgit管理対象外)。

### データ構造

- 主キー: `(ticker, signal_date)`
- **`features`テーブル**: signal_date時点で実際に知り得た情報だけを格納(識別情報・流動性・
  トレンド・モメンタム・ブレイクアウト・ボラティリティ・ローソク足・市場環境・セクター相対強度・
  Version1の`technical_score_v1`とその構成条件)。未来のデータは一切参照しない。
  **候補条件を満たさなかった日も含め、対象銘柄・対象期間の全営業日分を保存する**
  (`universe_included_flag`/`candidate_flag`/`candidate_reason`/`universe_version`列で
  「全営業日データ」「候補ユニバースに入った日」を後から正確に区別・抽出できるようにしている)。
- **`labels`テーブル**: 完全に別工程(`feature_store/labels.py`)で生成する将来の結果。
  `future_return_*` / `future_max_return_*` / `future_min_return_*` / `hit_plus_*pct_*d` /
  `days_to_plus_*pct` / 主要ラベル`target_15pct_within_10d`・`target_trade_success`。
  全ラベルは「シグナル翌営業日の始値」を基準に統一している。
  - `target_15pct_within_10d`: 翌営業日始値を基準に、10営業日以内に高値が+15%以上へ到達したら1
  - `target_trade_success`: 翌営業日始値エントリー・スリッページ0.2%・手数料0.1%・
    損切りEntry−ATR14×1.5・利確Entry×1.15・最大保有10営業日という、Version1の
    `backtest.engine.price_trade_at_signal`と全く同じロジックを再利用し、損切り到達前に
    利確到達すれば1(同日中に両方到達した場合は損切り優先の保守的判定)
  - 将来ウィンドウのデータが十分に無い場合は0/1で確定させずNoneを返す
    (「未到達」と「データ不足で判定不能」を区別するため)。加えて、保有期間
    (max_holding_days)を使い切る前にデータそのものが尽きた場合(上場廃止・データ末尾を想定)も、
    `backtest.engine.price_trade_at_signal`が返す`exit_reason=="data_end"`を見て
    `target_trade_success`/`hit_stop_atr_1_5_before_plus_15`をNoneのまま確定させない
    (`holding_period_limit`=正当な期間満了、とは区別する)。境界値は
    `tests/test_label_boundaries.py`でユニットテスト済み(ちょうど10営業日目到達・11営業日目
    到達・同日stop/target・データ末尾・ATR算出不能・直近未確定・分割ギャップ検知)。

### 証明実行の結果(5銘柄・1年、2026-07-22実行)

`python scripts/build_feature_store.py --symbols AAPL,MSFT,NVDA,TSLA,AMD --years 1` を実行した
結果、`features`/`labels`とも1255件。**この1255件は「5銘柄 × 対象期間(2025-06-27〜2026-06-27)の
全営業日」であり、候補条件で絞り込んだ結果ではない**(全営業日を保存する設計に変更済み)。
ただし対象の5銘柄(AAPL/MSFT/NVDA/TSLA/AMD)は株価・時価総額・平均出来高のいずれも一次スクリーニング
条件を常に大きく上回る大型株のため、今回はたまたま1255件全てが`candidate_flag=1`(候補日)にもなった
(全営業日件数と候補日件数が偶然一致している)。Phase 2で時価総額の小さい銘柄や新規上場銘柄を含めると、
`candidate_flag=0`の非候補日レコードが実際に現れる想定。

データ品質チェックは全項目PASS。欠損率は3区分に分けて確認した。

- **予期しない欠損(ウォームアップ済みのはずが欠損)**: 全20列で0.0%
- **ウォームアップ期間中の欠損**: 0件(下記の遡り取得により、保存対象期間内では発生しなかった)
- **仕様上常にNULLの列(品質問題ではない)**: `bid_ask_spread`/`turnover_ratio`/`beta`/`ma20`が
  いずれも100.0%(採用していない/無料データ源では取得不能なため意図的に常にNULL。0埋めはしていない。
  それぞれ`bid_ask_spread_available`/`turnover_ratio_available`/`beta_available`列が0であることでも
  「取得不能」であることを明示している)

### ウォームアップ期間の確保

MA200等の長期指標を正しく計算するため、`scripts/build_feature_store.py`は保存対象期間の開始日より
**420暦日(≒287営業日、実測でAAPL: 287営業日)前**から価格データを取得しており、要件の
「開始日より最低250〜300営業日前」を満たす。指標計算はpandasの`rolling(window, min_periods=window)`
方式のみを使用しており、ウォームアップ不足分は0埋め・後方補完・未来方向からの補完のいずれもせず
NaN(→NULL)のまま残る。

なお、`analysis.technical`のRSI(本番pipeline用)はウォームアップ不足分を中立値50でfillnaする仕様
(意図的な設計)だが、これをfeature storeでそのまま使うと「本当にRSI=50」なのか「データ不足で計算不能」
なのかを区別できず欠損チェックもすり抜けてしまう。そのため`feature_store/features.py`では同じRSI計算を
fillnaなしで独自に再計算しており(`rsi_7`/`rsi_14`)、ウォームアップ期間中はNULLのまま残る。

### 価格データの調整方法

過去株価は`api/yfinance_client.py`(`yf.Ticker(symbol).history(..., auto_adjust=False)`)で取得している。

- **株式分割調整: あり**。`auto_adjust=False`でも、yfinanceはOHLC・出来高を株式分割に対して
  自動的に調整して返す(2024-06-10のNVDA 10:1分割を実データで検証済み。分割日前後で不自然な
  ギャップが生じないことを確認した)。
- **配当調整: なし**。`auto_adjust=False`のため、返される`Close`は配当落ち調整前の生の終値であり、
  `Adj Close`列は使用していない。
- **OHLC・出来高は同一の取得元・同一の調整方式**(分割調整のみ・配当調整なし)で統一されており、
  一部の列だけAdjusted Closeを使うといった不整合はない。
- **featuresとlabelsは完全に同じ調整方式を使う**: 両モジュールとも同一の価格取得結果(同じDataFrame)
  を参照しており、`labels.py`が独自に別の価格ソースを取得することはない。

分割・調整方式に起因する異常(未調整の分割混入等)は、品質チェックの`suspicious_split_gap`
(前日比±40%超のギャップ検知)で継続的に監視する。

### market_regime の分類ルール(`feature_store/market_regime.py`に実装、固定・明示)

```
spy_above_ma200 かつ spy_above_ma50:
    spy_return_5d > +2.0% → "strong_bull"、それ以外 → "bull"
spy_above_ma200 が偽 かつ spy_above_ma50 が偽:
    spy_return_5d < -2.0% → "strong_bear"、それ以外 → "bear"
どちらでもない(200日線と50日線で判定が割れている過渡期): "neutral"
```

### 既知の制約・近似(Phase 2以降の課題)

- **候補ユニバースは近似**: FMP無料プランでは過去のある日の値動き上位ランキングを取得できないため、
  `feature_store/universe.py`では「あらかじめ決めた銘柄リストの中から、その日時点の一次スクリーニング
  条件(株価5ドル以上・時価総額1億ドル以上・平均出来高50万株以上)を満たすものを候補とする」近似を
  使っている(`candidate_source = "approx_universe"`)。上場廃止銘柄も含められないため生存者バイアスが残る。
- **セクター分類は現在時点のものを過去に遡って適用**: yfinanceの`Ticker.info['sector']`は現在の分類しか
  取得できないため、真のpoint-in-timeセクター分類ではない。
- **`bid_ask_spread`・`turnover_ratio`・`beta`は無料データ源では算出困難なため常にNULL**
  (0埋めではなくNULL。`bid_ask_spread_available`/`turnover_ratio_available`/`beta_available`列で
  「仕様上取得不能」であることを明示。`ma20`も本ツールでは未採用のため常にNULL)。
- Phase 1は5銘柄・1年分の証明実行のみ(かつ全て一次スクリーニング条件を常に満たす大型株のため、
  全営業日レコード=候補日レコードとなっている)。

## Version 2: Phase 2 Step1(46銘柄・5年への拡張 + 逆選択監査)

Phase 1の基盤の上に、対象を46銘柄(`backtest.watchlist.PHASE2_WATCHLIST`。既存29銘柄+
ヘルスケア/金融/エネルギー/資本財/生活必需品/REIT/公益の大型株+2023〜2024年IPO銘柄
ARM/CART/RDDTを追加し、セクター・上場時期を多様化)・5年(2021-06-28〜2026-06-27)へ拡張した。

### 実行方法

```bash
python scripts/build_feature_store_phase2.py --years 5   # DB構築(features/labels/daily_universe/candidate_snapshots/build_runs)
python scripts/analyze_phase2_step1.py                     # 単変量分析・Version1スコア分析レポート生成
python scripts/analyze_anti_selection.py                   # 逆選択の原因分析(監査)レポート生成
```

### 2層ユニバース設計

- **`features`/`labels`/`daily_universe`**: 対象46銘柄(取得失敗の`SQ`除く45銘柄)の全営業日
  (54,365件)。候補にならなかった日も含む。
- **`candidate_snapshots`**: 一次スクリーニング条件(price/market_cap/avg_volume)を満たした
  universe candidateのみ(52,166件)。`signaled_flag`(technical_score_v1>=45で信号化されたか)、
  `selected_by_v1_flag`(Version1の採用ルールで実際に採用されたか)、`rejected_reason`
  (見送りの場合、`no_slot`=3枠が全て埋まっていた/`cash_insufficient`=枠はあったが投資可能な
  現金が無かった、を明確に区別)を保持する。
- **`build_runs`**: Train(60%)/Validation(20%)/Test(20%)の日付境界を記録する
  (Train 2021-06-28〜2024-06-27 / Validation 2024-06-27〜2025-06-27 /
  Test 2025-06-27〜2026-06-27)。

### Test期間の封印(監査済み)

`scripts/analyze_phase2_step1.py`・`scripts/analyze_anti_selection.py`とも、Test期間のデータは
SQLの`WHERE signal_date < test_start`の段階で除外しており、読み込んだDataFrameに最初から
含まれていない(表示時に隠しているのではない)。スクリプト内で「読み込んだ全データの最大
signal_dateがtest_start未満であること」をassertで機械的に検証している。Test期間について集計
するのは、件数・欠損率などの品質情報のみ(目的変数に関する成績は一切計算しない)。

### 逆選択監査の主な結論(Train+Validation期間のみ、詳細は`reports/anti_selection_analysis_*.md`)

Version1の採用ルール(technical_score_v1>=45で信号化 → スコア降順で最大3銘柄まで採用)について、
46銘柄・5年のデータで以下を確認した。

1. **採用ルールは同日の見送り候補より一貫して悪い結果を選んでいた**: 競合が発生した194日のうち、
   採用が見送りの平均を上回ったのはわずか42.3%。最悪の候補を採用してしまった日が42.3%ある一方、
   最良の候補を採用できた日は34.0%に留まる。
2. **現行のスコア降順ランキングは、ランダム選択より明確に劣る**: 同じ候補集合に対して
   ランダムに1000回選び直した分布と比較すると、現行ルールのCAGRはその分布の13.5パーセンタイル
   (下位)、最大ドローダウンは89.5パーセンタイル(ランダムの89.5%より悪い)に位置する。
   スコアの高い順に採用するより、低い順・ブレイクアウト幅が小さい順に採用する方がCAGR・PFとも
   明確に良かった(例: ブレイクアウト幅が小さい順でCAGR -13.1% vs 現行 -26.1%)。
3. **3枠の順番待ち制約と資金制約は、それぞれ独立して悪化させている**: 制約なしで全シグナルを
   均等サイズ評価した場合(PF 0.924)と比べ、3枠制約のみを加えるとPF 0.81へ、さらに現実的な
   資金制約を加えるとPF 0.721へと、段階的に悪化する。
4. **半年ごとの成績は極めて不安定**(Train期間内だけで PF 0.365〜1.363の間で変動)。2023年は
   好調だったが2021H1・2022H1・2024H1は不調で、スコアの有効性が市場環境に強く依存している
   可能性が高い。
5. Version1で評価対象にしている特徴量(volume_ratio・breakout幅・RSI・ATR%・price_vs_ma25等)
   のうち、採用/見送り間で意味のある効果量が確認できたのはtechnical_score_v1自体(Cohen's d=0.611、
   スコアは意図通り機能している)くらいで、個別の技術的特徴量にはほぼ差がなかった
   (|d|<0.2がほとんど)。最も差が大きかったのは同日候補数(d=-0.69、採用日の方が競合が少ない)で、
   これは市場環境(同時に多くの銘柄がシグナルを出す日=過熱相場)による交絡の可能性を示唆する。

**結論**: 単一要因ではなく複数要因の組み合わせ(採用ルールのランキング能力の欠如、3枠制約による
タイミングの悪化、資金制約による追加の悪化、市場環境依存の不安定性)と判断する。Step 2(銘柄数
拡張)より前に、この技術スコア自体の設計を見直すか、少なくとも「現状のスコアには実質的な
銘柄選択能力が無い」という前提でVersion2の設計を進めるべきという示唆が強い。

## 今後のロードマップ

- [x] バックテストエンジン(テクニカル条件のみの簡易版。過去5年・勝率・PF・最大DD・Sharpe Ratio等)
- [x] Version 2 Phase 1: point-in-time feature store基盤(5銘柄・1年で証明済み)
- [ ] Version 2 Phase 2: 全銘柄・複数年への拡張、単変量分析、Version1スコアとの比較分析
- [ ] Version 2 Phase 3〜5: ロジスティック回帰→Random Forest→LightGBM/XGBoost、バックテスト接続
- [ ] Discord / LINE / Slack通知、メール配信
- [ ] TradingViewチャート画像表示・リンク生成
- [ ] ポートフォリオ管理・売買履歴管理
- [ ] 条件ごとのバックテスト比較
