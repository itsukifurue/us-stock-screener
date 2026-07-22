"""米国株AIスクリーニング結果を表示するWebダッシュボード(Streamlit)。

ローカルの main.py が生成した reports/*.json を読み込んで表示するだけで、
FMP/Anthropicへの通信は一切行わない(APIキー不要)。
パスワードは Streamlit Cloud の Secrets(st.secrets["APP_PASSWORD"])で管理する。
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

REPORTS_DIR = Path(__file__).parent / "reports"

st.set_page_config(page_title="米国株AIスクリーニング", page_icon="📈", layout="wide")


def check_password() -> bool:
    """簡易パスワードゲート。st.secrets['APP_PASSWORD']と一致すればTrueを返す。"""

    def password_entered() -> None:
        if st.session_state.get("password") == st.secrets.get("APP_PASSWORD"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct"):
        return True

    st.text_input("パスワード", type="password", on_change=password_entered, key="password")
    if st.session_state.get("password_correct") is False:
        st.error("パスワードが違います")
    return False


if not check_password():
    st.stop()

st.title("📈 米国株AIスクリーニング結果")

if not REPORTS_DIR.exists():
    st.warning("reportsフォルダがまだありません。")
    st.stop()

report_files = sorted(REPORTS_DIR.glob("*.json"), reverse=True)
if not report_files:
    st.warning("レポートがまだありません。main.py を実行してGitHubにpushすると、ここに表示されます。")
    st.stop()

dates = [f.stem for f in report_files]
selected_date = st.selectbox("日付を選択", dates)

with open(REPORTS_DIR / f"{selected_date}.json", encoding="utf-8") as f:
    data = json.load(f)

top_items = data.get("top", [])
st.caption(data.get("disclaimer", ""))

if not top_items:
    st.info("この日は条件を満たす銘柄がありませんでした。")

for i, item in enumerate(top_items, start=1):
    with st.container(border=True):
        st.subheader(f"{i}位: {item['symbol']} — {item['company_name']}(スコア {item['total_score']:.1f}/100)")
        cols = st.columns(4)
        cols[0].metric("現在価格", f"${item['current_price']:.2f}")
        cols[1].metric("エントリー", f"${item['entry_price']:.2f}")
        cols[2].metric("損切", f"${item['stop_loss']:.2f}")
        cols[3].metric("利確②", f"${item['take_profit_2']:.2f}")
        st.write(
            f"**期待上昇率**: {item['expected_return_pct']:.2f}%　"
            f"**勝率(暫定)**: {item['win_rate']:.1f}%　"
            f"**期待値(暫定)**: {item['expected_value']:.2f}"
        )
        st.write(f"**AIコメント**: {item['ai_comment']}")
        st.write(f"**ニュース要約**: {item['news_summary']}")
        st.write(f"**リスク要因**: {item['risk_factors']}")
