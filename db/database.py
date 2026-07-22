"""SQLite接続とCRUDヘルパー。FMPリクエスト回数の日次トラッキングもここで行う。"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class Database:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            self.conn.executescript(f.read())
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def cursor(self):
        cur = self.conn.cursor()
        try:
            yield cur
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cur.close()

    # ---------- stocks ----------
    def upsert_stock(self, stock: dict) -> None:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO stocks (symbol, name, exchange, sector, industry, country, is_etf, market_cap, updated_at)
                VALUES (:symbol, :name, :exchange, :sector, :industry, :country, :is_etf, :market_cap, :updated_at)
                ON CONFLICT(symbol) DO UPDATE SET
                    name=excluded.name, exchange=excluded.exchange, sector=excluded.sector,
                    industry=excluded.industry, country=excluded.country, is_etf=excluded.is_etf,
                    market_cap=excluded.market_cap, updated_at=excluded.updated_at
                """,
                {
                    "symbol": stock["symbol"],
                    "name": stock.get("name"),
                    "exchange": stock.get("exchange"),
                    "sector": stock.get("sector"),
                    "industry": stock.get("industry"),
                    "country": stock.get("country"),
                    "is_etf": int(stock.get("is_etf", False)),
                    "market_cap": stock.get("market_cap"),
                    "updated_at": datetime.utcnow().isoformat(),
                },
            )

    # ---------- daily_prices ----------
    def latest_price_date(self, symbol: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT MAX(date) AS d FROM daily_prices WHERE symbol = ?", (symbol,)
        ).fetchone()
        return row["d"] if row and row["d"] else None

    def insert_daily_prices(self, symbol: str, rows: Iterable[dict]) -> int:
        rows = list(rows)
        if not rows:
            return 0
        with self.cursor() as cur:
            cur.executemany(
                """
                INSERT OR IGNORE INTO daily_prices (symbol, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (symbol, r["date"], r.get("open"), r.get("high"), r.get("low"), r.get("close"), r.get("volume"))
                    for r in rows
                ],
            )
            return cur.rowcount

    def get_price_history(self, symbol: str, min_date: Optional[str] = None) -> list[dict]:
        query = "SELECT * FROM daily_prices WHERE symbol = ?"
        params: list = [symbol]
        if min_date:
            query += " AND date >= ?"
            params.append(min_date)
        query += " ORDER BY date ASC"
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ---------- api_usage ----------
    def record_api_usage(self, count: int = 1) -> int:
        today = date.today().isoformat()
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO api_usage (date, request_count) VALUES (?, ?)
                ON CONFLICT(date) DO UPDATE SET request_count = request_count + excluded.request_count
                """,
                (today, count),
            )
        return self.get_today_usage()

    def get_today_usage(self) -> int:
        today = date.today().isoformat()
        row = self.conn.execute(
            "SELECT request_count FROM api_usage WHERE date = ?", (today,)
        ).fetchone()
        return row["request_count"] if row else 0

    # ---------- screening_runs ----------
    def create_screening_run(self, run_date: str) -> int:
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO screening_runs (run_date, started_at, status) VALUES (?, ?, 'running')",
                (run_date, datetime.utcnow().isoformat()),
            )
            return cur.lastrowid

    def finish_screening_run(self, run_id: int, candidate_count: int, status: str = "completed") -> None:
        with self.cursor() as cur:
            cur.execute(
                "UPDATE screening_runs SET finished_at = ?, candidate_count = ?, status = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), candidate_count, status, run_id),
            )

    # ---------- candidates ----------
    def upsert_candidate(
        self,
        run_id: int,
        symbol: str,
        stage: str,
        technical_score: Optional[float] = None,
        ai_score: Optional[float] = None,
        total_score: Optional[float] = None,
        rank: Optional[int] = None,
    ) -> None:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO candidates (run_id, symbol, stage, technical_score, ai_score, total_score, rank)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, symbol) DO UPDATE SET
                    stage=excluded.stage, technical_score=excluded.technical_score,
                    ai_score=excluded.ai_score, total_score=excluded.total_score, rank=excluded.rank
                """,
                (run_id, symbol, stage, technical_score, ai_score, total_score, rank),
            )

    def get_candidates(self, run_id: int, stage: Optional[str] = None) -> list[dict]:
        if stage:
            rows = self.conn.execute(
                "SELECT * FROM candidates WHERE run_id = ? AND stage = ? ORDER BY total_score DESC",
                (run_id, stage),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM candidates WHERE run_id = ? ORDER BY total_score DESC", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------- analysis_results ----------
    def insert_analysis_result(self, result: dict) -> None:
        result = {**result, "created_at": datetime.utcnow().isoformat()}
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO analysis_results (
                    run_id, symbol, company_name, current_price, entry_price, stop_loss,
                    take_profit_1, take_profit_2, expected_return_pct, win_rate, expected_value,
                    ai_comment, news_summary, risk_factors, total_score, created_at
                ) VALUES (
                    :run_id, :symbol, :company_name, :current_price, :entry_price, :stop_loss,
                    :take_profit_1, :take_profit_2, :expected_return_pct, :win_rate, :expected_value,
                    :ai_comment, :news_summary, :risk_factors, :total_score, :created_at
                )
                ON CONFLICT(run_id, symbol) DO UPDATE SET
                    company_name=excluded.company_name, current_price=excluded.current_price,
                    entry_price=excluded.entry_price, stop_loss=excluded.stop_loss,
                    take_profit_1=excluded.take_profit_1, take_profit_2=excluded.take_profit_2,
                    expected_return_pct=excluded.expected_return_pct, win_rate=excluded.win_rate,
                    expected_value=excluded.expected_value, ai_comment=excluded.ai_comment,
                    news_summary=excluded.news_summary, risk_factors=excluded.risk_factors,
                    total_score=excluded.total_score, created_at=excluded.created_at
                """,
                result,
            )

    def get_analysis_results(self, run_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM analysis_results WHERE run_id = ? ORDER BY total_score DESC", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------- news_cache ----------
    def cache_news(self, symbol: str, items: Iterable[dict]) -> None:
        items = list(items)
        if not items:
            return
        with self.cursor() as cur:
            cur.executemany(
                """
                INSERT OR IGNORE INTO news_cache (symbol, published_date, title, summary, url, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (symbol, i.get("published_date"), i.get("title"), i.get("summary"), i["url"], datetime.utcnow().isoformat())
                    for i in items
                ],
            )

    def get_cached_news(self, symbol: str, limit: int = 10) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM news_cache WHERE symbol = ? ORDER BY published_date DESC LIMIT ?",
            (symbol, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def news_fetched_today(self, symbol: str) -> bool:
        today = date.today().isoformat()
        row = self.conn.execute(
            "SELECT 1 FROM news_cache WHERE symbol = ? AND fetched_at >= ? LIMIT 1",
            (symbol, today),
        ).fetchone()
        return row is not None

    # ---------- financials_cache ----------
    def cache_financials(self, symbol: str, period: str, statement_type: str, data: object) -> None:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO financials_cache (symbol, period, statement_type, fetched_at, data_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(symbol, period, statement_type) DO UPDATE SET
                    fetched_at=excluded.fetched_at, data_json=excluded.data_json
                """,
                (symbol, period, statement_type, datetime.utcnow().isoformat(), json.dumps(data)),
            )

    def get_cached_financials(self, symbol: str, period: str, statement_type: str) -> Optional[object]:
        row = self.conn.execute(
            "SELECT data_json FROM financials_cache WHERE symbol = ? AND period = ? AND statement_type = ?",
            (symbol, period, statement_type),
        ).fetchone()
        return json.loads(row["data_json"]) if row else None
