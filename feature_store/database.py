"""Version2 point-in-time feature store用のSQLiteラッパー。"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class FeatureStoreDB:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
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

    # ---------- features ----------
    def upsert_feature(self, feature: dict) -> None:
        columns = list(feature.keys())
        placeholders = ", ".join(f":{c}" for c in columns)
        col_list = ", ".join(columns)
        update_clause = ", ".join(f"{c}=excluded.{c}" for c in columns if c not in ("ticker", "signal_date"))
        with self.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO features ({col_list}) VALUES ({placeholders})
                ON CONFLICT(ticker, signal_date) DO UPDATE SET {update_clause}
                """,
                feature,
            )

    def upsert_features_bulk(self, features: list[dict]) -> None:
        for f in features:
            self.upsert_feature(f)

    def get_feature(self, ticker: str, signal_date: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM features WHERE ticker = ? AND signal_date = ?", (ticker, signal_date)
        ).fetchone()
        return dict(row) if row else None

    def get_all_features(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM features ORDER BY signal_date, ticker").fetchall()
        return [dict(r) for r in rows]

    def count_features(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]

    # ---------- labels ----------
    def upsert_label(self, label: dict) -> None:
        columns = list(label.keys())
        placeholders = ", ".join(f":{c}" for c in columns)
        col_list = ", ".join(columns)
        update_clause = ", ".join(f"{c}=excluded.{c}" for c in columns if c not in ("ticker", "signal_date"))
        with self.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO labels ({col_list}) VALUES ({placeholders})
                ON CONFLICT(ticker, signal_date) DO UPDATE SET {update_clause}
                """,
                label,
            )

    def upsert_labels_bulk(self, labels: list[dict]) -> None:
        for l in labels:
            self.upsert_label(l)

    def get_label(self, ticker: str, signal_date: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM labels WHERE ticker = ? AND signal_date = ?", (ticker, signal_date)
        ).fetchone()
        return dict(row) if row else None

    def get_all_labels(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM labels ORDER BY signal_date, ticker").fetchall()
        return [dict(r) for r in rows]

    def count_labels(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]

    def get_features_joined_labels(self) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT f.*, l.future_return_1d, l.future_return_3d, l.future_return_5d,
                   l.future_return_10d, l.future_return_15d,
                   l.future_max_return_5d, l.future_max_return_10d, l.future_max_return_15d,
                   l.future_min_return_5d, l.future_min_return_10d, l.future_min_return_15d,
                   l.hit_plus_5pct_10d, l.hit_plus_10pct_10d, l.hit_plus_15pct_10d, l.hit_plus_15pct_15d,
                   l.hit_stop_atr_1_5_before_plus_15,
                   l.days_to_plus_5pct, l.days_to_plus_10pct, l.days_to_plus_15pct,
                   l.target_15pct_within_10d, l.target_trade_success, l.target_trade_pnl_pct
            FROM features f
            LEFT JOIN labels l ON f.ticker = l.ticker AND f.signal_date = l.signal_date
            ORDER BY f.signal_date, f.ticker
            """
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------- data_quality_log ----------
    def log_quality_check(self, check_name: str, status: str, affected_count: int, details: str) -> None:
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO data_quality_log (run_at, check_name, status, affected_count, details) "
                "VALUES (?, ?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), check_name, status, affected_count, details),
            )

    def get_quality_log(self, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM data_quality_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------- Phase2共通ヘルパー(汎用upsert。テーブルごとの主キー列を渡す) ----------
    def _upsert(self, table: str, row: dict, key_columns: tuple[str, ...]) -> None:
        columns = list(row.keys())
        placeholders = ", ".join(f":{c}" for c in columns)
        col_list = ", ".join(columns)
        update_clause = ", ".join(f"{c}=excluded.{c}" for c in columns if c not in key_columns)
        with self.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {table} ({col_list}) VALUES ({placeholders})
                ON CONFLICT({", ".join(key_columns)}) DO UPDATE SET {update_clause}
                """,
                row,
            )

    # ---------- build_runs ----------
    def upsert_build_run(self, build_run: dict) -> None:
        self._upsert("build_runs", build_run, ("build_run_id",))

    # ---------- universe_membership ----------
    def upsert_universe_membership(self, row: dict) -> None:
        self._upsert("universe_membership", row, ("ticker", "build_run_id"))

    def upsert_universe_membership_bulk(self, rows: list[dict]) -> None:
        for r in rows:
            self.upsert_universe_membership(r)

    # ---------- daily_universe ----------
    def upsert_daily_universe(self, row: dict) -> None:
        self._upsert("daily_universe", row, ("ticker", "date", "build_run_id"))

    def upsert_daily_universe_bulk(self, rows: list[dict]) -> None:
        for r in rows:
            self.upsert_daily_universe(r)

    def count_daily_universe(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM daily_universe").fetchone()[0]

    # ---------- candidate_snapshots ----------
    def upsert_candidate_snapshot(self, row: dict) -> None:
        self._upsert("candidate_snapshots", row, ("ticker", "signal_date", "build_run_id"))

    def upsert_candidate_snapshots_bulk(self, rows: list[dict]) -> None:
        for r in rows:
            self.upsert_candidate_snapshot(r)

    def count_candidate_snapshots(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM candidate_snapshots").fetchone()[0]

    def get_all_candidate_snapshots(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM candidate_snapshots ORDER BY signal_date, ticker"
        ).fetchall()
        return [dict(r) for r in rows]
