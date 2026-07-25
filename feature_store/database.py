"""Version2 point-in-time feature store用のSQLiteラッパー。"""
from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

_CREATE_TABLE_RE = re.compile(r"CREATE TABLE IF NOT EXISTS\s+(\w+)\s*\((.*?)\n\);", re.DOTALL)
_COLUMN_LINE_RE = re.compile(r"^\s*(\w+)\s+(TEXT|REAL|INTEGER)\b")
_SQL_KEYWORDS = {"PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "CONSTRAINT"}


class FeatureStoreDB:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        self._migrate_missing_columns()

    def _init_schema(self) -> None:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            self.conn.executescript(f.read())
        self.conn.commit()

    def _migrate_missing_columns(self) -> None:
        """schema.sqlに定義されているが、既存DBファイルには無い列を追加する軽量マイグレーション
        (CREATE TABLE IF NOT EXISTSは既存テーブルへ新列を追加してくれないため)。
        テーブル自体の新規追加は_init_schemaのCREATE TABLE IF NOT EXISTSで既に処理済み。
        """
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        for match in _CREATE_TABLE_RE.finditer(schema_sql):
            table_name, body = match.group(1), match.group(2)
            body = re.sub(r"--[^\n]*", "", body)  # 行コメントを除去してからカンマ分割する
            existing_cols = {
                row[1] for row in self.conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            }
            if not existing_cols:
                continue  # テーブルがまだ存在しない(通常は起こらないが念のため)

            for line in body.split(","):
                line = line.strip()
                col_match = _COLUMN_LINE_RE.match(line)
                if not col_match:
                    continue
                col_name, col_type = col_match.group(1), col_match.group(2)
                if col_name.upper() in _SQL_KEYWORDS or col_name in existing_cols:
                    continue
                self.conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}")
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

    # ---------- ticker_aliases ----------
    def upsert_ticker_alias(self, row: dict) -> None:
        self._upsert("ticker_aliases", row, ("old_ticker", "new_ticker"))

    def get_ticker_alias_for(self, old_ticker: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM ticker_aliases WHERE old_ticker = ? ORDER BY effective_date DESC LIMIT 1",
            (old_ticker,),
        ).fetchone()
        return dict(row) if row else None

    def get_all_ticker_aliases(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM ticker_aliases ORDER BY old_ticker").fetchall()
        return [dict(r) for r in rows]

    # ---------- build_run_symbol_status(再開可能ビルド用) ----------
    def set_symbol_stage_status(
        self, build_run_id: str, symbol: str, stage: str, status: str,
        started_at: str | None = None, completed_at: str | None = None,
        rows_written: int | None = None, error_type: str | None = None,
        error_message: str | None = None, retry_count: int = 0,
    ) -> None:
        self._upsert(
            "build_run_symbol_status",
            {
                "build_run_id": build_run_id, "symbol": symbol, "stage": stage, "status": status,
                "started_at": started_at, "completed_at": completed_at, "rows_written": rows_written,
                "error_type": error_type, "error_message": error_message, "retry_count": retry_count,
            },
            ("build_run_id", "symbol", "stage"),
        )

    def get_symbol_stage_status(self, build_run_id: str, symbol: str, stage: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM build_run_symbol_status WHERE build_run_id=? AND symbol=? AND stage=?",
            (build_run_id, symbol, stage),
        ).fetchone()
        return dict(row) if row else None

    def get_all_symbol_stage_status(self, build_run_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM build_run_symbol_status WHERE build_run_id=? ORDER BY symbol, stage",
            (build_run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------- feature_metadata(Feature Freeze対応) ----------
    def upsert_feature_metadata(self, row: dict) -> None:
        self._upsert("feature_metadata", row, ("feature_name",))

    def upsert_feature_metadata_bulk(self, rows: list[dict]) -> None:
        for r in rows:
            self.upsert_feature_metadata(r)

    def get_all_feature_metadata(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM feature_metadata ORDER BY feature_name").fetchall()
        return [dict(r) for r in rows]

    # ---------- Phase3A: model_experiments/model_metrics/model_coefficients ----------
    def upsert_model_experiment(self, row: dict) -> None:
        self._upsert("model_experiments", row, ("experiment_id",))

    def insert_model_metric(self, experiment_id: str, split_name: str, metric_name: str, metric_value, metric_detail: str | None = None) -> None:
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO model_metrics (experiment_id, split_name, metric_name, metric_value, metric_detail) VALUES (?, ?, ?, ?, ?)",
                (experiment_id, split_name, metric_name, metric_value, metric_detail),
            )

    def insert_model_coefficient(self, experiment_id: str, fold_name: str, feature_name: str, coefficient: float, odds_ratio: float, abs_rank: int) -> None:
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO model_coefficients (experiment_id, fold_name, feature_name, coefficient, odds_ratio, abs_rank) VALUES (?, ?, ?, ?, ?, ?)",
                (experiment_id, fold_name, feature_name, coefficient, odds_ratio, abs_rank),
            )

    def get_model_metrics(self, experiment_id: str) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM model_metrics WHERE experiment_id = ?", (experiment_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_model_coefficients(self, experiment_id: str) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM model_coefficients WHERE experiment_id = ?", (experiment_id,)).fetchall()
        return [dict(r) for r in rows]

    # ---------- price_cache_meta ----------
    def upsert_price_cache_meta(self, row: dict) -> None:
        self._upsert("price_cache_meta", row, ("symbol", "provider", "cache_version"))

    def get_price_cache_meta(self, symbol: str, provider: str, cache_version: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM price_cache_meta WHERE symbol=? AND provider=? AND cache_version=?",
            (symbol, provider, cache_version),
        ).fetchone()
        return dict(row) if row else None
