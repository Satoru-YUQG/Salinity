from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from config import PROJECT_ROOT


DEFAULT_HISTORY_DB = PROJECT_ROOT / "data" / "experiment_history.sqlite3"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else DEFAULT_HISTORY_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS experiment_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            title TEXT NOT NULL,
            run_type TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            config_json TEXT NOT NULL,
            summary_json TEXT NOT NULL,
            results_json TEXT NOT NULL
        )
        """
    )
    return connection


@contextmanager
def _connection(db_path: Optional[Path] = None):
    connection = _connect(db_path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def save_run(
    *,
    title: str,
    run_type: str,
    algorithm: str,
    config: dict,
    results: dict,
    db_path: Optional[Path] = None,
) -> int:
    rows = results.get("rows", [])
    with _connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO experiment_runs
                (created_at, title, run_type, algorithm, config_json, summary_json, results_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().astimezone().isoformat(timespec="seconds"),
                title,
                run_type,
                algorithm,
                json.dumps(config, default=_json_default, allow_nan=False),
                json.dumps(rows, default=_json_default, allow_nan=False),
                json.dumps(results, default=_json_default, allow_nan=False),
            ),
        )
        return int(cursor.lastrowid)


def list_runs(db_path: Optional[Path] = None) -> list[dict]:
    with _connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT id, created_at, title, run_type, algorithm, config_json, summary_json
            FROM experiment_runs
            ORDER BY id DESC
            """
        ).fetchall()
    return [
        {
            "id": row["id"],
            "created_at": row["created_at"],
            "title": row["title"],
            "run_type": row["run_type"],
            "algorithm": row["algorithm"],
            "config": json.loads(row["config_json"]),
            "summary": json.loads(row["summary_json"]),
        }
        for row in rows
    ]


def get_run(run_id: int, db_path: Optional[Path] = None) -> Optional[dict]:
    with _connection(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM experiment_runs WHERE id = ?", (int(run_id),)
        ).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "title": row["title"],
        "run_type": row["run_type"],
        "algorithm": row["algorithm"],
        "config": json.loads(row["config_json"]),
        "summary": json.loads(row["summary_json"]),
        "results": json.loads(row["results_json"]),
    }


def delete_run(run_id: int, db_path: Optional[Path] = None) -> bool:
    with _connection(db_path) as connection:
        cursor = connection.execute(
            "DELETE FROM experiment_runs WHERE id = ?", (int(run_id),)
        )
        return cursor.rowcount > 0