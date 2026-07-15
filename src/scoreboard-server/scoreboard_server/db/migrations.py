from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

from .connection import Database, DatabaseSession


INITIAL_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS schema_migrations (
        version INTEGER PRIMARY KEY, name TEXT NOT NULL, checksum TEXT NOT NULL,
        applied_at TEXT NOT NULL, applied_by TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS evaluation_runs (
        run_id TEXT PRIMARY KEY, publisher_subject TEXT NOT NULL,
        idempotency_key TEXT NOT NULL, create_digest TEXT NOT NULL,
        identity_json TEXT NOT NULL, identity_digest TEXT NOT NULL,
        suite TEXT NOT NULL, task_name TEXT NOT NULL, task_version TEXT NOT NULL,
        split_name TEXT NOT NULL, fewshot INTEGER NOT NULL, cot_mode TEXT NOT NULL,
        repair_strategy TEXT NOT NULL, model_name TEXT NOT NULL,
        checkpoint_digest TEXT NOT NULL, provider_revision TEXT NOT NULL,
        config_digest TEXT NOT NULL, dataset_digest TEXT NOT NULL,
        eligibility TEXT NOT NULL, comparable INTEGER NOT NULL,
        expected_sample_count INTEGER NOT NULL, sample_set_digest TEXT NOT NULL,
        status TEXT NOT NULL, revision INTEGER NOT NULL, ingest_digest TEXT,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL, completed_at TEXT,
        UNIQUE (publisher_subject, idempotency_key)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_runs_leaderboard ON evaluation_runs(status, eligibility, comparable, completed_at, run_id)",
    "CREATE INDEX IF NOT EXISTS idx_runs_model ON evaluation_runs(model_name, completed_at, run_id)",
    "CREATE INDEX IF NOT EXISTS idx_runs_task ON evaluation_runs(suite, task_name, task_version, completed_at, run_id)",
    """CREATE TABLE IF NOT EXISTS sample_evidence (
        run_id TEXT NOT NULL REFERENCES evaluation_runs(run_id) ON DELETE RESTRICT,
        sample_id TEXT NOT NULL, attempt INTEGER NOT NULL, status TEXT NOT NULL,
        prompt TEXT NOT NULL, raw_completion TEXT NOT NULL,
        scored_completion TEXT NOT NULL, generation_json TEXT NOT NULL,
        scoring_json TEXT NOT NULL, metrics_json TEXT NOT NULL,
        provenance_json TEXT NOT NULL, evidence_digest TEXT NOT NULL,
        error_code TEXT, error_message TEXT,
        PRIMARY KEY (run_id, sample_id, attempt)
    )""",
    """CREATE TABLE IF NOT EXISTS run_aggregates (
        run_id TEXT PRIMARY KEY REFERENCES evaluation_runs(run_id) ON DELETE RESTRICT,
        accounting_json TEXT NOT NULL, metrics_json TEXT NOT NULL,
        truncated_samples INTEGER NOT NULL, generated_samples INTEGER NOT NULL,
        manifest_digest TEXT NOT NULL, payload_digest TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS run_metrics (
        run_id TEXT NOT NULL REFERENCES evaluation_runs(run_id) ON DELETE RESTRICT,
        metric_name TEXT NOT NULL, metric_value DOUBLE PRECISION NOT NULL,
        is_primary INTEGER NOT NULL, PRIMARY KEY (run_id, metric_name)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_metrics_query ON run_metrics(metric_name, metric_value, run_id)",
    "CREATE INDEX IF NOT EXISTS idx_samples_page ON sample_evidence(run_id, sample_id, attempt)",
    """CREATE TABLE IF NOT EXISTS ingest_receipts (
        run_id TEXT NOT NULL REFERENCES evaluation_runs(run_id) ON DELETE RESTRICT,
        idempotency_key TEXT NOT NULL, request_digest TEXT NOT NULL,
        response_status TEXT NOT NULL, response_revision INTEGER NOT NULL,
        created_at TEXT NOT NULL, PRIMARY KEY (run_id, idempotency_key)
    )""",
    """CREATE TABLE IF NOT EXISTS run_performance (
        run_id TEXT PRIMARY KEY REFERENCES evaluation_runs(run_id) ON DELETE RESTRICT,
        revision INTEGER NOT NULL, metrics_json TEXT NOT NULL, updated_at TEXT NOT NULL
    )""",
)


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...]

    @property
    def checksum(self) -> str:
        return hashlib.sha256("\0".join(self.statements).encode()).hexdigest()


MIGRATIONS = (Migration(1, "api_v1_complete_schema", INITIAL_STATEMENTS),)

_TABLE_COLUMNS = {
    "schema_migrations": {"version", "name", "checksum", "applied_at", "applied_by"},
    "evaluation_runs": {
        "run_id",
        "publisher_subject",
        "idempotency_key",
        "create_digest",
        "identity_json",
        "identity_digest",
        "suite",
        "task_name",
        "task_version",
        "split_name",
        "fewshot",
        "cot_mode",
        "repair_strategy",
        "model_name",
        "checkpoint_digest",
        "provider_revision",
        "config_digest",
        "dataset_digest",
        "eligibility",
        "comparable",
        "expected_sample_count",
        "sample_set_digest",
        "status",
        "revision",
        "ingest_digest",
        "created_at",
        "updated_at",
        "completed_at",
    },
    "sample_evidence": {
        "run_id",
        "sample_id",
        "attempt",
        "status",
        "prompt",
        "raw_completion",
        "scored_completion",
        "generation_json",
        "scoring_json",
        "metrics_json",
        "provenance_json",
        "evidence_digest",
        "error_code",
        "error_message",
    },
    "run_aggregates": {
        "run_id",
        "accounting_json",
        "metrics_json",
        "truncated_samples",
        "generated_samples",
        "manifest_digest",
        "payload_digest",
    },
    "run_metrics": {"run_id", "metric_name", "metric_value", "is_primary"},
    "ingest_receipts": {
        "run_id",
        "idempotency_key",
        "request_digest",
        "response_status",
        "response_revision",
        "created_at",
    },
    "run_performance": {"run_id", "revision", "metrics_json", "updated_at"},
}
_REQUIRED_INDEXES = {
    "idx_runs_leaderboard",
    "idx_runs_model",
    "idx_runs_task",
    "idx_metrics_query",
    "idx_samples_page",
}
_CHILD_TABLES = {
    "sample_evidence",
    "run_aggregates",
    "run_metrics",
    "ingest_receipts",
    "run_performance",
}
_PRIMARY_KEYS = {
    "schema_migrations": ("version",),
    "evaluation_runs": ("run_id",),
    "sample_evidence": ("run_id", "sample_id", "attempt"),
    "run_aggregates": ("run_id",),
    "run_metrics": ("run_id", "metric_name"),
    "ingest_receipts": ("run_id", "idempotency_key"),
    "run_performance": ("run_id",),
}
_NULLABLE_COLUMNS = {
    "evaluation_runs": {"ingest_digest", "completed_at"},
    "sample_evidence": {"error_code", "error_message"},
}
_INTEGER_COLUMNS = {
    ("schema_migrations", "version"),
    ("evaluation_runs", "fewshot"),
    ("evaluation_runs", "comparable"),
    ("evaluation_runs", "expected_sample_count"),
    ("evaluation_runs", "revision"),
    ("sample_evidence", "attempt"),
    ("run_aggregates", "truncated_samples"),
    ("run_aggregates", "generated_samples"),
    ("run_metrics", "is_primary"),
    ("ingest_receipts", "response_revision"),
    ("run_performance", "revision"),
}


async def migration_state(database: Database) -> str:
    async with database.read() as session:
        if not await _table_exists(
            session, database.target.backend, "schema_migrations"
        ):
            return "missing"
        rows = await session.fetchall(
            "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
        )
        state = _ledger_state(rows)
        if state != "ready":
            return state
        return (
            "ready"
            if await _schema_shape_is_valid(session, database.target.backend)
            else "drift"
        )


async def apply_migrations(database: Database, *, subject: str) -> str:
    async with database.transaction(immediate=True) as session:
        if database.target.backend == "postgres":
            await session.execute("SELECT pg_advisory_xact_lock(749238501)")
        await _reject_unledgered_schema(session, database.target.backend)
        await _bootstrap_ledger(session)
        rows = await session.fetchall(
            "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
        )
        state = _ledger_state(rows)
        if state in {"ahead", "drift"}:
            raise ValueError(f"database migration ledger is {state}")
        applied = {int(row["version"]): row for row in rows}
        changed = False
        for migration in MIGRATIONS:
            if migration.version in applied:
                continue
            for statement in migration.statements:
                await session.execute(statement)
            await session.execute(
                "INSERT INTO schema_migrations(version, name, checksum, applied_at, applied_by) VALUES (?, ?, ?, ?, ?)",
                (
                    migration.version,
                    migration.name,
                    migration.checksum,
                    datetime.now(UTC).isoformat(),
                    subject,
                ),
            )
            changed = True
        if not await _schema_shape_is_valid(session, database.target.backend):
            raise ValueError(
                "migration postcondition failed: schema fingerprint mismatch"
            )
        return "applied" if changed else "unchanged"


def _ledger_state(rows: list[dict]) -> str:
    applied = {int(row["version"]): row for row in rows}
    expected_versions = {migration.version for migration in MIGRATIONS}
    if set(applied) - expected_versions:
        return "ahead"
    for migration in MIGRATIONS:
        row = applied.get(migration.version)
        if row is None:
            return "outdated"
        if row["name"] != migration.name or row["checksum"] != migration.checksum:
            return "drift"
    return "ready"


async def _reject_unledgered_schema(session: DatabaseSession, backend: str) -> None:
    ledger = await _table_exists(session, backend, "schema_migrations")
    product = await _table_exists(session, backend, "evaluation_runs")
    if product and not ledger:
        raise ValueError("refusing to certify an existing unledgered scoreboard schema")


async def _table_exists(session: DatabaseSession, backend: str, name: str) -> bool:
    if backend == "sqlite":
        row = await session.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        )
    else:
        row = await session.fetchone(
            "SELECT table_name AS name FROM information_schema.tables "
            "WHERE table_schema='public' AND table_name=?",
            (name,),
        )
    return bool(row and row.get("name"))


@dataclass(frozen=True, slots=True)
class _ColumnShape:
    data_type: str
    nullable: bool


@dataclass(frozen=True, slots=True)
class _ForeignKeyShape:
    owner: str
    columns: tuple[str, ...]
    target: str
    target_columns: tuple[str, ...]
    delete_rule: str


@dataclass(frozen=True, slots=True)
class _SchemaFingerprint:
    columns: dict[str, dict[str, _ColumnShape]]
    primary_keys: dict[str, tuple[str, ...]]
    unique_keys: frozenset[tuple[str, tuple[str, ...]]]
    foreign_keys: frozenset[_ForeignKeyShape]
    indexes: frozenset[str]


async def _schema_shape_is_valid(session: DatabaseSession, backend: str) -> bool:
    fingerprint = (
        await _read_sqlite_fingerprint(session)
        if backend == "sqlite"
        else await _read_postgres_fingerprint(session)
    )
    return fingerprint is not None and _matches_expected_schema(fingerprint)


async def _read_sqlite_fingerprint(
    session: DatabaseSession,
) -> _SchemaFingerprint | None:
    columns: dict[str, dict[str, _ColumnShape]] = {}
    primary_keys: dict[str, tuple[str, ...]] = {}
    for table in _TABLE_COLUMNS:
        if not await _table_exists(session, "sqlite", table):
            return None
        rows = await session.fetchall(f"PRAGMA table_info({table})")
        columns[table] = {
            row["name"]: _ColumnShape(
                str(row["type"]).lower(), not bool(row["notnull"])
            )
            for row in rows
        }
        primary_keys[table] = tuple(
            row["name"]
            for row in sorted(rows, key=lambda item: item["pk"])
            if row["pk"]
        )

    index_rows = await session.fetchall(
        "SELECT name FROM sqlite_master WHERE type='index' AND name IS NOT NULL"
    )
    unique_keys: set[tuple[str, tuple[str, ...]]] = set()
    for index in await session.fetchall("PRAGMA index_list(evaluation_runs)"):
        if index["unique"]:
            members = await session.fetchall(f"PRAGMA index_info({index['name']})")
            unique_keys.add(
                ("evaluation_runs", tuple(item["name"] for item in members))
            )

    foreign_keys: set[_ForeignKeyShape] = set()
    for table in _CHILD_TABLES:
        rows = await session.fetchall(f"PRAGMA foreign_key_list({table})")
        grouped: dict[int, list[dict]] = {}
        for row in rows:
            grouped.setdefault(int(row["id"]), []).append(row)
        for members in grouped.values():
            ordered = sorted(members, key=lambda item: item["seq"])
            foreign_keys.add(
                _ForeignKeyShape(
                    owner=table,
                    columns=tuple(item["from"] for item in ordered),
                    target=str(ordered[0]["table"]),
                    target_columns=tuple(item["to"] for item in ordered),
                    delete_rule=str(ordered[0]["on_delete"]).upper(),
                )
            )
    return _SchemaFingerprint(
        columns=columns,
        primary_keys=primary_keys,
        unique_keys=frozenset(unique_keys),
        foreign_keys=frozenset(foreign_keys),
        indexes=frozenset(row["name"] for row in index_rows),
    )


async def _read_postgres_fingerprint(
    session: DatabaseSession,
) -> _SchemaFingerprint | None:
    columns: dict[str, dict[str, _ColumnShape]] = {}
    for table in _TABLE_COLUMNS:
        if not await _table_exists(session, "postgres", table):
            return None
        rows = await session.fetchall(
            "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name=?",
            (table,),
        )
        columns[table] = {
            row["column_name"]: _ColumnShape(
                str(row["data_type"]).lower(), row["is_nullable"] == "YES"
            )
            for row in rows
        }

    constraints = await session.fetchall(
        "SELECT tc.constraint_name, tc.table_name, tc.constraint_type, "
        "kcu.column_name, kcu.ordinal_position, ccu.table_name AS foreign_table_name, "
        "ccu.column_name AS foreign_column_name, rc.delete_rule "
        "FROM information_schema.table_constraints tc "
        "LEFT JOIN information_schema.key_column_usage kcu ON tc.constraint_name=kcu.constraint_name "
        "AND tc.constraint_schema=kcu.constraint_schema "
        "LEFT JOIN information_schema.constraint_column_usage ccu ON tc.constraint_name=ccu.constraint_name "
        "AND tc.constraint_schema=ccu.constraint_schema "
        "LEFT JOIN information_schema.referential_constraints rc "
        "ON tc.constraint_name=rc.constraint_name AND tc.constraint_schema=rc.constraint_schema "
        "WHERE tc.constraint_schema='public'"
    )
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in constraints:
        grouped.setdefault(
            (row["table_name"], row["constraint_type"], row["constraint_name"]), []
        ).append(row)

    primary_keys: dict[str, tuple[str, ...]] = {}
    unique_keys: set[tuple[str, tuple[str, ...]]] = set()
    foreign_keys: set[_ForeignKeyShape] = set()
    for (owner, kind, _name), rows in grouped.items():
        if kind == "PRIMARY KEY":
            primary_keys[owner] = _ordered_columns(rows)
        elif kind == "UNIQUE":
            unique_keys.add((owner, _ordered_columns(rows)))
        elif kind == "FOREIGN KEY":
            first = rows[0]
            foreign_keys.add(
                _ForeignKeyShape(
                    owner=owner,
                    columns=_ordered_columns(rows),
                    target=str(first["foreign_table_name"]),
                    target_columns=tuple(
                        sorted(
                            {
                                str(row["foreign_column_name"])
                                for row in rows
                                if row["foreign_column_name"] is not None
                            }
                        )
                    ),
                    delete_rule=str(first["delete_rule"]).upper(),
                )
            )
    index_rows = await session.fetchall(
        "SELECT indexname FROM pg_indexes WHERE schemaname='public'"
    )
    return _SchemaFingerprint(
        columns=columns,
        primary_keys=primary_keys,
        unique_keys=frozenset(unique_keys),
        foreign_keys=frozenset(foreign_keys),
        indexes=frozenset(row["indexname"] for row in index_rows),
    )


def _matches_expected_schema(fingerprint: _SchemaFingerprint) -> bool:
    if set(fingerprint.columns) != set(_TABLE_COLUMNS):
        return False
    for table, expected_columns in _TABLE_COLUMNS.items():
        actual_columns = fingerprint.columns[table]
        if set(actual_columns) != expected_columns:
            return False
        if fingerprint.primary_keys.get(table) != _PRIMARY_KEYS[table]:
            return False
        for column, shape in actual_columns.items():
            if shape.data_type != _column_type(table, column):
                return False
            if column not in _PRIMARY_KEYS[table] and shape.nullable != (
                column in _NULLABLE_COLUMNS.get(table, set())
            ):
                return False
    if _REQUIRED_INDEXES - fingerprint.indexes:
        return False
    if (
        "evaluation_runs",
        ("publisher_subject", "idempotency_key"),
    ) not in fingerprint.unique_keys:
        return False
    for table in _CHILD_TABLES:
        expected = _ForeignKeyShape(
            owner=table,
            columns=("run_id",),
            target="evaluation_runs",
            target_columns=("run_id",),
            delete_rule="RESTRICT",
        )
        owned = {item for item in fingerprint.foreign_keys if item.owner == table}
        if owned != {expected}:
            return False
    return True


def _column_type(table: str, column: str) -> str:
    if (table, column) == ("run_metrics", "metric_value"):
        return "double precision"
    if (table, column) in _INTEGER_COLUMNS:
        return "integer"
    return "text"


def _ordered_columns(rows: list[dict]) -> tuple[str, ...]:
    distinct = {
        (int(row["ordinal_position"]), str(row["column_name"]))
        for row in rows
        if row["ordinal_position"] is not None and row["column_name"] is not None
    }
    return tuple(column for _position, column in sorted(distinct))


async def _bootstrap_ledger(session: DatabaseSession) -> None:
    await session.execute(INITIAL_STATEMENTS[0])
