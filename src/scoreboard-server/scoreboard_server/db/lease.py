from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tortoise import Tortoise

from .connection import init_db
from .settings import DatabaseSettings


@dataclass(frozen=True, slots=True)
class SchedulerLeaseRecord:
    job_id: str
    owner_id: str
    node_id: str
    claimed_at: object
    heartbeat_at: object
    lease_until: object
    lease_meta: Mapping[str, Any] | None = None


class SchedulerLeaseStore:
    def __init__(self, settings: DatabaseSettings | None = None) -> None:
        self.settings = settings or DatabaseSettings.from_env()

    async def _connection(self):
        await init_db(self.settings)
        return Tortoise.get_connection("default")

    async def claim(
        self,
        *,
        job_id: str,
        owner_id: str,
        node_id: str,
        lease_duration_s: int,
        lease_meta: Mapping[str, Any] | None = None,
    ) -> bool:
        connection = await self._connection()
        rows = await connection.execute_query_dict(
            """
            INSERT INTO scheduler_lease (
                job_id,
                owner_id,
                node_id,
                claimed_at,
                heartbeat_at,
                lease_until,
                lease_meta
            )
            VALUES (
                $1,
                $2,
                $3,
                NOW(),
                NOW(),
                NOW() + make_interval(secs => $4),
                $5::jsonb
            )
            ON CONFLICT (job_id) DO UPDATE
            SET owner_id = EXCLUDED.owner_id,
                node_id = EXCLUDED.node_id,
                claimed_at = NOW(),
                heartbeat_at = NOW(),
                lease_until = NOW() + make_interval(secs => $6),
                lease_meta = EXCLUDED.lease_meta
            WHERE scheduler_lease.owner_id = EXCLUDED.owner_id
               OR scheduler_lease.lease_until <= NOW()
            RETURNING job_id
            """,
            [
                str(job_id),
                str(owner_id),
                str(node_id),
                int(lease_duration_s),
                json.dumps(dict(lease_meta), ensure_ascii=False) if lease_meta is not None else None,
                int(lease_duration_s),
            ],
        )
        return bool(rows)

    async def renew(self, *, job_ids: Sequence[str], owner_id: str, lease_duration_s: int) -> set[str]:
        normalized = [str(job_id) for job_id in job_ids if str(job_id).strip()]
        if not normalized:
            return set()
        connection = await self._connection()
        rows = await connection.execute_query_dict(
            """
            UPDATE scheduler_lease
            SET heartbeat_at = NOW(),
                lease_until = NOW() + make_interval(secs => $1)
            WHERE owner_id = $2
              AND job_id = ANY($3::text[])
              AND lease_until > NOW()
            RETURNING job_id
            """,
            [int(lease_duration_s), str(owner_id), normalized],
        )
        return {str(row["job_id"]) for row in rows}

    async def release(self, *, job_ids: Sequence[str], owner_id: str) -> int:
        normalized = [str(job_id) for job_id in job_ids if str(job_id).strip()]
        if not normalized:
            return 0
        connection = await self._connection()
        rows = await connection.execute_query_dict(
            """
            DELETE FROM scheduler_lease
            WHERE owner_id = $1
              AND job_id = ANY($2::text[])
            RETURNING job_id
            """,
            [str(owner_id), normalized],
        )
        return len(rows)

    async def release_all(self, *, owner_id: str) -> int:
        connection = await self._connection()
        rows = await connection.execute_query_dict(
            """
            DELETE FROM scheduler_lease
            WHERE owner_id = $1
            RETURNING job_id
            """,
            [str(owner_id)],
        )
        return len(rows)

    async def list_active(self) -> list[SchedulerLeaseRecord]:
        connection = await self._connection()
        rows = await connection.execute_query_dict(
            """
            SELECT
                job_id,
                owner_id,
                node_id,
                claimed_at,
                heartbeat_at,
                lease_until,
                lease_meta
            FROM scheduler_lease
            WHERE lease_until > NOW()
            ORDER BY job_id
            """
        )
        return [
            SchedulerLeaseRecord(
                job_id=str(row["job_id"]),
                owner_id=str(row["owner_id"]),
                node_id=str(row["node_id"]),
                claimed_at=row["claimed_at"],
                heartbeat_at=row["heartbeat_at"],
                lease_until=row["lease_until"],
                lease_meta=row.get("lease_meta") if isinstance(row.get("lease_meta"), dict) else None,
            )
            for row in rows
        ]


class SchedulerLeaseManager:
    def __init__(
        self,
        store: SchedulerLeaseStore | None = None,
        *,
        node_id: str,
        owner_id: str,
        lease_duration_s: int = 120,
    ) -> None:
        self.store = store or SchedulerLeaseStore()
        self.node_id = str(node_id)
        self.owner_id = str(owner_id)
        self.lease_duration_s = max(5, int(lease_duration_s))

    async def claim(self, job_id: str, *, lease_meta: Mapping[str, Any] | None = None) -> bool:
        return await self.store.claim(
            job_id=job_id,
            owner_id=self.owner_id,
            node_id=self.node_id,
            lease_duration_s=self.lease_duration_s,
            lease_meta=lease_meta,
        )

    async def renew(self, job_ids: Sequence[str]) -> set[str]:
        return await self.store.renew(
            job_ids=job_ids,
            owner_id=self.owner_id,
            lease_duration_s=self.lease_duration_s,
        )

    async def release(self, job_ids: Sequence[str]) -> int:
        return await self.store.release(job_ids=job_ids, owner_id=self.owner_id)

    async def release_all(self) -> int:
        return await self.store.release_all(owner_id=self.owner_id)

    async def active_foreign_job_ids(self) -> set[str]:
        return {lease.job_id for lease in await self.store.list_active() if lease.owner_id != self.owner_id}


__all__ = ["SchedulerLeaseManager", "SchedulerLeaseRecord", "SchedulerLeaseStore"]
