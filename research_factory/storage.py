from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
from typing import Iterator

from .models import CandidateManifest, CandidateStatus, GateResult, Mechanism, canonical_json


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS mechanisms (
    mechanism_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    semantic_fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    semantic_fingerprint TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS gate_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    gate_name TEXT NOT NULL,
    passed INTEGER NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
);
CREATE TABLE IF NOT EXISTS ai_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS telemetry (
    observed_at TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS research_cycles (
    cycle_id TEXT PRIMARY KEY,
    trigger_reason TEXT NOT NULL,
    baseline_effective_events INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS alpha_campaigns (
    campaign_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    max_hypotheses INTEGER NOT NULL,
    prior_hypotheses INTEGER NOT NULL,
    familywise_alpha REAL NOT NULL,
    evidence_label TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS alpha_slots (
    candidate_id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    slot_number INTEGER NOT NULL,
    alpha_threshold REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(campaign_id, slot_number),
    FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id),
    FOREIGN KEY(campaign_id) REFERENCES alpha_campaigns(campaign_id)
);
"""


class ResearchStore:
    def __init__(self, path: str | Path, *, writable: bool = False) -> None:
        self.path = Path(path)
        self.writable = writable
        if writable:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.connect() as connection:
                connection.executescript(SCHEMA)
        elif not self.path.is_file() or self.path.is_symlink():
            raise FileNotFoundError(f"read-only research database missing: {self.path}")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        if self.writable:
            connection = sqlite3.connect(self.path)
        else:
            connection = sqlite3.connect(
                f"file:{self.path.resolve().as_posix()}?mode=ro&immutable=1",
                uri=True,
            )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            if self.writable:
                connection.commit()
        finally:
            connection.close()

    def add_mechanism(self, mechanism: Mechanism, fingerprint: str) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO mechanisms(mechanism_id, payload, semantic_fingerprint) "
                "VALUES (?, ?, ?)",
                (mechanism.mechanism_id, canonical_json(asdict(mechanism)), fingerprint),
            )
            return cursor.rowcount == 1

    def add_candidate(self, candidate: CandidateManifest) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO candidates(candidate_id, semantic_fingerprint, status, payload) "
                "VALUES (?, ?, ?, ?)",
                (
                    candidate.candidate_id,
                    candidate.semantic_fingerprint,
                    candidate.status.value,
                    canonical_json(asdict(candidate)),
                ),
            )
            return cursor.rowcount == 1

    def set_candidate_status(self, candidate_id: str, status: CandidateStatus) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE candidates SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE candidate_id = ?",
                (status.value, candidate_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(candidate_id)

    def add_gate_result(self, candidate_id: str, result: GateResult) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO gate_results(candidate_id, gate_name, passed, payload) VALUES (?, ?, ?, ?)",
                (candidate_id, result.gate, int(result.passed), canonical_json(asdict(result))),
            )

    def apply_gate_transition(
        self,
        candidate_id: str,
        result: GateResult,
        status: CandidateStatus,
        alpha_campaign_id: str | None = None,
    ) -> dict | None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            slot = None
            if alpha_campaign_id is not None:
                slot = self._reserve_alpha_slot(connection, alpha_campaign_id, candidate_id)
            connection.execute(
                "INSERT INTO gate_results(candidate_id, gate_name, passed, payload) "
                "VALUES (?, ?, ?, ?)",
                (candidate_id, result.gate, int(result.passed), canonical_json(asdict(result))),
            )
            cursor = connection.execute(
                "UPDATE candidates SET status = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE candidate_id = ?",
                (status.value, candidate_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(candidate_id)
        return slot

    def record_usage(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO ai_usage(provider, model, prompt_tokens, completion_tokens, cost_usd) "
                "VALUES (?, ?, ?, ?, ?)",
                (provider, model, prompt_tokens, completion_tokens, cost_usd),
            )

    def monthly_ai_cost(self) -> float:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) AS cost FROM ai_usage "
                "WHERE created_at >= datetime('now', 'start of month')"
            ).fetchone()
        return float(row["cost"])

    def list_candidates(self) -> list[dict[str, str]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT candidate_id, status, created_at, updated_at FROM candidates ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def candidate_count(self) -> int:
        with self.connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM candidates").fetchone()
        return int(row["count"])

    def used_mechanism_ids(self) -> set[str]:
        with self.connect() as connection:
            rows = connection.execute("SELECT payload FROM candidates").fetchall()
        used: set[str] = set()
        for row in rows:
            payload = json.loads(row["payload"])
            used.update(payload["hypothesis"]["mechanism_ids"])
        return used

    def candidate_payload(self, candidate_id: str) -> dict:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload FROM candidates WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        return json.loads(row["payload"])

    def candidate_status(self, candidate_id: str) -> CandidateStatus:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT status FROM candidates WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        return CandidateStatus(row["status"])

    def gate_results(self, candidate_id: str) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM gate_results WHERE candidate_id = ? ORDER BY id",
                (candidate_id,),
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def add_telemetry(self, observed_at: str, payload: dict) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO telemetry(observed_at, payload) VALUES (?, ?)",
                (observed_at, canonical_json(payload)),
            )

    def recent_telemetry(self, limit: int = 30) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT observed_at, payload FROM telemetry ORDER BY observed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"observed_at": row["observed_at"], **json.loads(row["payload"])}
            for row in reversed(rows)
        ]

    def add_cycle(self, cycle_id: str, reason: str, baseline_effective_events: int) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO research_cycles(cycle_id, trigger_reason, baseline_effective_events) "
                "VALUES (?, ?, ?)",
                (cycle_id, reason, baseline_effective_events),
            )

    def latest_cycle(self) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT cycle_id, trigger_reason, baseline_effective_events, created_at "
                "FROM research_cycles ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def enqueue_job(
        self,
        job_id: str,
        idempotency_key: str,
        job_type: str,
        payload: dict,
    ) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO jobs(job_id, idempotency_key, job_type, status, payload) "
                "VALUES (?, ?, ?, 'queued', ?)",
                (job_id, idempotency_key, job_type, canonical_json(payload)),
            )
            return cursor.rowcount == 1

    def begin_idempotent_job(
        self,
        job_id: str,
        idempotency_key: str,
        job_type: str,
        payload: dict,
    ) -> dict:
        if not job_id.strip() or not idempotency_key.strip() or not job_type.strip():
            raise ValueError("job identity fields must be non-empty")
        encoded_payload = canonical_json(payload)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "INSERT OR IGNORE INTO jobs("
                "job_id, idempotency_key, job_type, status, payload"
                ") VALUES (?, ?, ?, 'running', ?)",
                (job_id, idempotency_key, job_type, encoded_payload),
            )
            row = connection.execute(
                "SELECT job_id, idempotency_key, job_type, status, payload, error "
                "FROM jobs WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        if row is None:
            raise RuntimeError("idempotent job claim disappeared")
        if (
            row["job_id"] != job_id
            or row["job_type"] != job_type
            or row["payload"] != encoded_payload
        ):
            raise RuntimeError("idempotency key was already used with different job inputs")
        return {
            "acquired": cursor.rowcount == 1,
            "job_id": row["job_id"],
            "status": row["status"],
            "error": row["error"],
        }

    def claim_job(self, job_type: str) -> dict | None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT job_id, payload FROM jobs WHERE job_type = ? AND status = 'queued' "
                "ORDER BY created_at LIMIT 1",
                (job_type,),
            ).fetchone()
            if row is None:
                return None
            cursor = connection.execute(
                "UPDATE jobs SET status = 'running', updated_at = CURRENT_TIMESTAMP "
                "WHERE job_id = ? AND status = 'queued'",
                (row["job_id"],),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"queued job claim lost for {row['job_id']}")
        return {"job_id": row["job_id"], "payload": json.loads(row["payload"])}

    def finish_job(self, job_id: str, status: str, error: str | None = None) -> None:
        if status not in {"completed", "failed", "blocked"}:
            raise ValueError(f"invalid terminal job status: {status}")
        with self.connect() as connection:
            connection.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = CURRENT_TIMESTAMP WHERE job_id = ?",
                (status, error, job_id),
            )

    def stale_running_jobs(self, older_than_minutes: int = 120) -> list[dict]:
        if older_than_minutes <= 0:
            raise ValueError("older_than_minutes must be positive")
        modifier = f"-{older_than_minutes} minutes"
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT job_id, idempotency_key, job_type, status, error, created_at, updated_at "
                "FROM jobs WHERE status = 'running' "
                "AND updated_at <= datetime('now', ?) ORDER BY updated_at",
                (modifier,),
            ).fetchall()
        return [dict(row) for row in rows]

    def resolve_stale_job(
        self,
        job_id: str,
        status: str,
        reason: str,
        older_than_minutes: int = 120,
    ) -> None:
        if status not in {"blocked", "failed"}:
            raise ValueError("stale job resolution must be blocked or failed")
        if not reason.strip():
            raise ValueError("stale job resolution requires a reason")
        if older_than_minutes <= 0:
            raise ValueError("older_than_minutes must be positive")
        modifier = f"-{older_than_minutes} minutes"
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE job_id = ? AND status = 'running' "
                "AND updated_at <= datetime('now', ?)",
                (status, reason, job_id, modifier),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"job is not stale and running: {job_id}")

    def list_jobs(self, limit: int = 20) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT job_id, job_type, status, error, created_at, updated_at "
                "FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def ensure_alpha_campaign(
        self,
        campaign_id: str,
        dataset_id: str,
        max_hypotheses: int,
        prior_hypotheses: int,
        familywise_alpha: float,
        evidence_label: str,
    ) -> None:
        if max_hypotheses <= 0 or prior_hypotheses < 0 or not 0 < familywise_alpha < 1:
            raise ValueError("invalid alpha campaign budget")
        with self.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO alpha_campaigns("
                "campaign_id, dataset_id, max_hypotheses, prior_hypotheses, "
                "familywise_alpha, evidence_label"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (
                    campaign_id,
                    dataset_id,
                    max_hypotheses,
                    prior_hypotheses,
                    familywise_alpha,
                    evidence_label,
                ),
            )
            row = connection.execute(
                "SELECT dataset_id, max_hypotheses, prior_hypotheses, "
                "familywise_alpha, evidence_label "
                "FROM alpha_campaigns WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
        expected = (
            dataset_id,
            max_hypotheses,
            prior_hypotheses,
            familywise_alpha,
            evidence_label,
        )
        actual = (
            row["dataset_id"],
            row["max_hypotheses"],
            row["prior_hypotheses"],
            row["familywise_alpha"],
            row["evidence_label"],
        )
        if actual != expected:
            raise RuntimeError("alpha campaign is immutable once created")

    def reserve_alpha_slot(self, campaign_id: str, candidate_id: str) -> dict:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            return self._reserve_alpha_slot(connection, campaign_id, candidate_id)

    @staticmethod
    def _reserve_alpha_slot(
        connection: sqlite3.Connection,
        campaign_id: str,
        candidate_id: str,
    ) -> dict:
        existing = connection.execute(
            "SELECT slot_number, alpha_threshold FROM alpha_slots WHERE candidate_id = ?",
            (candidate_id,),
        ).fetchone()
        if existing:
            return dict(existing)
        campaign = connection.execute(
            "SELECT max_hypotheses, prior_hypotheses, familywise_alpha "
            "FROM alpha_campaigns WHERE campaign_id = ?",
            (campaign_id,),
        ).fetchone()
        if campaign is None:
            raise KeyError(campaign_id)
        used = connection.execute(
            "SELECT COUNT(*) AS count FROM alpha_slots WHERE campaign_id = ?",
            (campaign_id,),
        ).fetchone()["count"]
        if used >= campaign["max_hypotheses"]:
            raise RuntimeError(f"alpha hypothesis budget exhausted for {campaign_id}")
        slot = int(used) + 1
        threshold = float(campaign["familywise_alpha"]) / (
            int(campaign["prior_hypotheses"]) + int(campaign["max_hypotheses"])
        )
        connection.execute(
            "INSERT INTO alpha_slots(candidate_id, campaign_id, slot_number, alpha_threshold) "
            "VALUES (?, ?, ?, ?)",
            (candidate_id, campaign_id, slot, threshold),
        )
        return {"slot_number": slot, "alpha_threshold": threshold}

    def alpha_slot(self, candidate_id: str) -> dict:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT s.campaign_id, s.slot_number, s.alpha_threshold, c.dataset_id, "
                "c.max_hypotheses, c.prior_hypotheses, c.familywise_alpha, c.evidence_label "
                "FROM alpha_slots s JOIN alpha_campaigns c USING(campaign_id) "
                "WHERE s.candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"candidate has no preregistered alpha slot: {candidate_id}")
        return dict(row)

    def alpha_campaign_status(self, campaign_id: str) -> dict:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT c.campaign_id, c.dataset_id, c.max_hypotheses, c.prior_hypotheses, "
                "c.familywise_alpha, "
                "c.evidence_label, COUNT(s.candidate_id) AS used_hypotheses "
                "FROM alpha_campaigns c LEFT JOIN alpha_slots s USING(campaign_id) "
                "WHERE c.campaign_id = ? GROUP BY c.campaign_id",
                (campaign_id,),
            ).fetchone()
        if row is None:
            raise KeyError(campaign_id)
        return dict(row)
