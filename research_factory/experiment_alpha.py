from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Iterator


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS experiment_alpha_campaigns (
    campaign_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    max_hypotheses INTEGER NOT NULL,
    evidence_label TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS experiment_alpha_slots (
    experiment_id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    slot_number INTEGER NOT NULL,
    task_sha256 TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(campaign_id, slot_number),
    FOREIGN KEY(campaign_id) REFERENCES experiment_alpha_campaigns(campaign_id)
);
"""


class ExperimentAlphaStore:
    """Independent alpha ledger for non-factory experiments."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def ensure_campaign(
        self,
        campaign_id: str,
        dataset_id: str,
        max_hypotheses: int,
        evidence_label: str,
    ) -> None:
        if max_hypotheses <= 0:
            raise ValueError("invalid experiment alpha campaign budget")
        expected = (dataset_id, int(max_hypotheses), evidence_label)
        with self.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO experiment_alpha_campaigns("
                "campaign_id, dataset_id, max_hypotheses, evidence_label"
                ") VALUES (?, ?, ?, ?)",
                (campaign_id, *expected),
            )
            row = connection.execute(
                "SELECT dataset_id, max_hypotheses, evidence_label "
                "FROM experiment_alpha_campaigns WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
        if row is None or tuple(row) != expected:
            raise RuntimeError("experiment alpha campaign is immutable once created")

    def reserve_slot(
        self,
        campaign_id: str,
        experiment_id: str,
        *,
        task_sha256: str,
        manifest_sha256: str,
    ) -> dict:
        if not experiment_id.strip() or len(task_sha256) != 64 or len(manifest_sha256) != 64:
            raise ValueError("invalid experiment alpha slot identity")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT campaign_id, slot_number, task_sha256, manifest_sha256 "
                "FROM experiment_alpha_slots WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
            if existing is not None:
                actual = dict(existing)
                if (
                    actual["campaign_id"] != campaign_id
                    or actual["task_sha256"] != task_sha256
                    or actual["manifest_sha256"] != manifest_sha256
                ):
                    raise RuntimeError("experiment alpha slot identity mismatch")
                return actual
            campaign = connection.execute(
                "SELECT max_hypotheses FROM experiment_alpha_campaigns "
                "WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if campaign is None:
                raise KeyError(campaign_id)
            used = int(
                connection.execute(
                    "SELECT COUNT(*) FROM experiment_alpha_slots WHERE campaign_id = ?",
                    (campaign_id,),
                ).fetchone()[0]
            )
            if used >= int(campaign["max_hypotheses"]):
                raise RuntimeError(f"experiment alpha budget exhausted for {campaign_id}")
            slot_number = used + 1
            connection.execute(
                "INSERT INTO experiment_alpha_slots("
                "experiment_id, campaign_id, slot_number, task_sha256, manifest_sha256"
                ") VALUES (?, ?, ?, ?, ?)",
                (experiment_id, campaign_id, slot_number, task_sha256, manifest_sha256),
            )
            return {
                "campaign_id": campaign_id,
                "slot_number": slot_number,
                "task_sha256": task_sha256,
                "manifest_sha256": manifest_sha256,
            }

    def campaign_status(self, campaign_id: str) -> dict:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT c.campaign_id, c.dataset_id, c.max_hypotheses, "
                "c.evidence_label, COUNT(s.experiment_id) AS used_hypotheses "
                "FROM experiment_alpha_campaigns c "
                "LEFT JOIN experiment_alpha_slots s USING(campaign_id) "
                "WHERE c.campaign_id = ? GROUP BY c.campaign_id",
                (campaign_id,),
            ).fetchone()
        if row is None:
            raise KeyError(campaign_id)
        return dict(row)
