"""
Background ingestion jobs
=========================

``POST /ingest`` answers with a job id at once; the files are loaded (OCR
included), embedded and indexed afterwards. Jobs run one at a time on a single
worker thread, so index writes never overlap; queries keep running meanwhile
(the RAG pipeline's index lock lets searches and a write take turns).

Job records are kept in memory (the most recent ``max_jobs``): a restart
forgets them, not the documents they indexed.
"""

import logging
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

FINISHED = ("succeeded", "failed")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IngestJobs:
    """Queue of indexing jobs run one by one in a background thread."""

    def __init__(self, max_jobs: int = 100):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
        self._jobs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self.max_jobs = max(1, max_jobs)

    def submit(
        self,
        work: Callable[[], int],
        files: List[str],
        cleanup: Optional[Callable[[], None]] = None,
    ) -> Tuple[Dict[str, Any], Future]:
        """
        Queue ``work`` (returns the number of chunks indexed).

        ``cleanup`` runs after the job whatever happens, e.g. to delete the
        uploaded files. Returns the job record and a future for callers that
        want to wait.
        """
        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "status": "queued",
            "files": list(files),
            "chunks": None,
            "error": None,
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._forget_old_jobs()
            snapshot = dict(job)
        future = self._executor.submit(self._run, job_id, work, cleanup)
        return snapshot, future

    def _run(self, job_id: str, work: Callable[[], int], cleanup: Optional[Callable[[], None]]) -> None:
        self._update(job_id, status="running", started_at=_now())
        try:
            chunks = work()
            self._update(job_id, status="succeeded", chunks=chunks)
        except Exception as e:
            logger.exception(f"Ingest job {job_id} failed")
            self._update(job_id, status="failed", error=f"{type(e).__name__}: {e}")
        finally:
            if cleanup:
                try:
                    cleanup()
                except Exception as e:
                    logger.warning(f"Ingest job {job_id} cleanup failed: {e}")
            self._update(job_id, finished_at=_now())

    def _update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(fields)

    def _forget_old_jobs(self) -> None:
        """Drop the oldest finished records beyond max_jobs (caller holds the lock)."""
        excess = len(self._jobs) - self.max_jobs
        for job_id in [jid for jid, job in self._jobs.items() if job["status"] in FINISHED][:max(0, excess)]:
            del self._jobs[job_id]

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def list(self) -> List[Dict[str, Any]]:
        """Most recent first."""
        with self._lock:
            return [dict(job) for job in reversed(self._jobs.values())]

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)
