from datetime import UTC, datetime
from threading import Lock
from time import perf_counter

from app.rag.ingest_danske_spil import ingest_danske_spil_rules


class ReindexInProgressError(RuntimeError):
    pass


_reindex_lock = Lock()


def reindex_knowledge_base() -> dict:
    if not _reindex_lock.acquire(blocking=False):
        raise ReindexInProgressError("A knowledge base re-index is already running.")

    started_at = datetime.now(UTC)
    started_timer = perf_counter()
    try:
        chunk_count = ingest_danske_spil_rules(recreate=True)
        return {
            "status": "completed",
            "chunks_indexed": chunk_count,
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
            "duration_seconds": round(perf_counter() - started_timer, 2),
        }
    finally:
        _reindex_lock.release()
