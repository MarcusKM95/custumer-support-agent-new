import os

from app.db.qdrant_client import get_qdrant_client
from app.rag.embedder import DEFAULT_EMBEDDING_MODEL, get_embedding_size
from app.rag.sources import COLLECTION_NAME


def get_knowledge_base_status() -> dict:
    client = get_qdrant_client()

    if not client.collection_exists(COLLECTION_NAME):
        return {
            "status": "missing",
            "collection": COLLECTION_NAME,
            "total_chunks": 0,
            "products": [],
            "embedding_model": os.getenv(
                "EMBEDDING_MODEL",
                DEFAULT_EMBEDDING_MODEL,
            ),
            "embedding_dimensions": get_embedding_size(),
            "last_indexed_at": None,
        }

    products: dict[str, dict] = {}
    last_indexed_at = None
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            section = payload.get("section") or "Unknown"
            product = products.setdefault(
                section,
                {
                    "name": section,
                    "chunks": 0,
                    "source_url": payload.get("source_url"),
                },
            )
            product["chunks"] += 1

            indexed_at = payload.get("indexed_at")
            if indexed_at and (last_indexed_at is None or indexed_at > last_indexed_at):
                last_indexed_at = indexed_at

        if offset is None:
            break

    total_chunks = sum(product["chunks"] for product in products.values())

    return {
        "status": "ready" if total_chunks else "empty",
        "collection": COLLECTION_NAME,
        "total_chunks": total_chunks,
        "products": sorted(products.values(), key=lambda product: product["name"]),
        "embedding_model": os.getenv(
            "EMBEDDING_MODEL",
            DEFAULT_EMBEDDING_MODEL,
        ),
        "embedding_dimensions": get_embedding_size(),
        "last_indexed_at": last_indexed_at,
    }
