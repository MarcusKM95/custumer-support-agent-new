from app.db.qdrant_client import get_qdrant_client
from app.rag.embedder import embed_text
from app.rag.ingest_test_data import COLLECTION_NAME


def search_guidelines(query: str, limit: int = 3) -> list[dict]:
    """
    Search the Qdrant test collection for guideline chunks relevant to the query.
    """
    client = get_qdrant_client()
    query_vector = embed_text(query)

    try:
        if not client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Qdrant collection '{COLLECTION_NAME}' does not exist. "
                "Run the test data ingestion script first."
            )

        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Could not query Qdrant: {e}") from e

    if not response.points:
        return []

    return [
        {
            "score": result.score,
            "text": result.payload.get("text") if result.payload else None,
            "source": result.payload.get("source") if result.payload else None,
            "chunk_index": result.payload.get("chunk_index") if result.payload else None,
        }
        for result in response.points
    ]
