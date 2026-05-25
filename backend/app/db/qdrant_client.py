from qdrant_client import QdrantClient


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host="localhost", port=6333)


def check_qdrant_connection() -> dict:
    client = get_qdrant_client()
    collections = client.get_collections()

    return {
        "status": "connected",
        "collections": [collection.name for collection in collections.collections],
    }