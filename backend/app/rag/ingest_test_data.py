from pathlib import Path
from uuid import uuid4

from qdrant_client.models import Distance, PointStruct, VectorParams

from app.db.qdrant_client import get_qdrant_client
from app.rag.chunker import chunk_text
from app.rag.embedder import embed_text, get_embedding_size


COLLECTION_NAME = "customer_guidelines_test"


def recreate_collection() -> None:
    """
    Delete and recreate the test collection.

    This is fine for test data because we want a clean collection every time.
    Later, for real data, we should use versioning instead of deleting everything.
    """
    client = get_qdrant_client()

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=get_embedding_size(),
            distance=Distance.COSINE,
        ),
    )


def ingest_test_guidelines() -> None:
    """
    Read the local test guideline file, split it into chunks, embed each chunk,
    and store the chunks in Qdrant.
    """
    file_path = Path("app/data/test_guidelines.txt")

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find test guideline file: {file_path}")

    text = file_path.read_text(encoding="utf-8")
    chunks = chunk_text(text)

    points = []

    for index, chunk in enumerate(chunks):
        vector = embed_text(chunk)

        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload={
                    "text": chunk,
                    "source": "test_guidelines.txt",
                    "chunk_index": index,
                },
            )
        )

    client = get_qdrant_client()
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(f"Ingested {len(points)} chunks into Qdrant collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    recreate_collection()
    ingest_test_guidelines()
