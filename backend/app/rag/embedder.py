import os
from pathlib import Path
import time

import httpx
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[2] / ".env")

OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_SIZE = 384
DEFAULT_EMBEDDING_BATCH_SIZE = 16
DEFAULT_EMBEDDING_RETRY_SECONDS = 8.0
MAX_EMBEDDING_RETRIES = 6


def embed_text(text: str) -> list[float]:
    """
    Embed one text with OpenAI embeddings.

    text-embedding-3-small supports custom output dimensions, so the default
    stays at 384 dimensions to match the current Qdrant collection schema.
    """
    return embed_texts([text])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to create embeddings.")

    if not texts:
        return []

    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", str(DEFAULT_EMBEDDING_BATCH_SIZE)))
    embeddings: list[list[float]] = []

    for start_index in range(0, len(texts), batch_size):
        batch = texts[start_index : start_index + batch_size]
        embeddings.extend(_embed_text_batch(batch))

    return embeddings


def _embed_text_batch(texts: list[str]) -> list[list[float]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to create embeddings.")

    model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", str(EMBEDDING_SIZE)))

    for attempt in range(MAX_EMBEDDING_RETRIES + 1):
        response = httpx.post(
            OPENAI_EMBEDDINGS_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "input": texts,
                "encoding_format": "float",
                "dimensions": dimensions,
            },
            timeout=60.0,
        )

        if response.status_code != 429:
            break

        if attempt == MAX_EMBEDDING_RETRIES:
            break

        time.sleep(_retry_delay(response, attempt))

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"OpenAI embeddings request failed: {response.text}") from exc

    data = response.json()["data"]
    embeddings = [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]

    for embedding in embeddings:
        if len(embedding) != dimensions:
            raise RuntimeError(
                f"Embedding model '{model}' returned {len(embedding)} dimensions, "
                f"expected {dimensions}."
            )

    return embeddings


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass

    default_delay = float(
        os.getenv("EMBEDDING_RETRY_SECONDS", str(DEFAULT_EMBEDDING_RETRY_SECONDS))
    )
    return default_delay * (attempt + 1)


def get_embedding_size() -> int:
    """
    Return the vector size for the current embedding model.
    """
    return int(os.getenv("EMBEDDING_DIMENSIONS", str(EMBEDDING_SIZE)))
