import hashlib
import math
import re


EMBEDDING_SIZE = 384
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def embed_text(text: str) -> list[float]:
    """
    Simple local embedding using token hashing.

    FOR TESTING ONLY - produces consistent 384-dimensional vectors while keeping
    similar texts close when they share words. In production, replace this with
    a real embedding model.
    """
    embedding = [0.0] * EMBEDDING_SIZE
    tokens = TOKEN_PATTERN.findall(text.lower())

    for token in tokens:
        token_hash = int(hashlib.sha256(token.encode()).hexdigest(), 16)
        index = token_hash % EMBEDDING_SIZE
        sign = 1.0 if (token_hash // EMBEDDING_SIZE) % 2 == 0 else -1.0
        embedding[index] += sign

    norm = math.sqrt(sum(value * value for value in embedding))
    if norm == 0:
        return embedding

    embedding = [value / norm for value in embedding]
    return embedding


def get_embedding_size() -> int:
    """
    Return the vector size for the current embedding model.
    """
    return EMBEDDING_SIZE
