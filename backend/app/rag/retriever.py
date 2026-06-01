from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.db.qdrant_client import get_qdrant_client
from app.rag.embedder import embed_text
from app.rag.sources import COLLECTION_NAME


SECTION_ALIASES = {
    "Bingo": ["bingo"],
    "Casino": ["casino"],
    "Dantoto": ["dantoto", "heste", "hundevæddeløb", "hestevæddeløb"],
    "LiveCasino": ["livecasino", "live casino"],
    "Måljagt": ["måljagt", "maaljagt"],
    "Oddset": ["oddset", "væddemål", "odds"],
    "Poker": ["poker"],
    "Tips": ["tips", "tippe"],
}


def search_guidelines(query: str, limit: int = 3) -> list[dict]:
    """
    Search Qdrant for rules chunks relevant to the query.
    """
    client = get_qdrant_client()
    query_vector = embed_text(query)
    query_filter = build_section_filter(query)

    try:
        if not client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Qdrant collection '{COLLECTION_NAME}' does not exist. "
                "Run `python -m app.rag.ingest_danske_spil` first."
            )

        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
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
            "source_url": result.payload.get("source_url") if result.payload else None,
            "section": result.payload.get("section") if result.payload else None,
            "chunk_index": result.payload.get("chunk_index") if result.payload else None,
        }
        for result in response.points
    ]


def build_section_filter(query: str) -> Filter | None:
    normalized_query = query.casefold()

    section_aliases = [
        (section, alias)
        for section, aliases in SECTION_ALIASES.items()
        for alias in aliases
    ]

    for section, alias in sorted(section_aliases, key=lambda item: len(item[1]), reverse=True):
        if alias.casefold() in normalized_query:
            return Filter(
                must=[
                    FieldCondition(
                        key="section",
                        match=MatchValue(value=section),
                    )
                ]
            )

    return None
