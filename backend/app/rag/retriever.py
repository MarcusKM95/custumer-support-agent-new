import re

from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.db.qdrant_client import get_qdrant_client
from app.rag.embedder import embed_text
from app.rag.sources import COLLECTION_NAME


SECTION_ALIASES = {
    "Bingo": ["bingo"],
    "Casino": ["casino", "blackjack", "chatten", "chat", "karantæne", "banned", "ban"],
    "Dantoto": ["dantoto", "heste", "hundevæddeløb", "hestevæddeløb"],
    "LiveCasino": ["livecasino", "live casino", "live blackjack"],
    "Måljagt": ["måljagt", "maaljagt"],
    "Oddset": ["oddset", "væddemål", "odds", "ob", "scorede", "mål", "udskiftet"],
    "Poker": ["poker"],
    "Tips": ["tips", "tippe"],
}

TOKEN_PATTERN = re.compile(r"[a-zæøå0-9]+")
STOPWORDS = {
    "at",
    "blev",
    "det",
    "en",
    "er",
    "for",
    "har",
    "han",
    "i",
    "inden",
    "jeg",
    "lige",
    "men",
    "og",
    "på",
    "som",
    "synes",
    "var",
}
FOOTBALL_QUERY_MARKERS = {
    "fck",
    "ob",
    "kampen",
    "mål",
    "score",
    "scorede",
    "spiller",
    "udskiftet",
    "skiftet",
}
FOOTBALL_RULE_MARKERS = {
    "fodbold",
    "målscorer",
    "selvmål",
    "straffesparksfeltet",
    "frispark",
    "hjørnespark",
    "halvleg",
    "var",
    "udskiftet",
    "spilletid",
    "navngivne spiller",
}
NON_FOOTBALL_MARKERS = {
    "assists",
    "blocked shots",
    "double/double",
    "greyhound",
    "heste",
    "håndbold",
    "rebounds",
    "steals",
    "triple/double",
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
            limit=max(limit * 40, 100),
            with_payload=True,
        )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Could not query Qdrant: {e}") from e

    if not response.points:
        return []

    results = [
        {
            "score": result.score,
            "text": result.payload.get("text") if result.payload else None,
            "source": result.payload.get("source") if result.payload else None,
            "source_url": result.payload.get("source_url") if result.payload else None,
            "section": result.payload.get("section") if result.payload else None,
            "rule_number": result.payload.get("rule_number") if result.payload else None,
            "keywords": result.payload.get("keywords") if result.payload else None,
            "chunk_index": result.payload.get("chunk_index") if result.payload else None,
        }
        for result in response.points
    ]

    return rerank_results(query, results)[:limit]


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


def rerank_results(query: str, results: list[dict]) -> list[dict]:
    query_tokens = meaningful_tokens(query)
    football_query = bool(query_tokens & FOOTBALL_QUERY_MARKERS)

    def rerank_score(result: dict) -> float:
        text = result.get("text") or ""
        text_tokens = meaningful_tokens(text)
        overlap = len(query_tokens & text_tokens)
        lexical_score = overlap / max(len(query_tokens), 1)

        phrase_boost = 0.0
        lower_text = text.casefold()
        lower_query = query.casefold()

        for phrase in ["udskiftet", "spilletid", "navngivne spiller", "score", "mål"]:
            if phrase in lower_query and phrase in lower_text:
                phrase_boost += 0.08

        if "skiftet ud" in lower_query and "udskiftet" in lower_text:
            phrase_boost += 0.2

        if football_query:
            phrase_boost += 0.06 * sum(
                1 for marker in FOOTBALL_RULE_MARKERS if marker in lower_text
            )
            phrase_boost -= 0.12 * sum(
                1 for marker in NON_FOOTBALL_MARKERS if marker in lower_text
            )

        return float(result.get("score") or 0.0) + lexical_score + phrase_boost

    return sorted(results, key=rerank_score, reverse=True)


def meaningful_tokens(text: str) -> set[str]:
    return {
        token
        for token in TOKEN_PATTERN.findall(text.casefold())
        if len(token) > 1 and token not in STOPWORDS
    }
