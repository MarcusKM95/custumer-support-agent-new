import re
from uuid import uuid4

from qdrant_client.models import Distance, PointStruct, VectorParams

from app.db.qdrant_client import get_qdrant_client
from app.rag.chunker import chunk_rules
from app.rag.embedder import embed_texts, get_embedding_size
from app.rag.html_text import fetch_html_text
from app.rag.sources import COLLECTION_NAME, DANSKE_SPIL_RULES_URL


SECTION_HEADINGS = [
    ("Bingo", "Bingo Spilleregler"),
    ("Casino", "Casino Spilleregler"),
    ("Dantoto", "Dantoto spilleregler"),
    ("LiveCasino", "LiveCasino Spilleregler"),
    ("Måljagt", "Måljagt spilleregler"),
    ("Oddset", "Oddset spilleregler"),
    ("Poker", "Poker Spilleregler"),
    ("Tips", "Tips spilleregler"),
]


def recreate_collection() -> None:
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


def extract_rule_sections(page_text: str) -> list[tuple[str, str]]:
    located_headings = []

    for section_name, heading in SECTION_HEADINGS:
        index = page_text.find(heading)
        if index == -1:
            raise ValueError(f"Could not find section heading: {heading}")
        located_headings.append((index, section_name, heading))

    located_headings.sort(key=lambda item: item[0])
    sections: list[tuple[str, str]] = []

    for position, (start_index, section_name, _heading) in enumerate(located_headings):
        if position + 1 < len(located_headings):
            end_index = located_headings[position + 1][0]
            section_text = page_text[start_index:end_index]
        else:
            section_text = page_text[start_index:]

        sections.append((section_name, clean_rules_text(section_text)))

    return sections


def clean_rules_text(text: str) -> str:
    text = text.replace("Artiklen forsætter nedenfor", "")
    text = text.replace("Artiklen fortsætter nedenfor", "")
    text = re.sub(r"(?m)^(\d+)\.\s*", r"\1. ", text)
    text = re.sub(r"(?<=[a-zæøå])\.(?=[A-ZÆØÅ])", ". ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ingest_danske_spil_rules(recreate: bool = True) -> int:
    """
    Fetch Danske Spil rules, chunk them by product, embed them, and store them in Qdrant.
    """
    if recreate:
        recreate_collection()

    page_text = fetch_html_text(DANSKE_SPIL_RULES_URL)
    sections = extract_rule_sections(page_text)
    points: list[PointStruct] = []

    for section_name, rules_text in sections:
        source_url = f"{DANSKE_SPIL_RULES_URL}#{section_name.lower()}"
        chunks = chunk_rules(rules_text)
        embeddings = embed_texts([chunk["text"] for chunk in chunks])

        for chunk_index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            points.append(
                PointStruct(
                    id=str(uuid4()),
                    vector=embedding,
                    payload={
                        "text": chunk["text"],
                        "source": "Danske Spil DLI spilleregler",
                        "source_url": source_url,
                        "section": section_name,
                        "rule_number": chunk["rule_number"],
                        "keywords": chunk["keywords"],
                        "chunk_index": chunk_index,
                    },
                )
            )

    client = get_qdrant_client()
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


if __name__ == "__main__":
    count = ingest_danske_spil_rules()
    print(f"Ingested {count} chunks into Qdrant collection '{COLLECTION_NAME}'.")
