def chunk_text(text: str) -> list[str]:
    """
    Very simple test chunker.

    For now, each paragraph separated by a blank line becomes one chunk.
    This is enough for the first local RAG test.
    Later this can be replaced by a better heading-aware chunker.
    """
    return [section.strip() for section in text.split("\n\n") if section.strip()]
