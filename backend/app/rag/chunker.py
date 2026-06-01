import re


NUMBERED_RULE_PATTERN = re.compile(r"(?m)^(?=\d+\.\s*)")


def chunk_text(text: str, max_chars: int = 1200) -> list[str]:
    """
    Chunk policy/rules text into retrieval-friendly sections.

    Numbered legal/support rules are kept together when possible. Oversized
    chunks are split on paragraph boundaries so they still fit prompt context.
    """
    normalized_text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not normalized_text:
        return []

    raw_sections = [
        section.strip()
        for section in NUMBERED_RULE_PATTERN.split(normalized_text)
        if section.strip()
    ]

    chunks: list[str] = []
    current = ""

    for section in raw_sections:
        if len(section) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_section(section, max_chars))
            continue

        candidate = f"{current}\n\n{section}".strip() if current else section
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = section

    if current:
        chunks.append(current)

    return chunks


def _split_long_section(section: str, max_chars: int) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in section.split("\n\n") if paragraph.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_by_sentence(paragraph, max_chars))
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    return chunks


def _split_by_sentence(text: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks
