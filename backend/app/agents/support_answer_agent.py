import os
from pathlib import Path
import time

import httpx
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[2] / ".env")

OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_CHAT_MODEL = "gpt-4.1-mini"
DEFAULT_CHAT_RETRY_SECONDS = 8.0
MAX_CHAT_RETRIES = 4


SUPPORT_ANSWER_INSTRUCTIONS = """
Du er en kundesupport-agent for et spilprodukt.

Svar på dansk, kort og konkret. Brug kun den kontekst, der er givet under
"Relevante regler". Hvis konteksten ikke er nok til at svare sikkert, så sig
det tydeligt og foreslå at eskalere til en menneskelig supportmedarbejder.

Du må ikke opfinde regler, beløb, frister eller procestrin. Når svaret bygger
på en regel, skal du nævne produktsektionen og URL'en som kilde. Skriv fx:
"Kilde: Oddset - https://..."

Hvis kunden spørger om kontospecifikke forhold, betalinger, identitet,
udelukkelse, ansvarligt spil eller juridisk tvivl, skal du give generel
vejledning ud fra reglerne og anbefale menneskelig hjælp.
""".strip()


def generate_support_answer(user_message: str, retrieved_guidelines: list[dict]) -> dict:
    if not retrieved_guidelines:
        return {
            "answer": (
                "Jeg kunne ikke finde relevante regler i vidensbasen. "
                "Sagen bør derfor eskaleres til en menneskelig supportmedarbejder."
            ),
            "model": None,
            "agent": "support_answer_agent",
        }

    context = format_retrieved_context(retrieved_guidelines)
    response = _create_chat_completion(
        messages=[
            {"role": "developer", "content": SUPPORT_ANSWER_INSTRUCTIONS},
            {
                "role": "user",
                "content": (
                    f"Kundens spørgsmål:\n{user_message}\n\n"
                    f"Relevante regler:\n{context}\n\n"
                    "Skriv et kundesupport-svar baseret på reglerne."
                ),
            },
        ],
    )

    return {
        "answer": response["choices"][0]["message"]["content"].strip(),
        "model": response.get("model"),
        "agent": "support_answer_agent",
    }


def format_retrieved_context(retrieved_guidelines: list[dict]) -> str:
    context_blocks = []

    for index, guideline in enumerate(retrieved_guidelines, start=1):
        section = guideline.get("section") or "Ukendt sektion"
        chunk_index = guideline.get("chunk_index")
        source_url = guideline.get("source_url") or "Ukendt kilde"
        text = guideline.get("text") or ""

        context_blocks.append(
            "\n".join(
                [
                    f"[{index}] Section: {section}",
                    f"Internal chunk: {chunk_index}",
                    f"Source URL: {source_url}",
                    f"Text: {text}",
                ]
            )
        )

    return "\n\n".join(context_blocks)


def _create_chat_completion(messages: list[dict]) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to generate support answers.")

    model = os.getenv("CHAT_MODEL", DEFAULT_CHAT_MODEL)

    for attempt in range(MAX_CHAT_RETRIES + 1):
        response = httpx.post(
            OPENAI_CHAT_COMPLETIONS_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.2,
            },
            timeout=60.0,
        )

        if response.status_code != 429:
            break

        if attempt == MAX_CHAT_RETRIES:
            break

        time.sleep(_retry_delay(response, attempt))

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"OpenAI chat request failed: {response.text}") from exc

    return response.json()


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass

    default_delay = float(os.getenv("CHAT_RETRY_SECONDS", str(DEFAULT_CHAT_RETRY_SECONDS)))
    return default_delay * (attempt + 1)
