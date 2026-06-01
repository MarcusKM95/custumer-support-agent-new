import json
import os
from pathlib import Path
import time

import httpx
from dotenv import load_dotenv

from app.agents.support_answer_agent import format_retrieved_context


load_dotenv(Path(__file__).resolve().parents[2] / ".env")

OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_RERANKER_MODEL = "gpt-4.1-mini"
DEFAULT_RERANKER_RETRY_SECONDS = 8.0
MAX_RERANKER_RETRIES = 4

RERANKER_INSTRUCTIONS = """
Du er reranker-agent for en dansk kundesupport-RAG.

Vælg de regeluddrag, der mest direkte kan besvare kundens spørgsmål.
Prioriter konkrete regler over generelle eller beslægtede regler. Fravælg
uddrag om andre sportsgrene eller produkter, hvis spørgsmålet handler om en
bestemt sport, spilkategori eller produkt.

Returner kun valid JSON:
{
  "selected_indexes": [1, 2, 3],
  "confidence": 0.0,
  "reason": "kort dansk begrundelse"
}
""".strip()


def rerank_guidelines(
    user_message: str,
    candidates: list[dict],
    limit: int = 4,
    conversation_history: str | None = None,
) -> dict:
    if not candidates:
        return {
            "guidelines": [],
            "reranker": {
                "selected_indexes": [],
                "confidence": 0.0,
                "reason": "Ingen kandidater fundet.",
            },
        }

    response = _create_reranker_completion(
        messages=[
            {"role": "developer", "content": RERANKER_INSTRUCTIONS},
            {
                "role": "user",
                "content": (
                    f"Tidligere samtale:\n{conversation_history or 'Ingen tidligere beskeder.'}\n\n"
                    f"Kundens spørgsmål:\n{user_message}\n\n"
                    f"Kandidatregler:\n{format_retrieved_context(candidates)}"
                ),
            },
        ]
    )
    content = response["choices"][0]["message"]["content"]
    reranker = normalize_reranker_response(content, len(candidates), limit)
    selected_guidelines = [
        candidates[index - 1]
        for index in reranker["selected_indexes"]
        if 1 <= index <= len(candidates)
    ]

    if not selected_guidelines:
        selected_guidelines = candidates[:limit]

    return {
        "guidelines": selected_guidelines,
        "reranker": reranker,
    }


def normalize_reranker_response(content: str, candidate_count: int, limit: int) -> dict:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return {
            "selected_indexes": list(range(1, min(candidate_count, limit) + 1)),
            "confidence": 0.0,
            "reason": "Rerankeren returnerede ikke gyldig JSON.",
        }

    selected_indexes = []
    for value in data.get("selected_indexes", []):
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= index <= candidate_count and index not in selected_indexes:
            selected_indexes.append(index)
        if len(selected_indexes) == limit:
            break

    confidence = data.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "selected_indexes": selected_indexes,
        "confidence": max(0.0, min(confidence, 1.0)),
        "reason": str(data.get("reason") or "Ingen begrundelse angivet."),
    }


def _create_reranker_completion(messages: list[dict]) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to rerank retrieved rules.")

    model = os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL)

    for attempt in range(MAX_RERANKER_RETRIES + 1):
        response = httpx.post(
            OPENAI_CHAT_COMPLETIONS_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
            timeout=60.0,
        )

        if response.status_code != 429:
            break

        if attempt == MAX_RERANKER_RETRIES:
            break

        time.sleep(_retry_delay(response, attempt))

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"OpenAI reranker request failed: {response.text}") from exc

    return response.json()


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass

    default_delay = float(
        os.getenv("RERANKER_RETRY_SECONDS", str(DEFAULT_RERANKER_RETRY_SECONDS))
    )
    return default_delay * (attempt + 1)
