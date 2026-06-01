import json
import os
from pathlib import Path
import time

import httpx
from dotenv import load_dotenv

from app.agents.support_answer_agent import format_retrieved_context


load_dotenv(Path(__file__).resolve().parents[2] / ".env")

OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_VERIFIER_MODEL = "gpt-4.1-mini"
DEFAULT_VERIFIER_RETRY_SECONDS = 8.0
MAX_VERIFIER_RETRIES = 4

VERIFIER_INSTRUCTIONS = """
Du er verifikations-agent for en dansk kundesupport-chatbot.

Kontrollér om svaret kun indeholder påstande, der er understøttet af de
relevante regler. Hvis svaret er for sikkert, upræcist eller bygger på noget,
der ikke står i reglerne, skal du rette det.

Returner kun valid JSON:
{
  "supported": true,
  "confidence": 0.0,
  "reason": "kort dansk begrundelse",
  "corrected_answer": "svar der er sikkert understøttet af reglerne"
}
""".strip()


def verify_support_answer(
    user_message: str,
    answer: str,
    retrieved_guidelines: list[dict],
    conversation_history: str | None = None,
) -> dict:
    if not retrieved_guidelines:
        return {
            "supported": False,
            "confidence": 0.0,
            "reason": "Der var ingen regeluddrag at verificere imod.",
            "corrected_answer": answer,
        }

    response = _create_verifier_completion(
        messages=[
            {"role": "developer", "content": VERIFIER_INSTRUCTIONS},
            {
                "role": "user",
                "content": (
                    f"Tidligere samtale:\n{conversation_history or 'Ingen tidligere beskeder.'}\n\n"
                    f"Kundens spørgsmål:\n{user_message}\n\n"
                    f"Svar:\n{answer}\n\n"
                    f"Relevante regler:\n{format_retrieved_context(retrieved_guidelines)}"
                ),
            },
        ]
    )

    return normalize_verifier_response(
        response["choices"][0]["message"]["content"],
        fallback_answer=answer,
    )


def normalize_verifier_response(content: str, fallback_answer: str) -> dict:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return {
            "supported": False,
            "confidence": 0.0,
            "reason": "Verifikatoren returnerede ikke gyldig JSON.",
            "corrected_answer": fallback_answer,
        }

    confidence = data.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    corrected_answer = data.get("corrected_answer") or fallback_answer

    return {
        "supported": bool(data.get("supported")),
        "confidence": max(0.0, min(confidence, 1.0)),
        "reason": str(data.get("reason") or "Ingen begrundelse angivet."),
        "corrected_answer": str(corrected_answer),
    }


def _create_verifier_completion(messages: list[dict]) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to verify support answers.")

    model = os.getenv("VERIFIER_MODEL", DEFAULT_VERIFIER_MODEL)

    for attempt in range(MAX_VERIFIER_RETRIES + 1):
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

        if attempt == MAX_VERIFIER_RETRIES:
            break

        time.sleep(_retry_delay(response, attempt))

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"OpenAI verifier request failed: {response.text}") from exc

    return response.json()


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass

    default_delay = float(
        os.getenv("VERIFIER_RETRY_SECONDS", str(DEFAULT_VERIFIER_RETRY_SECONDS))
    )
    return default_delay * (attempt + 1)
