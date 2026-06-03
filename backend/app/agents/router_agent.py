import json
import os
from pathlib import Path
import time

import httpx
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[2] / ".env")

OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_ROUTER_MODEL = "gpt-4.1-mini"
DEFAULT_ROUTER_RETRY_SECONDS = 8.0
MAX_ROUTER_RETRIES = 4

SUPPORTED_INTENTS = {
    "rules_question",
    "greeting",
    "needs_clarification",
    "account_issue",
    "payment_issue",
    "complaint",
    "responsible_gambling",
    "technical_issue",
    "unknown",
}

ROUTER_INSTRUCTIONS = """
Du er routing-agent for en dansk kundesupport-chatbot.

Klassificer kundens besked i præcis én intent:
- rules_question: spørgsmål om spilleregler, produkter, vinderregler, odds, bingo, casino, poker, tips, måljagt, dantoto eller livecasino.
- greeting: simple hilsner som "hej", "godmorgen", "halløjsa" uden konkret problem.
- needs_clarification: beskeden er for vag til at afgøre, hvad kunden har brug for, fx "det virker ikke", "hjælp", "jeg har et problem".
- account_issue: login, konto, verificering, personlige oplysninger, spilgrænser eller kundens konkrete konto.
- payment_issue: indbetaling, udbetaling, manglende penge, refundering eller transaktioner.
- complaint: utilfredshed, klage, vred kunde, anklager om fejl/snyd eller ønske om at klage.
- responsible_gambling: ludomani, spilleproblemer, ROFUS, selvudelukkelse, stop med at spille eller bekymring om spiladfærd.
- technical_issue: konkret teknisk problem med app, hjemmeside, fejlmeddelelse, nedbrud, browser eller teknisk adgang.
- unknown: beskeden er uklar eller passer ikke på ovenstående.

Hvis beskeden både indeholder utilfredshed/ord som "snyd" og et konkret
spørgsmål om et spil, væddemål, resultat eller spilleregel, skal den
klassificeres som rules_question. Det samme gælder spørgsmål om chatregler,
karantæne, ban eller udelukkelse i et konkret spil, når årsagen kan forklares
med generelle regler. Brug kun complaint eller account_issue, når beskeden
primært handler om en kontoændring uden et generelt regelspørgsmål.

Simple hilsner må aldrig klassificeres som account_issue, complaint eller
unknown. Vage beskeder må ikke eskaleres direkte; brug needs_clarification.

Returner kun valid JSON med nøglerne:
{
  "intent": "...",
  "confidence": 0.0,
  "reason": "kort dansk begrundelse"
}
""".strip()


def route_message(user_message: str, conversation_history: str | None = None) -> dict:
    response = _create_router_completion(
        messages=[
            {"role": "developer", "content": ROUTER_INSTRUCTIONS},
            {
                "role": "user",
                "content": (
                    f"Tidligere samtale:\n{conversation_history or 'Ingen tidligere beskeder.'}\n\n"
                    f"Ny kundebesked:\n{user_message}"
                ),
            },
        ],
    )
    content = response["choices"][0]["message"]["content"]
    return normalize_route(content)


def normalize_route(content: str) -> dict:
    try:
        route = json.loads(content)
    except json.JSONDecodeError:
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "reason": "Routeren returnerede ikke gyldig JSON.",
        }

    intent = route.get("intent")
    if intent not in SUPPORTED_INTENTS:
        intent = "unknown"

    confidence = route.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(confidence, 1.0))
    reason = route.get("reason") or "Ingen begrundelse angivet."

    return {
        "intent": intent,
        "confidence": confidence,
        "reason": str(reason),
    }


def _create_router_completion(messages: list[dict]) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to route support messages.")

    model = os.getenv("ROUTER_MODEL", DEFAULT_ROUTER_MODEL)

    for attempt in range(MAX_ROUTER_RETRIES + 1):
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

        if attempt == MAX_ROUTER_RETRIES:
            break

        time.sleep(_retry_delay(response, attempt))

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"OpenAI router request failed: {response.text}") from exc

    return response.json()


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass

    default_delay = float(os.getenv("ROUTER_RETRY_SECONDS", str(DEFAULT_ROUTER_RETRY_SECONDS)))
    return default_delay * (attempt + 1)
