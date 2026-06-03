CLARIFICATION_CONFIG = {
    "greeting": {
        "answer": "Hej! Hvad kan jeg hjælpe dig med?",
        "reason": "Kunden hilser uden at stille et konkret spørgsmål.",
    },
    "needs_clarification": {
        "answer": (
            "Kan du fortælle lidt mere om, hvad du har brug for hjælp til? "
            "Skriv gerne hvilket spil eller hvilken situation det handler om."
        ),
        "reason": "Beskeden er for uklar til at vælge den rigtige hjælp.",
    },
    "technical_issue": {
        "answer": (
            "Kan du beskrive, hvad der ikke virker? Skriv gerne om det sker i appen "
            "eller på hjemmesiden, og om du får en fejlbesked."
        ),
        "reason": "Tekniske problemer bør afklares, før de eskaleres.",
    },
}


def build_clarification_response(route: dict) -> dict:
    intent = route.get("intent", "needs_clarification")
    config = CLARIFICATION_CONFIG.get(intent, CLARIFICATION_CONFIG["needs_clarification"])

    return {
        "answer": config["answer"],
        "agent": "clarification_agent",
        "model": None,
        "sources": [],
        "retrieved_guidelines": [],
        "clarification": {
            "required": True,
            "intent": intent,
            "reason": route.get("reason") or config["reason"],
            "router_confidence": route.get("confidence", 0.0),
        },
        "escalation": {"required": False},
    }
