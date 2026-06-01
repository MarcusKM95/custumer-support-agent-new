ESCALATION_CONFIG = {
    "account_issue": {
        "queue": "account_support",
        "priority": "medium",
        "answer": (
            "Det lyder som et kontospecifikt spørgsmål. Jeg kan ikke se eller ændre "
            "din konto her, så jeg sender sagen videre til en supportmedarbejder."
        ),
    },
    "payment_issue": {
        "queue": "payments",
        "priority": "high",
        "answer": (
            "Det lyder som et betalings- eller udbetalingsspørgsmål. Af sikkerhedsgrunde "
            "sender jeg sagen videre til en supportmedarbejder."
        ),
    },
    "complaint": {
        "queue": "complaints",
        "priority": "high",
        "answer": (
            "Jeg forstår, at du vil klage eller er utilfreds. Jeg sender sagen videre, "
            "så en supportmedarbejder kan gennemgå den."
        ),
    },
    "responsible_gambling": {
        "queue": "responsible_gambling",
        "priority": "urgent",
        "answer": (
            "Det lyder som noget, der handler om ansvarligt spil eller spilleadfærd. "
            "Jeg sender sagen videre, så du kan få hjælp af en supportmedarbejder."
        ),
    },
    "technical_issue": {
        "queue": "technical_support",
        "priority": "medium",
        "answer": (
            "Det lyder som et teknisk problem. Jeg sender sagen videre til teknisk "
            "support, som kan hjælpe med fejlfinding."
        ),
    },
    "unknown": {
        "queue": "general_support",
        "priority": "low",
        "answer": (
            "Jeg er ikke helt sikker på, hvad du har brug for hjælp til. Jeg sender "
            "sagen videre, så en supportmedarbejder kan hjælpe dig korrekt."
        ),
    },
}


def build_escalation_response(user_message: str, route: dict) -> dict:
    intent = route.get("intent", "unknown")
    config = ESCALATION_CONFIG.get(intent, ESCALATION_CONFIG["unknown"])

    escalation = {
        "required": True,
        "intent": intent,
        "queue": config["queue"],
        "priority": config["priority"],
        "reason": route.get("reason") or "Routeren markerede sagen til eskalering.",
        "customer_message": user_message,
        "router_confidence": route.get("confidence", 0.0),
    }

    return {
        "answer": config["answer"],
        "agent": "escalation_agent",
        "model": None,
        "sources": [],
        "retrieved_guidelines": [],
        "escalation": escalation,
    }
