from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.agents.escalation_agent import build_escalation_response
from app.agents.reranker_agent import rerank_guidelines
from app.agents.router_agent import route_message
from app.agents.support_answer_agent import add_source_citation, generate_support_answer
from app.agents.verification_agent import verify_support_answer
from app.db.postgres import initialize_database
from app.db.qdrant_client import check_qdrant_connection
from app.memory.conversation_memory import (
    add_message,
    ensure_conversation,
    format_history,
    get_recent_messages,
)
from app.rag.retriever import search_guidelines

app = FastAPI(title="Customer Support Agents")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: str | None = None


@app.on_event("startup")
def startup() -> None:
    initialize_database()


@app.get("/")
def root():
    return {"message": "Customer Support Agents backend is running"}


@app.get("/health/qdrant")
def qdrant_health():
    return check_qdrant_connection()


@app.post("/chat")
def chat(request: ChatRequest):
    user_message = request.message.strip()
    try:
        conversation_id = ensure_conversation(request.conversation_id)
        previous_messages = get_recent_messages(conversation_id)
        conversation_history = format_history(previous_messages)
        add_message(conversation_id, "user", user_message)

        route = route_message(user_message, conversation_history)
        if route["intent"] == "rules_question":
            candidate_guidelines = search_guidelines(user_message, limit=12)
            reranked = rerank_guidelines(
                user_message,
                candidate_guidelines,
                conversation_history=conversation_history,
            )
            retrieved_guidelines = reranked["guidelines"]
            agent_response = generate_support_answer(
                user_message,
                retrieved_guidelines,
                conversation_history=conversation_history,
            )
            verification = verify_support_answer(
                user_message,
                agent_response["answer"],
                retrieved_guidelines,
                conversation_history=conversation_history,
            )
            agent_response["answer"] = verification["corrected_answer"]
            agent_response["answer"] = add_source_citation(
                agent_response["answer"],
                retrieved_guidelines,
            )
            agent_response["reranker"] = reranked["reranker"]
            agent_response["verification"] = verification
        else:
            agent_response = build_escalation_response(user_message, route)

        add_message(
            conversation_id,
            "assistant",
            agent_response["answer"],
            metadata={
                "router": route,
                "agent": agent_response["agent"],
                "model": agent_response["model"],
                "sources": agent_response.get("sources")
                or build_sources(agent_response.get("retrieved_guidelines", [])),
                "escalation": agent_response.get("escalation", {"required": False}),
                "reranker": agent_response.get("reranker"),
                "verification": agent_response.get("verification"),
            },
        )
    except RuntimeError as e:
        return {
            "error": str(e),
            "user_message": user_message,
        }

    return {
        "conversation_id": conversation_id,
        "user_message": user_message,
        "router": route,
        "answer": agent_response["answer"],
        "agent": agent_response["agent"],
        "model": agent_response["model"],
        "sources": agent_response.get("sources")
        or build_sources(agent_response.get("retrieved_guidelines", [])),
        "retrieved_guidelines": agent_response.get("retrieved_guidelines", []),
        "reranker": agent_response.get("reranker"),
        "verification": agent_response.get("verification"),
        "escalation": agent_response.get("escalation", {"required": False}),
    }


def build_sources(retrieved_guidelines: list[dict]) -> list[dict]:
    return [
        {
            "section": guideline.get("section"),
            "source_url": guideline.get("source_url"),
            "rule_number": guideline.get("rule_number"),
            "chunk_index": guideline.get("chunk_index"),
            "score": guideline.get("score"),
        }
        for guideline in retrieved_guidelines
    ]
