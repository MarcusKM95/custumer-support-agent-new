from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agents.clarification_agent import build_clarification_response
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
from app.rag.knowledge_base_admin import (
    ReindexInProgressError,
    reindex_knowledge_base,
)
from app.rag.knowledge_base_status import get_knowledge_base_status
from app.rag.retriever import search_guidelines
from app.tickets.ticket_repository import create_ticket

app = FastAPI(title="Customer Support Agents")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "null",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/knowledge-base/status")
def knowledge_base_status():
    try:
        return get_knowledge_base_status()
    except Exception as error:
        return {
            "status": "error",
            "error": str(error),
        }


@app.post("/knowledge-base/reindex")
def knowledge_base_reindex():
    try:
        return reindex_knowledge_base()
    except ReindexInProgressError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/chat")
def chat(request: ChatRequest):
    user_message = request.message.strip()
    trace = []
    request_started_at = perf_counter()
    try:
        step_started_at = perf_counter()
        conversation_id = ensure_conversation(request.conversation_id)
        previous_messages = get_recent_messages(conversation_id)
        conversation_history = format_history(previous_messages)
        add_message(conversation_id, "user", user_message)
        trace.append(
            build_trace_step(
                "memory",
                "Conversation Memory",
                step_started_at,
                {
                    "conversation_id": conversation_id,
                    "history_messages": len(previous_messages),
                },
            )
        )

        step_started_at = perf_counter()
        route = route_message(user_message, conversation_history)
        trace.append(
            build_trace_step(
                "router",
                "Router Agent",
                step_started_at,
                {
                    "intent": route["intent"],
                    "confidence": route["confidence"],
                    "reason": route["reason"],
                },
            )
        )

        if route["intent"] == "rules_question":
            step_started_at = perf_counter()
            candidate_guidelines = search_guidelines(user_message, limit=12)
            trace.append(
                build_trace_step(
                    "retrieval",
                    "Qdrant Retrieval",
                    step_started_at,
                    {
                        "candidate_count": len(candidate_guidelines),
                        "sections": sorted(
                            {
                                guideline.get("section")
                                for guideline in candidate_guidelines
                                if guideline.get("section")
                            }
                        ),
                    },
                )
            )

            step_started_at = perf_counter()
            reranked = rerank_guidelines(
                user_message,
                candidate_guidelines,
                conversation_history=conversation_history,
            )
            retrieved_guidelines = reranked["guidelines"]
            trace.append(
                build_trace_step(
                    "reranker",
                    "Reranker Agent",
                    step_started_at,
                    {
                        "selected_count": len(retrieved_guidelines),
                        "confidence": reranked["reranker"]["confidence"],
                        "reason": reranked["reranker"]["reason"],
                    },
                )
            )

            step_started_at = perf_counter()
            agent_response = generate_support_answer(
                user_message,
                retrieved_guidelines,
                conversation_history=conversation_history,
            )
            trace.append(
                build_trace_step(
                    "answer",
                    "Support Answer Agent",
                    step_started_at,
                    {
                        "model": agent_response["model"],
                        "source_count": len(retrieved_guidelines),
                    },
                )
            )

            step_started_at = perf_counter()
            verification = verify_support_answer(
                user_message,
                agent_response["answer"],
                retrieved_guidelines,
                conversation_history=conversation_history,
            )
            trace.append(
                build_trace_step(
                    "verification",
                    "Verification Agent",
                    step_started_at,
                    {
                        "supported": verification["supported"],
                        "confidence": verification["confidence"],
                        "reason": verification["reason"],
                    },
                )
            )
            agent_response["answer"] = verification["corrected_answer"]
            agent_response["answer"] = add_source_citation(
                agent_response["answer"],
                retrieved_guidelines,
            )
            agent_response["reranker"] = reranked["reranker"]
            agent_response["verification"] = verification
        elif route["intent"] in {"greeting", "needs_clarification", "technical_issue"}:
            step_started_at = perf_counter()
            agent_response = build_clarification_response(route)
            trace.append(
                build_trace_step(
                    "clarification",
                    "Clarification Agent",
                    step_started_at,
                    {
                        "intent": route["intent"],
                        "reason": agent_response["clarification"]["reason"],
                    },
                )
            )
        else:
            step_started_at = perf_counter()
            agent_response = build_escalation_response(user_message, route)
            trace.append(
                build_trace_step(
                    "escalation",
                    "Escalation Agent",
                    step_started_at,
                    {
                        "queue": agent_response["escalation"]["queue"],
                        "priority": agent_response["escalation"]["priority"],
                        "reason": agent_response["escalation"]["reason"],
                    },
                )
            )
            step_started_at = perf_counter()
            ticket = create_ticket(conversation_id, agent_response["escalation"])
            agent_response["ticket"] = ticket
            trace.append(
                build_trace_step(
                    "ticket",
                    "Create Support Ticket",
                    step_started_at,
                    {
                        "ticket_number": ticket["ticket_number"],
                        "queue": ticket["queue"],
                        "priority": ticket["priority"],
                        "status": ticket["status"],
                    },
                )
            )

        step_started_at = perf_counter()
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
                "ticket": agent_response.get("ticket"),
                "reranker": agent_response.get("reranker"),
                "verification": agent_response.get("verification"),
                "trace": trace,
            },
        )
        trace.append(
            build_trace_step(
                "persistence",
                "Persist Response",
                step_started_at,
                {"stored": True},
            )
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
        "clarification": agent_response.get("clarification", {"required": False}),
        "escalation": agent_response.get("escalation", {"required": False}),
        "ticket": agent_response.get("ticket"),
        "trace": trace,
        "duration_ms": round((perf_counter() - request_started_at) * 1000, 1),
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


def build_trace_step(
    step_id: str,
    label: str,
    started_at: float,
    details: dict,
) -> dict:
    return {
        "id": step_id,
        "label": label,
        "status": "completed",
        "duration_ms": round((perf_counter() - started_at) * 1000, 1),
        "details": details,
    }
