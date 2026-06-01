from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.agents.support_answer_agent import generate_support_answer
from app.db.qdrant_client import check_qdrant_connection
from app.rag.retriever import search_guidelines

app = FastAPI(title="Customer Support Agents")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


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
        retrieved_guidelines = search_guidelines(user_message)
        agent_response = generate_support_answer(user_message, retrieved_guidelines)
    except RuntimeError as e:
        return {
            "error": str(e),
            "user_message": user_message,
        }

    return {
        "user_message": user_message,
        "answer": agent_response["answer"],
        "agent": agent_response["agent"],
        "model": agent_response["model"],
        "sources": [
            {
                "section": guideline.get("section"),
                "source_url": guideline.get("source_url"),
                "chunk_index": guideline.get("chunk_index"),
                "score": guideline.get("score"),
            }
            for guideline in retrieved_guidelines
        ],
        "retrieved_guidelines": retrieved_guidelines,
    }
