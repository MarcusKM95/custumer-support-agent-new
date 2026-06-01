from fastapi import FastAPI

from app.db.qdrant_client import check_qdrant_connection
from app.rag.retriever import search_guidelines

app = FastAPI(title="Customer Support Agents")


@app.get("/")
def root():
    return {"message": "Customer Support Agents backend is running"}


@app.get("/health/qdrant")
def qdrant_health():
    return check_qdrant_connection()


@app.post("/chat")
def chat(request: dict):
    user_message = request.get("message", "")

    if not user_message:
        return {
            "error": "Missing message field",
            "example": {"message": "Can I get a refund after 20 days?"},
        }

    try:
        retrieved_guidelines = search_guidelines(user_message)
    except RuntimeError as e:
        return {
            "error": str(e),
            "user_message": user_message,
        }

    return {
        "user_message": user_message,
        "retrieved_guidelines": retrieved_guidelines,
    }