from fastapi import FastAPI

from app.db.qdrant_client import check_qdrant_connection

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

    return {
        "user_message": user_message,
        "answer": "This is the first version of the customer support AI.",
    }