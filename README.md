# Customer Support Agent

A customer support agent application with backend and frontend components.

## Project Structure

- `backend/` - Backend API service
- `frontend/` - Frontend application
- `docker-compose.yml` - Docker Compose configuration

## Getting Started

Start Qdrant:

```bash
docker compose up -d qdrant
```

Install backend dependencies:

```bash
backend/.venv/bin/pip install -r backend/requirements.txt
```

Configure embeddings:

```bash
export OPENAI_API_KEY="your-api-key"
export EMBEDDING_MODEL="text-embedding-3-small"
export EMBEDDING_DIMENSIONS="384"
export EMBEDDING_BATCH_SIZE="16"
export EMBEDDING_RETRY_SECONDS="8"
export CHAT_MODEL="gpt-4.1-mini"
export CHAT_RETRY_SECONDS="8"
```

Ingest the Danske Spil rules into Qdrant:

```bash
cd backend
python3 -m app.rag.ingest_danske_spil
```

Run the backend:

```bash
uvicorn app.main:app --reload
```
