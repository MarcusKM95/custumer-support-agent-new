from uuid import UUID, uuid4

from psycopg.types.json import Jsonb

from app.db.postgres import get_connection


def create_conversation() -> str:
    conversation_id = uuid4()

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO conversations (id) VALUES (%s)",
                (conversation_id,),
            )
        connection.commit()

    return str(conversation_id)


def ensure_conversation(conversation_id: str | None) -> str:
    if not conversation_id:
        return create_conversation()

    parsed_id = UUID(conversation_id)

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO conversations (id)
                VALUES (%s)
                ON CONFLICT (id) DO UPDATE SET updated_at = now()
                """,
                (parsed_id,),
            )
        connection.commit()

    return str(parsed_id)


def add_message(
    conversation_id: str,
    role: str,
    content: str,
    metadata: dict | None = None,
) -> str:
    message_id = uuid4()
    parsed_conversation_id = UUID(conversation_id)

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, metadata)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    message_id,
                    parsed_conversation_id,
                    role,
                    content,
                    Jsonb(metadata or {}),
                ),
            )
            cursor.execute(
                "UPDATE conversations SET updated_at = now() WHERE id = %s",
                (parsed_conversation_id,),
            )
        connection.commit()

    return str(message_id)


def get_recent_messages(conversation_id: str, limit: int = 8) -> list[dict]:
    parsed_conversation_id = UUID(conversation_id)

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, role, content, metadata, created_at
                FROM messages
                WHERE conversation_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (parsed_conversation_id, limit),
            )
            rows = cursor.fetchall()

    return [
        {
            "id": str(row["id"]),
            "role": row["role"],
            "content": row["content"],
            "metadata": row["metadata"],
            "created_at": row["created_at"].isoformat(),
        }
        for row in reversed(rows)
    ]


def format_history(messages: list[dict]) -> str:
    if not messages:
        return "Ingen tidligere beskeder."

    lines = []
    for message in messages:
        role = "Kunde" if message["role"] == "user" else "Assistant"
        lines.append(f"{role}: {message['content']}")

    return "\n".join(lines)
