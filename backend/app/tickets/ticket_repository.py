from uuid import UUID

from app.db.postgres import get_connection


def create_ticket(conversation_id: str, escalation: dict) -> dict:
    parsed_conversation_id = UUID(conversation_id)

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO tickets (
                    conversation_id,
                    intent,
                    queue,
                    priority,
                    reason,
                    customer_message,
                    router_confidence
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING
                    id,
                    conversation_id,
                    intent,
                    queue,
                    priority,
                    status,
                    reason,
                    customer_message,
                    router_confidence,
                    created_at,
                    updated_at
                """,
                (
                    parsed_conversation_id,
                    escalation["intent"],
                    escalation["queue"],
                    escalation["priority"],
                    escalation["reason"],
                    escalation["customer_message"],
                    escalation["router_confidence"],
                ),
            )
            row = cursor.fetchone()
        connection.commit()

    return serialize_ticket(row)


def serialize_ticket(row: dict) -> dict:
    return {
        "id": row["id"],
        "ticket_number": f"TKT-{row['id']:06d}",
        "conversation_id": str(row["conversation_id"]),
        "intent": row["intent"],
        "queue": row["queue"],
        "priority": row["priority"],
        "status": row["status"],
        "reason": row["reason"],
        "customer_message": row["customer_message"],
        "router_confidence": row["router_confidence"],
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
    }
