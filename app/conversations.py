from typing import cast

from anthropic.types import MessageParam
from sqlmodel import Session, col, select

from app.ai import GeneratedReply
from app.db.models import Conversation, ConversationTurn

# Only this many of a conversation's most recent turns are sent to Claude, which
# stops the context and cost from growing without limit. Turns are dropped whole,
# so a tool_use is never separated from its tool_result. Older turns stay in the
# database and are still returned by the history endpoint.
MAX_HISTORY_TURNS = 10


def get_owned_conversation(
    session: Session, conversation_id: str, customer_id: str
) -> Conversation | None:
    return session.exec(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.customer_id == customer_id,
        )
    ).first()


def list_conversations(session: Session, customer_id: str) -> list[Conversation]:
    conversations = session.exec(
        select(Conversation)
        .where(Conversation.customer_id == customer_id)
        .order_by(col(Conversation.created_at).desc())
    ).all()
    return list(conversations)


def list_turns(session: Session, conversation_id: str) -> list[ConversationTurn]:
    turns = session.exec(
        select(ConversationTurn)
        .where(ConversationTurn.conversation_id == conversation_id)
        .order_by(col(ConversationTurn.id))
    ).all()
    return list(turns)


def load_llm_history(session: Session, conversation_id: str) -> list[MessageParam]:
    newest_turns = session.exec(
        select(ConversationTurn)
        .where(ConversationTurn.conversation_id == conversation_id)
        .order_by(col(ConversationTurn.id).desc())
        .limit(MAX_HISTORY_TURNS)
    ).all()
    return [
        cast(MessageParam, message)
        for turn in reversed(newest_turns)
        for message in turn.llm_messages
    ]


def save_turn(
    session: Session,
    conversation_id: str | None,
    customer_id: str,
    question: str,
    generated: GeneratedReply,
) -> str:
    if conversation_id is None:
        conversation = Conversation(customer_id=customer_id)
        session.add(conversation)
        session.flush()
        conversation_id = conversation.id
    session.add(
        ConversationTurn(
            conversation_id=conversation_id,
            question=question,
            reply=generated.reply.reply,
            citations=generated.reply.citations,
            analysis=generated.reply.analysis.model_dump(mode="json"),
            llm_messages=generated.llm_messages,
        )
    )
    session.commit()
    return conversation_id
