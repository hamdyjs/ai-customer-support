from datetime import datetime

from anthropic.types import MessageParam
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict
from sqlmodel import Session

from app.ai import (
    AnalysisError,
    LLMServiceDep,
    QuestionAnalysis,
    ReplyError,
    SupportReply,
)
from app.auth import CurrentCustomerDep
from app.conversations import (
    get_owned_conversation,
    list_conversations,
    list_turns,
    load_llm_history,
    save_turn,
)
from app.db.database import SessionDep
from app.db.models import Conversation

router = APIRouter(prefix="/support", tags=["support"])


class AIRequest(BaseModel):
    question: str


class MessageRequest(AIRequest):
    conversation_id: str | None = None


class MessageResponse(SupportReply):
    conversation_id: str


class ConversationSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime


class ConversationTurnResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    question: str
    reply: str
    analysis: QuestionAnalysis
    created_at: datetime


class ConversationHistory(ConversationSummary):
    turns: list[ConversationTurnResponse]


def get_conversation_or_404(
    session: Session, conversation_id: str, customer_id: str
) -> Conversation:
    conversation = get_owned_conversation(session, conversation_id, customer_id)
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return conversation


@router.post("/analyze")
async def analyze(request: AIRequest, llm: LLMServiceDep) -> QuestionAnalysis:
    try:
        return await llm.analyze_question(request.question)
    except AnalysisError as error:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error


@router.post("/messages")
async def messages(
    request: MessageRequest,
    customer: CurrentCustomerDep,
    session: SessionDep,
    llm: LLMServiceDep,
) -> MessageResponse:
    """Send a message, starting a new conversation unless `conversation_id` is set.

    Claude sees only the conversation's most recent turns, and a failed turn isn't
    saved. The full history stays available from the conversation endpoints.
    """
    conversation = None
    history: list[MessageParam] = []
    if request.conversation_id is not None:
        conversation = get_conversation_or_404(
            session, request.conversation_id, customer.id
        )
        history = load_llm_history(session, conversation.id)

    try:
        generated = await llm.generate_reply(
            request.question, history, session, customer.id
        )
    except ReplyError as error:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error

    conversation_id = save_turn(
        session, conversation, customer.id, request.question, generated
    )
    return MessageResponse(
        conversation_id=conversation_id,
        reply=generated.reply.reply,
        analysis=generated.reply.analysis,
    )


@router.get("/conversations")
def get_conversations(
    customer: CurrentCustomerDep, session: SessionDep
) -> list[ConversationSummary]:
    return [
        ConversationSummary.model_validate(conversation)
        for conversation in list_conversations(session, customer.id)
    ]


@router.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str, customer: CurrentCustomerDep, session: SessionDep
) -> ConversationHistory:
    conversation = get_conversation_or_404(session, conversation_id, customer.id)
    return ConversationHistory(
        id=conversation.id,
        created_at=conversation.created_at,
        turns=[
            ConversationTurnResponse.model_validate(turn)
            for turn in list_turns(session, conversation.id)
        ],
    )
