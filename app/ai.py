from enum import StrEnum
from functools import lru_cache
from typing import Annotated

from anthropic import APIError, AsyncAnthropic
from dotenv import load_dotenv
from fastapi import Depends
from pydantic import BaseModel, ValidationError

from app.db.models import TicketPriority

load_dotenv()

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 16000
SYSTEM_PROMPT = """\
You analyze customer support questions for Flowdesk, a project management product.

Categorize the question:
- billing: charges, invoices, payments, refunds, or plan pricing
- technical: bugs, errors, integrations, or product behaviour
- account: sign-in, profile, team members, or cancelling
- general: anything else, including how-to and product questions

Set the priority:
- high: the customer is blocked, or reports an incorrect charge
- medium: the customer has a problem but can keep working
- low: a general question or minor request

Summarize the question in one sentence, then draft a reply addressed to the customer.

You can't see this customer's account or billing data. The reply must never state
or guess those details, and must never promise a refund or an account change.

Recommend human review if answering needs account or billing data, if the customer
disputes a charge or asks for a refund, or if you're unsure of the answer."""


class QuestionCategory(StrEnum):
    billing = "billing"
    technical = "technical"
    account = "account"
    general = "general"


class QuestionAnalysis(BaseModel):
    category: QuestionCategory
    priority: TicketPriority
    summary: str
    suggested_response: str
    human_review_recommended: bool


class AnalysisError(Exception):
    pass


class LLMService:
    def __init__(self, client: AsyncAnthropic):
        self.client = client

    async def analyze_question(self, question: str) -> QuestionAnalysis:
        try:
            response = await self.client.messages.parse(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                output_config={"effort": "medium"},
                output_format=QuestionAnalysis,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": question}],
            )
        except ValidationError as error:
            raise AnalysisError("The model returned an invalid analysis") from error
        except APIError as error:
            raise AnalysisError("The Claude API request failed") from error
        if response.stop_reason != "end_turn" or response.parsed_output is None:
            raise AnalysisError(f"No analysis returned: {response.stop_reason}")
        return response.parsed_output


@lru_cache
def get_llm_service() -> LLMService:
    return LLMService(AsyncAnthropic())


LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]
