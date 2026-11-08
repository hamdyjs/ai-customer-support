import asyncio
import json

import httpx2
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient
from sqlmodel import Session

from app.ai import MODEL, LLMService

ANALYSIS = {
    "category": "billing",
    "priority": "low",
    "summary": "The customer asked what they paid.",
    "suggested_response": "You paid $49.00.",
    "human_review_recommended": False,
}


def text_block(text):
    return {"type": "text", "text": text}


def tool_call(name, tool_id="toolu_1", tool_input=None):
    return {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input or {}}


def message_response(content, stop_reason="end_turn"):
    return httpx2.Response(
        200,
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )


def tool_response(*calls):
    return message_response(list(calls), stop_reason="tool_use")


def reply_response(reply="You paid $49.00."):
    body = {"reply": reply, "analysis": ANALYSIS}
    return message_response([text_block(json.dumps(body))])


class FakeClaude:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(json.loads(request.content))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def service(self):
        client = AsyncAnthropic(
            api_key="test-key",
            max_retries=0,
            http_client=DefaultAsyncHttpxClient(transport=httpx2.MockTransport(self)),
        )
        return LLMService(client)

    def generate(
        self, session, customer_id="cust_001", history=None, message="What did I pay?"
    ):
        return asyncio.run(
            self.service().generate_reply(
                message,
                history or [],
                lambda: Session(session.get_bind()),
                customer_id,
            )
        )

    def reply(self, session, customer_id="cust_001"):
        return self.generate(session, customer_id).reply
