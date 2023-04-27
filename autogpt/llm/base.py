from dataclasses import dataclass
from typing import List, TypedDict


class Message(TypedDict):
    """OpenAI Message object containing a role and the message content"""
    role: str
    content: str


@dataclass
class LLMResponse:
    model: str
    prompt_tokens_used: int
    completion_tokens_used: int


@dataclass
class EmbeddingResponse(LLMResponse):
    completion_tokens_used: in
    embedding: List[float]


@dataclass
class ChatCompletionResponse(LLMResponse):
    content: str
