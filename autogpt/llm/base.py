from dataclasses import dataclass, field
from enum import StrEnum
from typing import List, TypedDict


class Message(TypedDict):
    """OpenAI Message object containing a role and the message content"""

    role: str
    content: str


class ModelType(StrEnum):
    chat = "chat"
    embedding = "embedding"


@dataclass
class ModelInfo:
    name: str
    model_type: str
    prompt_token_cost: float
    completion_token_cost: float
    max_tokens: int


@dataclass
class LLMResponse:
    model_info: ModelInfo
    prompt_tokens_used: int = 0
    completion_tokens_used: int = 0


@dataclass
class EmbeddingResponse(LLMResponse):
    embedding: List[float] = field(default_factory=list)

    def __post_init__(self):
        if self.completion_tokens_used:
            raise ValueError("Embeddings should not have completion tokens used.")


@dataclass
class ChatCompletionResponse(LLMResponse):
    content: str = None
