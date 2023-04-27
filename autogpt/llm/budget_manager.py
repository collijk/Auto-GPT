from __future__ import annotations
import math

from autogpt.logs import logger
from autogpt.singleton import Singleton
from autogpt.llm.base import LLMResponse



class BudgetManager(metaclass=Singleton):

    # TODO: Tune these parameters.
    _GRACEFUL_SHUTDOWN_THRESHOLD = 0.005
    _WARNING_THRESHOLD = 0.01


    def __init__(self):
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost = 0
        self._total_budget = 0

    def reset(self):
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost = 0
        self._total_budget = 0.0

    @property
    def total_prompt_tokens(self):
        """Total prompt tokens used by the API calls."""
        return self._total_prompt_tokens

    @property
    def total_completion_tokens(self):
        """Total completion tokens used by the API calls."""
        return self._total_completion_tokens

    @property
    def total_cost(self):
        """Total cost of the API calls."""
        return self._total_cost

    @property
    def total_budget(self):
        """Total user-defined budget for API calls."""
        return self._total_budget

    @property
    def remaining_budget(self):
        """Remaining budget for API calls."""
        return self._total_budget - self._total_cost

    @total_budget.setter
    def total_budget(self, total_budget: float | None):
        """Set the total user-defined budget for API calls."""
        self._total_budget = math.inf if total_budget is None else total_budget

    def update_cost(self, llm_response: LLMResponse):
        """
        Update the total cost, prompt tokens, and completion tokens.

        Args:
            llm_response (LLMResponse): The response from the LLM API call.

        """
        model_info = llm_response.model_info
        self._total_prompt_tokens += llm_response.prompt_tokens_used
        self._total_completion_tokens += llm_response.completion_tokens_used
        self._total_cost += (
            llm_response.prompt_tokens_used * model_info.prompt_token_cost
            + llm_response.completion_tokens_used * model_info.completion_token_cost
        ) / 1000
        logger.debug(f"Total running cost: ${self._total_cost:.3f}")

    def get_agent_budget_prompt(self) -> str:
        """Help the agent manage its budget by providing a prompt."""
        if self._total_budget is math.inf:
            return ""

        base_prompt = f"Your remaining API budget is ${self.remaining_budget:.3f}"
        if self.remaining_budget <= 0:
            base_prompt += " BUDGET EXCEEDED! SHUT DOWN!\n\n"
        elif self.remaining_budget < self._GRACEFUL_SHUTDOWN_THRESHOLD:
            base_prompt += " Budget very nearly exceeded! Shut down gracefully!\n\n"
        elif self.remaining_budget < self._WARNING_THRESHOLD:
            base_prompt += " Budget nearly exceeded. Finish up.\n\n"

        return base_prompt



