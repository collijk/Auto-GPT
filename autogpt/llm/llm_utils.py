from __future__ import annotations

from typing import List, Optional

from colorama import Fore

from autogpt.api_manager import ApiManager
from autogpt.config import Config
from autogpt.llm import openai_provider
from autogpt.logs import logger
from autogpt.types.openai import Message


def call_ai_function(
    function: str, args: list, description: str, model: str | None = None
) -> str:
    """Call an AI function

    This is a magic function that can do anything with no-code. See
    https://github.com/Torantulino/AI-Functions for more info.

    Args:
        function (str): The function to call
        args (list): The arguments to pass to the function
        description (str): The description of the function
        model (str, optional): The model to use. Defaults to None.

    Returns:
        str: The response from the function
    """
    cfg = Config()
    if model is None:
        model = cfg.smart_llm_model
    # For each arg, if any are None, convert to "None":
    args = [str(arg) if arg is not None else "None" for arg in args]
    # parse args to comma separated string
    args: str = ", ".join(args)
    messages: List[Message] = [
        {
            "role": "system",
            "content": f"You are now the following python function: ```# {description}"
            f"\n{function}```\n\nOnly respond with your `return` value.",
        },
        {"role": "user", "content": args},
    ]

    return create_chat_completion(model=model, messages=messages, temperature=0)


# Overly simple abstraction until we create something better
def create_chat_completion(
    messages: List[Message],  # type: ignore
    model: Optional[str] = None,
    temperature: float = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Create a chat completion using the OpenAI API

    Args:
        messages (List[Message]): The messages to send to the chat completion
        model (str, optional): The model to use. Defaults to None.
        temperature (float, optional): The temperature to use. Defaults to 0.9.
        max_tokens (int, optional): The max tokens to use. Defaults to None.

    Returns:
        str: The response from the chat completion
    """
    cfg = Config()
    if temperature is None:
        temperature = cfg.temperature

    logger.debug(
        f"{Fore.GREEN}Creating chat completion with model {model}, "
        f"temperature {temperature}, max_tokens {max_tokens}{Fore.RESET}"
    )

    chat_completion_kwargs = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for plugin in cfg.plugins:
        if plugin.can_handle_chat_completion(messages, **chat_completion_kwargs):
            message = plugin.handle_chat_completion(messages, **chat_completion_kwargs)
            if message is not None:
                # Fixme: using the first available plugin is bound to be confusing for users.
                return message

    if cfg.use_azure:
        chat_completion_kwargs["deployment_id"] = cfg.get_azure_deployment_id_for_model(
            model
        )

    response = openai_provider.create_chat_completion(
        messages, **chat_completion_kwargs
    )
    api_manager = ApiManager()
    api_manager.update_cost(
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        model=model,
    )

    response = response.choices[0].message["content"]
    for plugin in cfg.plugins:
        if plugin.can_handle_on_response():
            response = plugin.on_response(response)

    return response


def get_ada_embedding(text: str) -> List[float]:
    """Get an embedding from the ada model.

    Args:
        text (str): The text to embed.

    Returns:
        List[float]: The embedding.
    """
    cfg = Config()
    model = "text-embedding-ada-002"
    text = text.replace("\n", " ")

    if cfg.use_azure:
        kwargs = {"engine": cfg.get_azure_deployment_id_for_model(model)}
    else:
        kwargs = {"model": model}

    embedding = openai_provider.create_embedding(text, **kwargs)
    api_manager = ApiManager()
    api_manager.update_cost(
        prompt_tokens=embedding.usage.prompt_tokens,
        completion_tokens=0,
        model=model,
    )
    return embedding["data"][0]["embedding"]
