import functools
import time

import openai
from colorama import Fore, Style
from openai.error import APIError, RateLimitError

from autogpt.config import Config
from autogpt.logs import logger


def retry_openai_api(
    num_retries: int = 10,
    backoff_base: float = 2.0,
    warn_user: bool = True,
):
    """Retry an OpenAI API call.

    Args:
        num_retries int: Number of retries. Defaults to 10.
        backoff_base float: Base for exponential backoff. Defaults to 2.
        warn_user bool: Whether to warn the user. Defaults to True.
    """
    retry_limit_msg = f"{Fore.RED}Error: " f"Reached rate limit, passing...{Fore.RESET}"
    api_key_error_msg = (
        f"Please double check that you have setup a "
        f"{Fore.CYAN + Style.BRIGHT}PAID{Style.RESET_ALL} OpenAI API Account. You can "
        f"read more here: {Fore.CYAN}https://github.com/Significant-Gravitas/Auto-GPT#openai-api-keys-configuration{Fore.RESET}"
    )
    backoff_msg = (
        f"{Fore.RED}Error: API Bad gateway. Waiting {{backoff}} seconds...{Fore.RESET}"
    )

    no_response_typewriter_msg = {
        "title": "FAILED TO GET RESPONSE FROM OPENAI",
        "title_color": Fore.RED,
        "content": (
            "Auto-GPT has failed to get a response from OpenAI's services. "
            f"Try running Auto-GPT again, and if the problem the persists try "
            f"running it with `{Fore.CYAN}--debug{Fore.RESET}`."
        ),
    }

    def _wrapper(func):
        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            response = None
            user_warned = not warn_user
            num_attempts = num_retries + 1  # +1 for the first attempt
            for attempt in range(1, num_attempts + 1):
                try:
                    response = func(*args, **kwargs)
                    break

                except RateLimitError:
                    if attempt == num_attempts:
                        raise

                    logger.debug(retry_limit_msg)
                    if not user_warned:
                        logger.double_check(api_key_error_msg)
                        user_warned = True

                except APIError as e:
                    if (e.http_status != 502) or (attempt == num_attempts):
                        raise

                backoff = backoff_base ** (attempt + 2)
                logger.debug(backoff_msg.format(backoff=backoff))
                time.sleep(backoff)

            if response is None:
                logger.typewriter_log(**no_response_typewriter_msg)
                logger.double_check()
                quit(1)

        return _wrapped

    return _wrapper


@retry_openai_api()
def create_embedding(
    text: str,
    *_,
    **kwargs,
) -> openai.Embedding:
    """Create an embedding using the OpenAI API

    Args:
        text (str): The text to embed.
        kwargs: Other arguments to pass to the OpenAI API embedding creation call.

    Returns:
        openai.Embedding: The embedding object.
    """
    cfg = Config()
    return openai.Embedding.create(
        input=[text],
        api_key=cfg.openai_api_key,
        **kwargs,
    )


@retry_openai_api
def create_chat_completion(
    messages: list,
    *_,
    **kwargs,
) -> openai.ChatCompletion:
    """Create a chat completion using the OpenAI API

    Args:
        messages (list): A list of messages to feed to the chatbot.
        kwargs: Other arguments to pass to the OpenAI API chat completion call.

    Returns:
        openai.ChatCompletion: The chat completion object.

    """
    cfg = Config()
    return openai.Completion.create(
        messages=messages,
        api_key=cfg.openai_api_key,
        **kwargs,
    )
