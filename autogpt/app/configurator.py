"""Configurator module."""
from __future__ import annotations

import re
from typing import Any, Literal

import yaml
from colorama import Fore
from dotenv import dotenv_values
import openai

from autogpt.config import GPT_3_MODEL, REPO_ROOT, Config
from autogpt.logs import logger


def extract_env_file_configuration() -> dict[str, str]:
    """Extracts the configuration from the environment variables."""
    env_file = REPO_ROOT / ".env"
    env_values = {}
    if env_file.exists():
        env_values = dotenv_values(env_file)

    return {
        _map_env_key(key): _process_env_value(key, value)
        for key, value in env_values.items()
    }


def _map_env_key(key: str) -> str:
    """Map from environment variable names to config keys."""
    env_key_map = {
        "AUTHORISE_COMMAND_KEY": "authorise_key",
        "FAST_LLM_MODEL": "fast_llm",
        "SMART_LLM_MODEL": "smart_llm",
        "USE_WEB_BROWSER": "selenium_web_browser",
        "HEADLESS_BROWSER": "selenium_headless",
        "OPEN_API_BASE_URL": "openai_api_base_url",
        "ALLOWLISTED_PLUGINS": "plugins_allowlist",
        "DENYLISTED_PLUGINS": "plugins_denylist",
    }
    return env_key_map[key] if key in env_key_map else key.lower()


def _process_env_value(key: str, value: str) -> Any:
    """Post-processes the environment variable value."""
    if key in ["IMAGE_SIZE", "REDIS_PORT"]:
        return int(value)

    if key in ["TEMPERATURE"]:
        return float(value)

    if value in ["True", "False"]:
        return value == "True"

    if len(value.split(",")) > 1:
        return value.split(",")

    return value


def validate_configuration(config: Config) -> Config:
    config.openai_api_key = _validate_openai_api_key(config)
    config.fast_llm = _validate_model(config.fast_llm, "fast_llm", config=config)
    config.smart_llm = _validate_model(config.smart_llm, "smart_llm", config=config)
    if config.ai_settings_file:
        config.ai_settings_file = _validate_settings_file(config.ai_settings_file)
    config.prompt_settings_file = _validate_settings_file(config.prompt_settings_file)
    return config


def _validate_openai_api_key(config: Config) -> str:
    """Check if the OpenAI API key is set in config.py or as an environment variable."""
    if not config.openai_api_key:
        logger.typewriter_log(
            "Please set your OpenAI API key in .env or as an environment variable.",
            Fore.RED,
            "You can get your key from https://platform.openai.com/account/api-keys",
        )
        openai_api_key = input(
            "If you do have the key, please enter your OpenAI API key now:\n"
        )
        key_pattern = r"^sk-\w{48}"
        openai_api_key = openai_api_key.strip()
        if re.search(key_pattern, openai_api_key):
            logger.typewriter_log(
                "OpenAI API key successfully set!",
                Fore.GREEN,
            )
            logger.typewriter_log(
                "NOTE",
                Fore.YELLOW,
                "The API key you've set is only temporary.\n"
                "For longer sessions, please set it in .env file",
            )
            return openai_api_key
        else:
            raise RuntimeError("Invalid OpenAI API key!")


def _validate_model(
    model_name: str,
    model_type: Literal["smart_llm", "fast_llm"],
    config: Config,
) -> str:
    """Check if model is available for use. If not, return gpt-3.5-turbo."""
    openai_credentials = config.get_openai_credentials(model_name)

    try:
        models = openai.Model.list(**openai_credentials)["data"]
        models = [model for model in models if "gpt" in model["id"]]
        if any(model_name in m["id"] for m in models):
            return model_name
    except openai.error.AuthenticationError:
        logger.typewriter_log(
            "Please set your OpenAI API key in .env or as an environment variable.",
            Fore.RED,
            "You can get your key from https://platform.openai.com/account/api-keys",
        )
        raise RuntimeError("No valid API key found!")

    logger.typewriter_log(
        "WARNING: ",
        Fore.YELLOW,
        f"You do not have access to {model_name}. "
        f"Setting {model_type} to {GPT_3_MODEL}.",
    )
    return GPT_3_MODEL


def _validate_settings_file(settings_file: str) -> str:
    try:
        with open(settings_file, encoding="utf-8") as fp:
            yaml.load(fp.read(), Loader=yaml.FullLoader)
            return settings_file
    except FileNotFoundError:
        message = f"The file {Fore.CYAN}`{settings_file}`{Fore.RESET} wasn't found"
    except yaml.YAMLError as e:
        message = (
            f"There was an issue while trying to read with your AI Settings file: {e}",
        )
    logger.typewriter_log(
        "FAILED FILE VALIDATION",
        Fore.RED,
        message,
    )
    logger.double_check()
    raise RuntimeError(message)
