"""The application entry point.  Can be invoked by a CLI or any other front end application."""
import logging
import sys
from pathlib import Path
from typing import Optional

from colorama import Back, Fore, Style

from autogpt.agent import Agent
from autogpt.config import Config, ConfigBuilder
from autogpt.logs import logger
from autogpt.memory.vector import get_memory
from autogpt.models.command_registry import CommandRegistry
from autogpt.plugins import scan_plugins
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT, construct_main_ai_config
from autogpt.utils import (
    get_current_git_branch,
    get_latest_bulletin,
    get_legal_warning,
    markdown_to_ansi_style,
)
from autogpt.workspace import Workspace
from scripts.install_plugin_deps import install_plugin_dependencies

COMMAND_CATEGORIES = [
    "autogpt.commands.execute_code",
    "autogpt.commands.file_operations",
    "autogpt.commands.web_search",
    "autogpt.commands.web_selenium",
    "autogpt.app",
    "autogpt.commands.task_statuses",
]


def run_auto_gpt(
    workspace_directory: str | Path,
    install_plugin_deps: bool,
    ai_name: Optional[str] = None,
    ai_role: Optional[str] = None,
    ai_goals: tuple[str] = tuple(),
    **config_overrides,
):
    # Configure logging before we do anything else.
    logger.set_level(logging.DEBUG if config_overrides["debug_mode"] else logging.INFO)

    config = ConfigBuilder.build_config_from_env(config_overrides)
    config.validate()
    # HACK: This is a hack to allow the config into the logger without having to pass it around everywhere
    # or import it directly.
    logger.config = config

    if install_plugin_deps:
        install_plugin_dependencies()

    # TODO: have this directory live outside the repository (e.g. in a user's
    #   home directory) and have it come in as a command line argument or part of
    #   the env file.
    workspace_directory = Workspace.get_workspace_directory(config, workspace_directory)

    # HACK: doing this here to collect some globals that depend on the workspace.
    Workspace.build_file_logger_path(config, workspace_directory)

    config.plugins = scan_plugins(config, config.debug_mode)
    # Create a CommandRegistry instance and scan default folder
    command_registry = CommandRegistry()

    logger.debug(
        f"The following command categories are disabled: {config.disabled_command_categories}"
    )
    enabled_command_categories = [
        x for x in COMMAND_CATEGORIES if x not in config.disabled_command_categories
    ]

    logger.debug(
        f"The following command categories are enabled: {enabled_command_categories}"
    )

    for command_category in enabled_command_categories:
        command_registry.import_commands(command_category)

    # Unregister commands that are incompatible with the current config
    incompatible_commands = []
    for command in command_registry.commands.values():
        if callable(command.enabled) and not command.enabled(config):
            command.enabled = False
            incompatible_commands.append(command)

    for command in incompatible_commands:
        command_registry.unregister(command)
        logger.debug(
            f"Unregistering incompatible command: {command.name}, "
            f"reason - {command.disabled_reason or 'Disabled by current config.'}"
        )

    ai_config = construct_main_ai_config(
        config,
        name=ai_name,
        role=ai_role,
        goals=ai_goals,
    )
    ai_config.command_registry = command_registry
    ai_name = ai_config.ai_name
    # print(prompt)
    # Initialize variables
    next_action_count = 0

    # add chat plugins capable of report to logger
    if config.chat_messages_enabled:
        for plugin in config.plugins:
            if hasattr(plugin, "can_handle_report") and plugin.can_handle_report():
                logger.info(f"Loaded plugin into logger: {plugin.__class__.__name__}")
                logger.chat_plugins.append(plugin)

    # Initialize memory and make sure it is empty.
    # this is particularly important for indexing and referencing pinecone memory
    memory = get_memory(config)
    memory.clear()
    logger.typewriter_log(
        "Using memory of type:", Fore.GREEN, f"{memory.__class__.__name__}"
    )
    logger.typewriter_log("Using Browser:", Fore.GREEN, config.selenium_web_browser)
    system_prompt = ai_config.construct_full_prompt(config)
    if config.debug_mode:
        logger.typewriter_log("Prompt:", Fore.GREEN, system_prompt)

    agent = Agent(
        ai_name=ai_name,
        memory=memory,
        next_action_count=next_action_count,
        command_registry=command_registry,
        system_prompt=system_prompt,
        triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
        workspace_directory=workspace_directory,
        ai_config=ai_config,
        config=config,
    )
    agent.start_interaction_loop()


def do_agpt_preamble(config: Config):
    if config.debug_mode:
        logger.typewriter_log("Debug Mode: ", Fore.GREEN, "ENABLED")

    if config.continuous_mode:
        logger.typewriter_log("Continuous Mode: ", Fore.RED, "ENABLED")
        logger.typewriter_log(
            "WARNING: ",
            Fore.RED,
            "Continuous mode is not recommended. It is potentially dangerous and may"
            " cause your AI to run forever or carry out actions you would not usually"
            " authorise. Use at your own risk.",
        )
        if config.continuous_limit:
            logger.typewriter_log(
                "Continuous Limit: ", Fore.GREEN, f"{config.continuous_limit}"
            )

        for line in get_legal_warning().split("\n"):
            logger.warn(markdown_to_ansi_style(line), "LEGAL:", Fore.RED)

    if config.allow_downloads:
        logger.typewriter_log("Native Downloading:", Fore.GREEN, "ENABLED")
        logger.typewriter_log(
            "WARNING: ",
            Fore.YELLOW,
            f"{Back.LIGHTYELLOW_EX}Auto-GPT will now be able to download and save files to your machine.{Back.RESET} "
            + "It is recommended that you monitor any files it downloads carefully.",
        )
        logger.typewriter_log(
            "WARNING: ",
            Fore.YELLOW,
            f"{Back.RED + Style.BRIGHT}ALWAYS REMEMBER TO NEVER OPEN FILES YOU AREN'T SURE OF!{Style.RESET_ALL}",
        )

    if config.speak_mode:
        logger.typewriter_log("Speak Mode: ", Fore.GREEN, "ENABLED")

    logger.typewriter_log(
        "Language Model Configuration: ",
        Fore.GREEN,
        f"Fast LLM - {config.fast_llm}, Smart LLM - {config.smart_llm}",
    )

    logger.typewriter_log("Memory Backend: ", Fore.GREEN, config.memory_backend)

    if config.skip_reprompt:
        logger.typewriter_log("Skip Re-prompt: ", Fore.GREEN, "ENABLED")

    if config.ai_settings_file:
        logger.typewriter_log(
            "Using AI Settings File:", Fore.GREEN, config.ai_settings_file
        )
    logger.typewriter_log(
        "Using Prompt Settings File:", Fore.GREEN, config.prompt_settings_file
    )

    if not config.skip_news:
        motd, is_new_motd = get_latest_bulletin()
        if motd:
            motd = markdown_to_ansi_style(motd)
            for motd_line in motd.split("\n"):
                logger.info(motd_line, "NEWS:", Fore.GREEN)
            if is_new_motd and not config.chat_messages_enabled:
                input(
                    Fore.MAGENTA
                    + Style.BRIGHT
                    + "NEWS: Bulletin was updated! Press Enter to continue..."
                    + Style.RESET_ALL
                )

        git_branch = get_current_git_branch()
        if git_branch and git_branch != "stable":
            logger.typewriter_log(
                "WARNING: ",
                Fore.RED,
                f"You are running on `{git_branch}` branch "
                "- this is not a supported branch.",
            )
        if sys.version_info < (3, 10):
            logger.typewriter_log(
                "WARNING: ",
                Fore.RED,
                "You are running on an older version of Python. "
                "Some people have observed problems with certain "
                "parts of Auto-GPT with this version. "
                "Please consider upgrading to Python 3.10 or higher.",
            )
