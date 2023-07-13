from autogpt.agent import Agent
from autogpt.app.prompt import construct_full_prompt
from autogpt.commands import COMMAND_CATEGORIES
from autogpt.config import AIConfig, Config
from autogpt.memory.vector import get_memory
from autogpt.models.command_registry import CommandRegistry
from autogpt.prompts.default_prompts import DEFAULT_TRIGGERING_PROMPT
from autogpt.workspace import Workspace


def run_task(task) -> None:
    agent = bootstrap_agent(task)
    agent.start_interaction_loop()


def bootstrap_agent(task):
    config = Config(plain_output=True, memory_backend="no_memory")
    command_registry = get_command_registry(config)
    workspace_directory = Workspace.get_workspace_directory(config)
    workspace_directory_path = Workspace.make_workspace(workspace_directory)
    Workspace.build_file_logger_path(config, workspace_directory_path)
    ai_config = AIConfig(
        ai_name="Auto-GPT",
        ai_role="a multi-purpose AI assistant.",
        ai_goals=[task.user_input],
    )
    ai_config.command_registry = command_registry
    system_prompt = construct_full_prompt(config, ai_config, command_registry)
    return Agent(
        ai_name="Auto-GPT",
        memory=get_memory(config),
        command_registry=command_registry,
        ai_config=ai_config,
        config=config,
        next_action_count=0,
        system_prompt=system_prompt,
        triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
        workspace_directory=str(workspace_directory_path),
    )


def get_command_registry(config: Config):
    command_registry = CommandRegistry()
    enabled_command_categories = [
        x for x in COMMAND_CATEGORIES if x not in config.disabled_command_categories
    ]
    for command_category in enabled_command_categories:
        command_registry.import_commands(command_category)
    return command_registry
