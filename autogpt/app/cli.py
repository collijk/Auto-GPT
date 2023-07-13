"""Main script for the autogpt package."""
from typing import Optional

import click


@click.group(invoke_without_command=True)
@click.option("-c", "--continuous", is_flag=True, help="Enable Continuous Mode")
@click.option(
    "--skip-reprompt",
    "-y",
    is_flag=True,
    help="Skips the re-prompting messages at the beginning of the script",
)
@click.option(
    "--ai-settings",
    "-C",
    help="Specifies which ai_settings.yaml file to use, will also automatically skip the re-prompt.",
    type=click.Path(exists=True),
)
@click.option(
    "--prompt-settings",
    "-P",
    help="Specifies which prompt_settings.yaml file to use.",
    type=click.Path(exists=True),
)
@click.option(
    "-l",
    "--continuous-limit",
    type=int,
    help="Defines the number of times to run in continuous mode",
)
@click.option("--speak", is_flag=True, help="Enable Speak Mode")
@click.option("--debug", is_flag=True, help="Enable Debug Mode")
@click.option("--gpt3only", is_flag=True, help="Enable GPT3.5 Only Mode")
@click.option("--gpt4only", is_flag=True, help="Enable GPT4 Only Mode")
@click.option(
    "--use-memory",
    "-m",
    "memory_type",
    type=str,
    help="Defines which Memory backend to use",
)
@click.option(
    "-b",
    "--browser-name",
    help="Specifies which web-browser to use when using selenium to scrape the web.",
)
@click.option(
    "--allow-downloads",
    is_flag=True,
    help="Dangerous: Allows Auto-GPT to download files natively.",
)
@click.option(
    "--skip-news",
    is_flag=True,
    help="Specifies whether to suppress the output of latest news on startup.",
)
@click.option(
    # TODO: this is a hidden option for now, necessary for integration testing.
    #   We should make this public once we're ready to roll out agent specific workspaces.
    "--workspace-directory",
    "-w",
    type=click.Path(),
    hidden=True,
)
@click.option(
    "--install-plugin-deps",
    is_flag=True,
    help="Installs external dependencies for 3rd party plugins.",
)
@click.option(
    "--ai-name",
    type=str,
    help="AI name override",
)
@click.option(
    "--ai-role",
    type=str,
    help="AI role override",
)
@click.option(
    "--ai-goal",
    type=str,
    multiple=True,
    help="AI goal override; may be used multiple times to pass multiple goals",
)
@click.pass_context
def main(
    ctx: click.Context,
    continuous: bool,
    continuous_limit: int,
    ai_settings: str,
    prompt_settings: str,
    skip_reprompt: bool,
    speak: bool,
    debug: bool,
    gpt3only: bool,
    gpt4only: bool,
    memory_type: str,
    browser_name: str,
    allow_downloads: bool,
    skip_news: bool,
    workspace_directory: str,
    install_plugin_deps: bool,
    ai_name: Optional[str],
    ai_role: Optional[str],
    ai_goal: tuple[str],
) -> None:
    """
    Welcome to AutoGPT an experimental open-source application showcasing the capabilities of the GPT-4 pushing the boundaries of AI.

    Start an Auto-GPT assistant.
    """
    # Put imports inside function to avoid importing everything when starting the CLI
    import logging

    from autogpt.app.configurator import extract_env_file_configuration
    from autogpt.app.main import run_auto_gpt
    from autogpt.config import GPT_3_MODEL, GPT_4_MODEL
    from autogpt.logs import logger
    from autogpt.memory.vector import get_supported_memory_backends

    if ctx.invoked_subcommand is None:
        # Configure logging before we do anything else.
        logger.set_level(logging.DEBUG if debug else logging.INFO)

        ###################
        # Check CLI usage #
        ###################
        # This section should do CLI usage validation only.

        # Check if continuous limit is used without continuous mode
        if continuous_limit and not continuous:
            raise click.UsageError(
                "--continuous-limit can only be used with --continuous"
            )

        if gpt3only and gpt4only:
            raise click.UsageError("--gpt3only and --gpt4only cannot be used together")

        supported_memory = get_supported_memory_backends()
        if memory_type and memory_type not in supported_memory:
            raise click.UsageError(
                f"--use-memory must be one of {supported_memory}, got {memory_type}"
            )

        ###########################################
        # Map from CLI options to Auto-GPT config #
        ###########################################
        command_line_configuration = {
            "debug_mode": debug,
            "continuous_mode": continuous,
            "continuous_limit": continuous_limit,
            "speak_mode": speak,
            "skip_reprompt": skip_reprompt,
            "selenium_web_browser": browser_name,
            "allow_downloads": allow_downloads,
            "skip_news": skip_news,
        }
        if gpt3only:
            command_line_configuration["fast_llm"] = GPT_3_MODEL
            command_line_configuration["smart_llm"] = GPT_3_MODEL
        elif gpt4only:
            command_line_configuration["fast_llm"] = GPT_4_MODEL
            command_line_configuration["smart_llm"] = GPT_4_MODEL

        if memory_type:
            command_line_configuration["memory_backend"] = memory_type

        if ai_settings:
            command_line_configuration["ai_settings"] = ai_settings
            command_line_configuration["skip_reprompt"] = True
        if prompt_settings:
            command_line_configuration["prompt_settings_file"] = prompt_settings

        environment_configuration = extract_env_file_configuration()

        # Override environment configuration with command line configuration
        configuration_overrides = {
            **environment_configuration,
            **command_line_configuration,
        }

        run_auto_gpt(
            configuration_overrides,
            workspace_directory,
            install_plugin_deps,
            ai_name,
            ai_role,
            ai_goal,
        )


if __name__ == "__main__":
    main()
