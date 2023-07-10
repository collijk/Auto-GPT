"""Main script for the autogpt package."""
from typing import Optional

import click
import yaml


def validate_yaml_loadable(ctx, param, yaml_file):
    if yaml_file:
        try:
            with open(yaml_file, encoding="utf-8") as fp:
                yaml.load(fp.read(), Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            click.secho("DOUBLE CHECK CONFIGURATION", fg="yellow")
            click.secho(
                "Please ensure you've setup and configured everything"
                " correctly. Read https://github.com/Torantulino/Auto-GPT#readme to "
                "double check. You can also create a github issue or join the discord"
                " and ask there!",
            )
            raise click.BadParameter(
                f"Could not load {yaml_file} as valid YAML: {e}"
            ) from e

    return yaml_file


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
    type=click.Path(exists=True),
    help="Specifies which ai_settings.yaml file to use, will also automatically skip the re-prompt.",
    callback=validate_yaml_loadable,
)
@click.option(
    "--prompt-settings",
    "-P",
    type=click.Path(exists=True),
    help="Specifies which prompt_settings.yaml file to use.",
    callback=validate_yaml_loadable,
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
    from autogpt.config import GPT_3_MODEL, GPT_4_MODEL
    from autogpt.main import run_auto_gpt
    from autogpt.memory.vector import get_supported_memory_backends

    if ctx.invoked_subcommand is None:
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
        config_overrides = {
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
            config_overrides["fast_llm"] = GPT_3_MODEL
            config_overrides["smart_llm"] = GPT_3_MODEL
        elif gpt4only:
            config_overrides["fast_llm"] = GPT_4_MODEL
            config_overrides["smart_llm"] = GPT_4_MODEL

        if memory_type:
            config_overrides["memory_backend"] = memory_type

        if ai_settings:
            config_overrides["ai_settings"] = ai_settings
            config_overrides["skip_reprompt"] = True
        if prompt_settings:
            config_overrides["prompt_settings_file"] = prompt_settings

        run_auto_gpt(
            workspace_directory=workspace_directory,
            install_plugin_deps=install_plugin_deps,
            ai_name=ai_name,
            ai_role=ai_role,
            ai_goals=ai_goal,
            **config_overrides,
        )


if __name__ == "__main__":
    main()
