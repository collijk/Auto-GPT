"""Configuration class to store the state of bools for different scripts access."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from auto_gpt_plugin_template import AutoGPTPluginTemplate
from pydantic import Field, validator

import autogpt
from autogpt.core.configuration.schema import SystemSettings
from autogpt.plugins.plugins_config import PluginsConfig

REPO_ROOT = Path(autogpt.__file__).parent.parent
AZURE_CONFIG_PATH = REPO_ROOT / "azure.yaml"
PLUGINS_CONFIG_PATH = REPO_ROOT / "plugins_config.yaml"
GPT_4_MODEL = "gpt-4"
GPT_3_MODEL = "gpt-3.5-turbo"


class Config(SystemSettings, arbitrary_types_allowed=True):
    name: str = "Auto-GPT configuration"
    description: str = "Default configuration for the Auto-GPT application."
    ########################
    # Application Settings #
    ########################
    skip_news: bool = False
    skip_reprompt: bool = False
    authorise_key: str = "y"
    exit_key: str = "n"
    debug_mode: bool = False
    plain_output: bool = False
    chat_messages_enabled: bool = True
    # TTS configuration
    speak_mode: bool = False
    text_to_speech_provider: str = "gtts"
    streamelements_voice: str = "Brian"
    elevenlabs_voice_id: Optional[str] = None

    ##########################
    # Agent Control Settings #
    ##########################
    # Paths
    ai_settings_file: Optional[str] = None
    prompt_settings_file: str = "prompt_settings.yaml"
    workspace_path: Optional[str] = None
    file_logger_path: Optional[str] = None
    # Model configuration
    fast_llm: str = "gpt-3.5-turbo"
    smart_llm: str = "gpt-4"
    temperature: float = 0
    openai_functions: bool = False
    embedding_model: str = "text-embedding-ada-002"
    browse_spacy_language_model: str = "en_core_web_sm"
    # Run loop configuration
    continuous_mode: bool = False
    continuous_limit: int = 0

    ##########
    # Memory #
    ##########
    memory_backend: str = "json_file"
    memory_index: str = "auto-gpt-memory"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    wipe_redis_on_start: bool = True

    ############
    # Commands #
    ############
    # General
    disabled_command_categories: list[str] = Field(default_factory=list)
    # File ops
    restrict_to_workspace: bool = True
    allow_downloads: bool = False
    # Shell commands
    shell_command_control: str = "denylist"
    execute_local_commands: bool = False
    shell_denylist: list[str] = Field(default_factory=lambda: ["sudo", "su"])
    shell_allowlist: list[str] = Field(default_factory=list)
    # Text to image
    image_provider: Optional[str] = None
    huggingface_image_model: str = "CompVis/stable-diffusion-v1-4"
    sd_webui_url: Optional[str] = "http://localhost:7860"
    image_size: int = 256
    # Audio to text
    audio_to_text_provider: str = "huggingface"
    huggingface_audio_to_text_model: Optional[str] = None
    # Web browsing
    selenium_web_browser: str = "chrome"
    selenium_headless: bool = True
    user_agent: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"

    ###################
    # Plugin Settings #
    ###################
    plugins_dir: str = "plugins"
    plugins_config_file: str = str(PLUGINS_CONFIG_PATH)
    plugins_config: PluginsConfig = Field(
        default_factory=lambda: PluginsConfig(plugins={})
    )
    plugins: list[AutoGPTPluginTemplate] = Field(default_factory=list, exclude=True)
    plugins_allowlist: list[str] = Field(default_factory=list)
    plugins_denylist: list[str] = Field(default_factory=list)
    plugins_openai: list[str] = Field(default_factory=list)

    ###############
    # Credentials #
    ###############
    # OpenAI
    openai_api_key: Optional[str] = None
    openai_api_type: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_api_version: Optional[str] = None
    openai_organization: Optional[str] = None
    use_azure: bool = False
    azure_config_file: Optional[str] = str(AZURE_CONFIG_PATH)
    azure_model_to_deployment_id_map: Optional[Dict[str, str]] = None
    # Elevenlabs
    elevenlabs_api_key: Optional[str] = None
    # Github
    github_api_key: Optional[str] = None
    github_username: Optional[str] = None
    # Google
    google_api_key: Optional[str] = None
    google_custom_search_engine_id: Optional[str] = None
    # Huggingface
    huggingface_api_token: Optional[str] = None
    # Stable Diffusion
    sd_webui_auth: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        self.plugins_config = PluginsConfig.load_config(
            self.plugins_config_file,
            self.plugins_denylist,
            self.plugins_allowlist,
        )

        if self.use_azure:
            with open(self.azure_config_file) as file:
                config_params = yaml.load(file, Loader=yaml.FullLoader) or {}
            self.openai_api_type = config_params.get("azure_api_type", "azure")
            self.openai_api_base = config_params.get("azure_api_base", "")
            self.openai_api_version = config_params.get(
                "azure_api_version", "2023-03-15-preview"
            )
            self.azure_model_to_deployment_id_map = config_params.get(
                "azure_model_map", {}
            )

    @validator("plugins", each_item=True)
    def validate_plugins(cls, p: AutoGPTPluginTemplate | Any):
        assert issubclass(
            p.__class__, AutoGPTPluginTemplate
        ), f"{p} does not subclass AutoGPTPluginTemplate"
        assert (
            p.__class__.__name__ != "AutoGPTPluginTemplate"
        ), f"Plugins must subclass AutoGPTPluginTemplate; {p} is a template instance"
        return p

    def get_openai_credentials(self, model: str) -> dict[str, str]:
        credentials = {
            "api_key": self.openai_api_key,
            "api_base": self.openai_api_base,
            "organization": self.openai_organization,
        }
        if self.use_azure:
            azure_credentials = self.get_azure_credentials(model)
            credentials.update(azure_credentials)
        return credentials

    def get_azure_credentials(self, model: str) -> dict[str, str]:
        """Get the kwargs for the Azure API."""

        # Fix --gpt3only and --gpt4only in combination with Azure
        fast_llm = (
            self.fast_llm
            if not (
                self.fast_llm == self.smart_llm
                and self.fast_llm.startswith(GPT_4_MODEL)
            )
            else f"not_{self.fast_llm}"
        )
        smart_llm = (
            self.smart_llm
            if not (
                self.smart_llm == self.fast_llm
                and self.smart_llm.startswith(GPT_3_MODEL)
            )
            else f"not_{self.smart_llm}"
        )

        deployment_id = {
            fast_llm: self.azure_model_to_deployment_id_map.get(
                "fast_llm_deployment_id",
                self.azure_model_to_deployment_id_map.get(
                    "fast_llm_model_deployment_id"  # backwards compatibility
                ),
            ),
            smart_llm: self.azure_model_to_deployment_id_map.get(
                "smart_llm_deployment_id",
                self.azure_model_to_deployment_id_map.get(
                    "smart_llm_model_deployment_id"  # backwards compatibility
                ),
            ),
            self.embedding_model: self.azure_model_to_deployment_id_map.get(
                "embedding_model_deployment_id"
            ),
        }.get(model, None)

        kwargs = {
            "api_type": self.openai_api_type,
            "api_base": self.openai_api_base,
            "api_version": self.openai_api_version,
        }
        if model == self.embedding_model:
            kwargs["engine"] = deployment_id
        else:
            kwargs["deployment_id"] = deployment_id
        return kwargs
