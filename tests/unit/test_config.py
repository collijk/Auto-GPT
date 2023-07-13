"""
Test cases for the config class, which handles the configuration settings
for the AI and ensures it behaves as a singleton.
"""
from unittest import mock

import pytest

from autogpt.app.configurator import (
    validate_configuration,
)
from autogpt.config import Config, GPT_3_MODEL, GPT_4_MODEL
from autogpt.workspace.workspace import Workspace


def test_initial_values(config: Config):
    """
    Test if the initial values of the config class attributes are set correctly.
    """
    assert not config.debug_mode
    assert not config.continuous_mode
    assert not config.speak_mode
    assert config.fast_llm == GPT_3_MODEL
    assert config.smart_llm == GPT_4_MODEL


def test_missing_azure_config(workspace: Workspace):
    config_file = workspace.get_path("azure_config.yaml")
    with pytest.raises(FileNotFoundError):
        Config(
            use_azure=True,
            azure_config_file=str(config_file),
        )

    config_file.write_text("")
    config = Config(
        use_azure=True,
        azure_config_file=str(config_file),
    )

    assert config.openai_api_type == "azure"
    assert config.openai_api_base == ""
    assert config.openai_api_version == "2023-03-15-preview"
    assert config.azure_model_to_deployment_id_map == {}


def test_azure_config(config: Config, workspace: Workspace) -> None:
    config_file = workspace.get_path("azure_config.yaml")
    yaml_content = f"""
azure_api_type: azure
azure_api_base: https://dummy.openai.azure.com
azure_api_version: 2023-06-01-preview
azure_model_map:
    fast_llm_deployment_id: FAST-LLM_ID
    smart_llm_deployment_id: SMART-LLM_ID
    embedding_model_deployment_id: embedding-deployment-id-for-azure
"""
    config_file.write_text(yaml_content)

    config = Config(
        use_azure=True,
        azure_config_file=str(config_file),
    )

    assert config.openai_api_type == "azure"
    assert config.openai_api_base == "https://dummy.openai.azure.com"
    assert config.openai_api_version == "2023-06-01-preview"
    assert config.azure_model_to_deployment_id_map == {
        "fast_llm_deployment_id": "FAST-LLM_ID",
        "smart_llm_deployment_id": "SMART-LLM_ID",
        "embedding_model_deployment_id": "embedding-deployment-id-for-azure",
    }

    fast_llm = config.fast_llm
    smart_llm = config.smart_llm
    assert (
        config.get_azure_credentials(config.fast_llm)["deployment_id"] == "FAST-LLM_ID"
    )
    assert (
        config.get_azure_credentials(config.smart_llm)["deployment_id"]
        == "SMART-LLM_ID"
    )

    # Emulate --gpt4only
    config.fast_llm = smart_llm
    assert (
        config.get_azure_credentials(config.fast_llm)["deployment_id"] == "SMART-LLM_ID"
    )
    assert (
        config.get_azure_credentials(config.smart_llm)["deployment_id"]
        == "SMART-LLM_ID"
    )

    # Emulate --gpt3only
    config.fast_llm = config.smart_llm = fast_llm
    assert (
        config.get_azure_credentials(config.fast_llm)["deployment_id"] == "FAST-LLM_ID"
    )
    assert (
        config.get_azure_credentials(config.smart_llm)["deployment_id"] == "FAST-LLM_ID"
    )


@pytest.mark.parametrize(
    "fast_llm,smart_llm,available_llm",
    [
        (GPT_4_MODEL, GPT_4_MODEL, GPT_4_MODEL),
        (GPT_4_MODEL, GPT_3_MODEL, GPT_3_MODEL),
        (GPT_3_MODEL, GPT_4_MODEL, GPT_3_MODEL),
        (GPT_3_MODEL, GPT_3_MODEL, GPT_3_MODEL),
    ],
)
def test_create_config_valid_llm(fast_llm, smart_llm, available_llm) -> None:
    with mock.patch("autogpt.app.configurator.openai") as mock_get_models:
        mock_get_models.Model.list.return_value = {"data": [{"id": available_llm}]}
        config = Config(
            fast_llm=fast_llm,
            smart_llm=smart_llm,
        )
        config.openai_api_key = "dummy"
        config = validate_configuration(config)
        assert config.fast_llm == available_llm
        assert config.smart_llm == available_llm


