import pytest

from autogpt.modelsinfo import CHAT_COSTS, EMBEDDING_COSTS, COSTS
from autogpt.llm.llm_utils import get_ada_embedding, create_chat_completion


@pytest.fixture(params=[0, 5, 10])
def prompt_tokens(request):
    return request.param


@pytest.fixture(params=[0, 5, 10])
def completion_tokens(request):
    return request.param


@pytest.fixture(params=list(CHAT_COSTS))
def chat_model(request):
    return request.param


@pytest.fixture(params=list(EMBEDDING_COSTS))
def embedding_model(request):
    return request.param


@pytest.fixture
def mock_create_embedding(mocker, prompt_tokens):
    mock_response = mocker.MagicMock()
    mock_response.usage.prompt_tokens = prompt_tokens
    mock_response.__getitem__.side_effect = lambda key: [{"embedding": [0.1, 0.2, 0.3]}]
    return mocker.patch(
        "autogpt.llm.llm_utils.openai_provider.create_embedding", return_value=mock_response
    )


@pytest.fixture
def mock_create_chat_completion(mocker, prompt_tokens, completion_tokens):
    mock_response = mocker.MagicMock()
    mock_response.usage.prompt_tokens = prompt_tokens
    mock_response.usage.completion_tokens = completion_tokens
    return mocker.patch(
        "autogpt.llm.llm_utils.openai_provider.create_chat_completion", return_value=mock_response
    )


def test_create_chat_completion(
    mock_create_chat_completion,
    api_manager,
    chat_model,
    prompt_tokens,
    completion_tokens,
):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
    ]

    response = create_chat_completion(messages, model=chat_model)


def test_get_ada_embedding(mock_create_embedding, api_manager, embedding_model, prompt_tokens):
    arg = "test"
    embedding = get_ada_embedding(arg)
    mock_create_embedding.assert_called_once_with(arg, model=embedding_model)

    assert embedding == [0.1, 0.2, 0.3]

    cost = COSTS[embedding_model]["prompt"]
    assert api_manager.get_total_prompt_tokens() == prompt_tokens
    assert api_manager.get_total_completion_tokens() == 0
    assert api_manager.get_total_cost() == (prompt_tokens * cost) / 1000



