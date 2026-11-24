import os

import llm_sakura
import pytest

# To record new cassettes against the real API:
#   PYTEST_SAKURA_AI_ENGINE_TOKEN="$(llm keys get sakura)" uv run pytest --record-mode once
SAKURA_TOKEN = os.getenv("PYTEST_SAKURA_AI_ENGINE_TOKEN", "sakura-test-token")


def drop_cookies(response):
    response["headers"] = {
        name: value
        for name, value in response["headers"].items()
        if name.lower() != "set-cookie"
    }
    return response


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization", "cookie"],
        "before_record_response": drop_cookies,
        "decode_compressed_response": True,
    }


@pytest.fixture
def user_path(tmp_path):
    path = tmp_path / "llm.datasette.io"
    path.mkdir()
    return path


@pytest.fixture(autouse=True)
def env_setup(monkeypatch, user_path):
    monkeypatch.setenv("LLM_USER_PATH", str(user_path))
    monkeypatch.setenv("SAKURA_AI_ENGINE_TOKEN", SAKURA_TOKEN)


@pytest.fixture
def no_fetch(monkeypatch):
    "Use the models from the manual without calling /v1/models"
    monkeypatch.setattr(
        llm_sakura,
        "fetch_model_ids",
        lambda key: sorted({*llm_sakura.KNOWN_MODELS, *llm_sakura.KNOWN_EMBEDDING_MODELS}),
    )
