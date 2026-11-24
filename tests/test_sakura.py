import json
from types import SimpleNamespace

import httpx2
import llm
import llm_sakura
import pytest
from click.testing import CliRunner
from llm.cli import cli
from llm.parts import ReasoningPart
from openai.types.chat import ChatCompletion, ChatCompletionChunk

TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\x0f"
    b"\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


def chunk(delta=None, finish_reason=None, usage=None):
    data = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [],
    }
    if delta is not None or finish_reason is not None:
        data["choices"] = [
            {"index": 0, "delta": delta or {}, "finish_reason": finish_reason}
        ]
    if usage:
        data["usage"] = usage
    return ChatCompletionChunk.model_validate(data)


def completion(message, finish_reason="stop"):
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "m",
            "choices": [
                {"index": 0, "message": message, "finish_reason": finish_reason}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        }
    )


USAGE = {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}

REASONING_STREAM = [
    chunk({"role": "assistant", "content": ""}),
    chunk({"reasoning": "Think"}),
    chunk({"reasoning": "ing..."}),
    chunk({"content": "Hello"}),
    chunk({"content": " there"}, finish_reason="stop"),
    chunk(usage=USAGE),
]


@pytest.fixture
def fake_api(monkeypatch):
    "Replace the OpenAI client, recording every request"
    state = SimpleNamespace(calls=[], reply=REASONING_STREAM)

    def create(**kwargs):
        state.calls.append(kwargs)
        return state.reply

    async def acreate(**kwargs):
        state.calls.append(kwargs)
        reply = state.reply
        if not kwargs["stream"]:
            return reply

        async def iterate():
            for item in reply:
                yield item

        return iterate()

    def get_client(self, key, *, async_=False):
        state.key = self.get_key(key)
        return SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=acreate if async_ else create)
            )
        )

    monkeypatch.setattr(llm_sakura._SharedSakura, "get_client", get_client, raising=False)
    return state


# Model registration


def test_documented_models_registered_without_key(monkeypatch):
    monkeypatch.delenv("SAKURA_AI_ENGINE_TOKEN")
    ids = {model.model_id for model in llm.get_models()}
    assert "sakura/preview/GLM-5.3-Flash" in ids
    assert "sakura/gpt-oss-120b" in ids


def test_model_list_fetched_and_cached(monkeypatch, user_path):
    calls = []

    def fake_get(url, headers, **kwargs):
        calls.append((url, headers["Authorization"]))
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": "preview/GLM-5.3-Flash"},
                    {"id": "preview/New-Model"},
                    {"id": "multilingual-e5-large"},
                    {"id": "whisper-large-v3-turbo"},
                ],
            },
            request=httpx2.Request("GET", url),
        )

    monkeypatch.setattr(llm_sakura.httpx2, "get", fake_get)
    ids = {model.model_id for model in llm.get_models()}
    assert "sakura/preview/New-Model" in ids
    assert "sakura/multilingual-e5-large" not in ids
    assert "sakura/whisper-large-v3-turbo" not in ids
    embedding_ids = {
        model.model_id
        for model in llm.get_embedding_models()
        if model.model_id.startswith("sakura/")
    }
    assert embedding_ids == {"sakura/multilingual-e5-large"}
    # Not listed by the API any more, so not registered
    assert "sakura/gpt-oss-120b" not in ids
    assert calls == [(llm_sakura.MODELS_URL, "Bearer sakura-test-token")]
    assert json.loads((user_path / "sakura_models.json").read_text()) == [
        "multilingual-e5-large",
        "preview/GLM-5.3-Flash",
        "preview/New-Model",
        "whisper-large-v3-turbo",
    ]
    llm.get_models()
    assert len(calls) == 1, "second call should use the cache"


def test_api_failure_falls_back_to_manual(monkeypatch):
    def fake_get(url, **kwargs):
        raise httpx2.ConnectError("offline")

    monkeypatch.setattr(llm_sakura.httpx2, "get", fake_get)
    ids = {model.model_id for model in llm.get_models()}
    assert "sakura/gpt-oss-120b" in ids
    embedding_ids = {model.model_id for model in llm.get_embedding_models()}
    assert "sakura/preview/Qwen3-Embedding-4B-FP16" in embedding_ids


# Options


def option_names(model_id):
    return set(llm.get_model(model_id).Options.model_fields)


def test_ignored_parameters_are_not_offered(no_fetch):
    names = option_names("sakura/preview/GLM-5.3-Flash")
    assert {"temperature", "max_tokens", "json_object", "reasoning_effort"} <= names
    assert not names & {"top_p", "stop", "seed", "logprobs", "frequency_penalty"}


def test_reasoning_options_follow_the_manual(no_fetch):
    assert "reasoning_effort" not in option_names("sakura/preview/Kimi-K2.7-Code")
    assert "reasoning_effort" not in option_names("sakura/llm-jp-3.1-8x13b-instruct4")
    assert "thinking" in option_names("sakura/preview/Qwen3.6-35B-A3B")
    assert "thinking" not in option_names("sakura/gpt-oss-120b")
    model = llm.get_model("sakura/gpt-oss-120b")
    with pytest.raises(ValueError):
        model.Options(reasoning_effort="none")
    llm.get_model("sakura/preview/Qwen3.6-35B-A3B").Options(reasoning_effort="none")


def test_capabilities(no_fetch):
    assert not llm.get_model("sakura/gpt-oss-120b").attachment_types
    assert llm.get_model("sakura/preview/GLM-5.3-Flash").attachment_types
    assert not llm.get_model("sakura/preview/Qwen3-VL-30B-A3B-Instruct").supports_tools


# Requests


def test_stream_shows_reasoning(no_fetch, fake_api):
    model = llm.get_model("sakura/preview/GLM-5.3-Flash")
    response = model.prompt("hi", reasoning_effort="low", temperature=0.5)
    assert response.text() == "Hello there"
    call = fake_api.calls[0]
    assert call["model"] == "preview/GLM-5.3-Flash"
    assert call["stream"] is True
    assert call["reasoning_effort"] == "low"
    assert call["temperature"] == 0.5
    assert fake_api.key == "sakura-test-token"
    reasoning = [
        part
        for message in response.messages()
        for part in message.parts
        if isinstance(part, ReasoningPart)
    ]
    assert "".join(part.text for part in reasoning) == "Thinking..."
    assert response.response_json["reasoning"] == "Thinking..."
    assert response.usage().input == 5


def test_non_stream_reasoning(no_fetch, fake_api):
    fake_api.reply = completion(
        {"role": "assistant", "content": "Hello", "reasoning": "Hmm"}
    )
    model = llm.get_model("sakura/preview/GLM-5.3-Flash")
    response = model.prompt("hi", stream=False)
    assert response.text() == "Hello"
    assert response.response_json["choices"][0]["message"]["reasoning"] == "Hmm"


def test_thinking_switch_uses_chat_template_kwargs(no_fetch, fake_api):
    model = llm.get_model("sakura/preview/Qwen3.6-35B-A3B")
    model.prompt("hi", thinking=False).text()
    assert fake_api.calls[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_tool_call_stream(no_fetch, fake_api):
    fake_api.reply = [
        chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "llm_time", "arguments": ""},
                    }
                ]
            }
        ),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}),
        chunk({}, finish_reason="tool_calls"),
        chunk(usage=USAGE),
    ]
    model = llm.get_model("sakura/preview/GLM-5.3-Flash")

    def llm_time() -> str:
        "Current time"
        return "noon"

    response = model.prompt("time?", tools=[llm_time])
    response.text()
    calls = response.tool_calls()
    assert [(c.name, c.arguments) for c in calls] == [("llm_time", {})]
    assert fake_api.calls[0]["tools"][0]["function"]["name"] == "llm_time"


def test_empty_answer_at_max_tokens_is_an_error(no_fetch, fake_api):
    fake_api.reply = [
        chunk({"reasoning": "Long thoughts"}, finish_reason="length"),
        chunk(usage=USAGE),
    ]
    model = llm.get_model("sakura/preview/GLM-5.3-Flash")
    with pytest.raises(llm.ModelError, match="max_tokens"):
        model.prompt("hi", max_tokens=5).text()


def test_image_limit(no_fetch, fake_api):
    model = llm.get_model("sakura/preview/Qwen3-VL-30B-A3B-Instruct")
    images = [llm.Attachment(type="image/png", content=TINY_PNG) for _ in range(4)]
    with pytest.raises(llm.ModelError, match="at most 3 images"):
        model.prompt("describe", attachments=images).text()
    assert fake_api.calls == []
    fake_api.reply = REASONING_STREAM
    model.prompt("describe", attachments=images[:3]).text()
    assert len(fake_api.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (True, False))
async def test_async(no_fetch, fake_api, stream):
    if not stream:
        fake_api.reply = completion(
            {"role": "assistant", "content": "Hello there", "reasoning": "Hmm"}
        )
    model = llm.get_async_model("sakura/preview/GLM-5.3-Flash")
    response = model.prompt("hi", stream=stream)
    assert await response.text() == "Hello there"


# Commands


def test_models_command(no_fetch):
    result = CliRunner().invoke(cli, ["sakura", "models"])
    assert result.exit_code == 0, result.output
    assert "sakura/preview/Kimi-K2.7-Code" in result.output
    assert "Reasoning effort: cannot be changed" in result.output
    result = CliRunner().invoke(cli, ["sakura", "models", "--json"])
    data = json.loads(result.output)
    assert {"id": "gpt-oss-120b", "vision": False}.items() <= next(
        m for m in data if m["id"] == "gpt-oss-120b"
    ).items()


def test_refresh_needs_key(monkeypatch):
    monkeypatch.delenv("SAKURA_AI_ENGINE_TOKEN")
    result = CliRunner().invoke(cli, ["sakura", "refresh"])
    assert result.exit_code == 1
    assert "llm keys set sakura" in result.output


def test_refresh_reports_changes(monkeypatch):
    ids = ["a", "b"]
    monkeypatch.setattr(llm_sakura, "fetch_model_ids", lambda key: list(ids))
    result = CliRunner().invoke(cli, ["sakura", "refresh"])
    assert "Added: sakura/a" in result.output
    ids[:] = ["b", "c"]
    result = CliRunner().invoke(cli, ["sakura", "refresh"])
    assert "Added: sakura/c" in result.output
    assert "Removed: sakura/a" in result.output


def test_refresh_ignores_models_it_does_not_register(monkeypatch):
    monkeypatch.setattr(
        llm_sakura, "fetch_model_ids", lambda key: ["a", "whisper-large-v3-turbo"]
    )
    result = CliRunner().invoke(cli, ["sakura", "refresh"])
    assert result.output == "Added: sakura/a\n"


# Embeddings


def test_embed_batch(no_fetch, monkeypatch):
    calls = []

    class FakeEmbeddings:
        def create(self, **kwargs):
            calls.append(kwargs)
            data = [
                SimpleNamespace(index=i, embedding=[i, 0.5])
                for i in range(len(kwargs["input"]))
            ]
            # Out of order, to check results are sorted by index
            return SimpleNamespace(data=list(reversed(data)))

    class FakeOpenAI:
        def __init__(self, base_url, api_key):
            calls.append({"base_url": base_url, "api_key": api_key})
            self.embeddings = FakeEmbeddings()

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    model = llm.get_embedding_model("sakura/multilingual-e5-large")
    assert model.batch_size == 100
    assert list(model.embed_multi(["a", "b"])) == [[0.0, 0.5], [1.0, 0.5]]
    assert calls[0] == {"base_url": llm_sakura.API_BASE, "api_key": "sakura-test-token"}
    assert calls[1] == {"model": "multilingual-e5-large", "input": ["a", "b"]}


def test_models_command_lists_embedding_models(no_fetch):
    result = CliRunner().invoke(cli, ["sakura", "models"])
    assert "sakura/multilingual-e5-large\n  Embedding model, 1024 dimensions" in result.output
    data = json.loads(CliRunner().invoke(cli, ["sakura", "models", "--json"]).output)
    assert {"id": "multilingual-e5-large", "type": "embedding", "dimensions": 1024} in data
