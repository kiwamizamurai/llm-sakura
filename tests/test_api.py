"""
Tests that talk to the real API, replayed from tests/cassettes.

To record them again:
  PYTEST_SAKURA_AI_ENGINE_TOKEN="$(llm keys get sakura)" uv run pytest --record-mode once
Delete the cassettes first to record from scratch.
"""

import json
import math
import struct
import zlib

import llm
import pytest
from click.testing import CliRunner
from llm.cli import cli
from llm.parts import ReasoningPart

GLM = "sakura/preview/GLM-5.3-Flash"


def solid_png(rgb, size=64):
    "A PNG image of one colour"

    def png_chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    row = b"\x00" + bytes(rgb) * size
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(row * size))
        + png_chunk(b"IEND", b"")
    )


RED = llm.Attachment(type="image/png", content=solid_png((255, 0, 0)))


def reasoning_of(response):
    return "".join(
        part.text
        for message in response.messages()
        for part in message.parts
        if isinstance(part, ReasoningPart)
    )


@pytest.mark.vcr
def test_refresh_lists_chat_and_embedding_models():
    result = CliRunner().invoke(cli, ["sakura", "refresh"])
    assert result.exit_code == 0, result.output
    assert "Added: sakura/preview/GLM-5.3-Flash" in result.output
    assert "Added: sakura/multilingual-e5-large" in result.output
    assert "whisper" not in result.output
    chat_ids = {model.model_id for model in llm.get_models()}
    assert GLM in chat_ids
    assert "sakura/multilingual-e5-large" not in chat_ids


@pytest.mark.vcr
@pytest.mark.parametrize("stream", (True, False))
def test_prompt_with_reasoning(no_fetch, stream):
    model = llm.get_model(GLM)
    response = model.prompt("日本の首都を一言で", stream=stream, max_tokens=1500)
    assert "東京" in response.text()
    assert reasoning_of(response)
    assert response.usage().input > 0
    assert response.usage().output > 0


@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (True, False))
async def test_async_prompt(no_fetch, stream):
    model = llm.get_async_model(GLM)
    response = model.prompt("日本の首都を一言で", stream=stream, max_tokens=1500)
    assert "東京" in await response.text()


@pytest.mark.vcr
def test_reasoning_effort(no_fetch):
    model = llm.get_model("sakura/gpt-oss-120b")
    response = model.prompt("こんにちは", reasoning_effort="low")
    assert response.text()
    assert reasoning_of(response)


@pytest.mark.vcr
@pytest.mark.parametrize("thinking", (True, False))
def test_thinking_switch(no_fetch, thinking):
    model = llm.get_model("sakura/preview/Qwen3.6-35B-A3B")
    response = model.prompt("1+1は？数字だけ", thinking=thinking, max_tokens=1500)
    assert "2" in response.text()
    assert bool(reasoning_of(response)) is thinking


@pytest.mark.vcr
def test_image(no_fetch):
    model = llm.get_model(GLM)
    response = model.prompt(
        "この画像は何色？色の名前だけ答えて", attachments=[RED], max_tokens=1500
    )
    assert "赤" in response.text() or "red" in response.text().lower()


@pytest.mark.vcr
def test_three_images_allowed(no_fetch):
    model = llm.get_model("sakura/preview/Qwen3-VL-30B-A3B-Instruct")
    response = model.prompt("画像は何枚？数字だけ", attachments=[RED] * 3)
    assert "3" in response.text()


@pytest.mark.vcr
def test_tools(no_fetch):
    calls = []

    def llm_time() -> str:
        "Return the current time"
        calls.append(1)
        return "12:00"

    model = llm.get_model(GLM)
    chain = model.chain(
        "いま何時？ツールで確かめて", tools=[llm_time], options={"max_tokens": 1500}
    )
    assert "12" in chain.text()
    assert calls == [1]


@pytest.mark.vcr
def test_schema(no_fetch):
    model = llm.get_model(GLM)
    response = model.prompt(
        "犬を一匹考えて",
        schema=llm.schema_dsl("name, age int"),
        max_tokens=1500,
    )
    data = json.loads(response.text())
    assert isinstance(data["name"], str)
    assert isinstance(data["age"], int)


@pytest.mark.vcr
def test_max_tokens_used_up_by_reasoning(no_fetch):
    model = llm.get_model(GLM)
    with pytest.raises(llm.ModelError, match="max_tokens"):
        model.prompt("こんにちは", max_tokens=5).text()


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


@pytest.mark.vcr
@pytest.mark.parametrize(
    "model_id,dimensions",
    (
        ("sakura/multilingual-e5-large", 1024),
        ("sakura/preview/Qwen3-Embedding-4B-FP16", 2560),
    ),
)
def test_embeddings(no_fetch, model_id, dimensions):
    model = llm.get_embedding_model(model_id)
    dog_ja, dog_en, car = model.embed_multi(["犬", "dog", "自動車のエンジン"])
    assert len(dog_ja) == dimensions
    assert cosine(dog_ja, dog_en) > cosine(dog_ja, car)


def test_ignored_option_is_rejected_by_cli(no_fetch):
    result = CliRunner().invoke(cli, ["-m", GLM, "hi", "-o", "top_p", "0.5"])
    assert result.exit_code != 0
    assert "top_p" in result.output
