import json
import re
import time
from dataclasses import asdict, dataclass
from enum import Enum
from functools import cache

import click
import httpx2
import llm
from llm.default_plugins.openai_models import (
    AsyncChat,
    Chat,
    # Private helpers, imported so tool calls are streamed exactly the way
    # LLM's own OpenAI plugin streams them. Tests will catch it if they move.
    _streamed_tool_call_index,
    _tool_call_arguments,
    combine_chunks,
    redact_data,
)
from llm.parts import StreamEvent
from llm.utils import remove_dict_none_values
from pydantic import Field, create_model

API_BASE = "https://api.ai.sakura.ad.jp/v1"
MODELS_URL = API_BASE + "/models"
CACHE_TIMEOUT = 3600
KEY_ALIAS = "sakura"
KEY_ENV_VAR = "SAKURA_AI_ENGINE_TOKEN"


@dataclass(frozen=True)
class ModelSpec:
    context_length: int | None = None
    vision: bool = True
    tools: bool = True
    reasoning: bool = True
    # Values accepted by reasoning_effort: a tuple, () if the model must not
    # be sent reasoning_effort at all, or None if the manual does not say.
    efforts: tuple[str, ...] | None = None
    # Whether chat_template_kwargs.enable_thinking switches reasoning on/off
    thinking_switch: bool = False
    max_images: int | None = None
    # False for models that /v1/models lists but the manual does not describe
    documented: bool = True


# From https://manual.sakura.ad.jp/cloud/ai-engine/chat-models.html
# (updated 2026-11-18). Models not listed here are assumed to support
# everything, the same approach llm-glm takes for Z.ai.
KNOWN_MODELS = {
    "gpt-oss-120b": ModelSpec(
        context_length=131072,
        # Images are silently ignored, so LLM should refuse them up front
        vision=False,
        efforts=("low", "medium", "high"),
    ),
    "preview/gemma-4-31B-it": ModelSpec(
        context_length=262144,
        efforts=("low", "medium", "high"),
        thinking_switch=True,
    ),
    "llm-jp-3.1-8x13b-instruct4": ModelSpec(
        context_length=4096, vision=False, tools=False, reasoning=False
    ),
    "preview/Phi-4-mini-instruct-cpu": ModelSpec(
        context_length=4096, vision=False, tools=False, reasoning=False
    ),
    "preview/Qwen3-0.6B-cpu": ModelSpec(
        context_length=40960, vision=False, tools=False, reasoning=False
    ),
    "preview/Qwen3-VL-30B-A3B-Instruct": ModelSpec(
        context_length=262144, tools=False, reasoning=False, max_images=3
    ),
    "preview/Qwen3.6-35B-A3B": ModelSpec(
        context_length=262144,
        efforts=("none", "low", "medium", "high"),
        thinking_switch=True,
    ),
    # Reasoning cannot be turned off, and trying to mixes it into the answer
    "preview/Kimi-K2.7-Code": ModelSpec(context_length=262144, efforts=()),
    "preview/GLM-5.3-Flash": ModelSpec(context_length=1048575),
    "preview/DeepSeek-V4.1-Flash-1017": ModelSpec(context_length=1048575),
}

UNDOCUMENTED = ModelSpec(documented=False)

# Embedding models and their dimensions, checked against the API on
# 2026-11-24. The API ignores the "dimensions" parameter.
KNOWN_EMBEDDING_MODELS = {
    "multilingual-e5-large": 1024,
    "preview/Qwen3-Embedding-4B-FP16": 2560,
}

# /v1/models lists every model, including embedding and transcription models
EMBEDDING_PATTERN = re.compile(r"embed|(^|[-_/])e5([-_]|$)", re.IGNORECASE)
NON_CHAT_PATTERN = re.compile(
    r"whisper|tts|speech|transcri|voicevox|" + EMBEDDING_PATTERN.pattern,
    re.IGNORECASE,
)


def get_key(explicit=None):
    return llm.get_key(explicit or "", KEY_ALIAS, KEY_ENV_VAR)


def cache_path():
    return llm.user_dir() / "sakura_models.json"


def fetch_model_ids(key):
    "Fetch the IDs of every model the key can use"
    response = httpx2.get(
        MODELS_URL,
        headers={"Authorization": f"Bearer {key}"},
        timeout=30,
        follow_redirects=True,
    )
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict):
        data = data["data"]
    return sorted(item["id"] if isinstance(item, dict) else item for item in data)


def get_model_ids(key=None, skip_cache=False):
    """
    The model IDs from /v1/models, cached for up to an hour.

    Falls back to a stale cached copy if the API cannot be reached, and to
    the models described in the manual if there is no cached copy either.
    """
    path = cache_path()
    cached = None
    if path.is_file():
        try:
            cached = json.loads(path.read_text())
        except ValueError:
            cached = None
        if (
            cached is not None
            and not skip_cache
            and time.time() - path.stat().st_mtime < CACHE_TIMEOUT
        ):
            return cached
    fallback = cached or sorted({*KNOWN_MODELS, *KNOWN_EMBEDDING_MODELS})
    key = key or get_key()
    if not key:
        return fallback
    try:
        ids = fetch_model_ids(key)
    except (httpx2.HTTPError, KeyError, TypeError, ValueError):
        if skip_cache:
            raise
        return fallback
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ids, indent=2))
    return ids


def chat_model_ids(ids):
    return [id_ for id_ in ids if not NON_CHAT_PATTERN.search(id_)]


def embedding_model_ids(ids):
    return [id_ for id_ in ids if EMBEDDING_PATTERN.search(id_)]


def usable(ids):
    "The IDs this plugin registers, as chat or embedding models"
    return chat_model_ids(ids) + embedding_model_ids(ids)


def spec_for(model_name):
    return KNOWN_MODELS.get(model_name, UNDOCUMENTED)


class SakuraOptions(llm.Options):
    # Only the parameters the API documents. Others, such as top_p, stop and
    # seed, are silently ignored by Sakura AI Engine, so they are not offered.
    temperature: float | None = Field(
        description=(
            "Sampling temperature, between 0.0 and 2.0. Higher values make the "
            "output more random, lower values make it more focused."
        ),
        ge=0,
        le=2,
        default=None,
    )
    max_tokens: int | None = Field(
        description=(
            "Maximum number of tokens to generate, including reasoning tokens. "
            "Too low a value can use it all up on reasoning, leaving no answer."
        ),
        ge=1,
        default=None,
    )
    json_object: bool | None = Field(
        description="Output a valid JSON object {...}. Prompt must mention JSON.",
        default=None,
    )


@cache
def build_options(spec):
    fields = {}
    if spec.reasoning and spec.efforts != ():
        efforts = spec.efforts or ("low", "medium", "high")
        enum = Enum("ReasoningEffort", {e: e for e in efforts}, type=str)
        description = "Reasoning effort, one of: {}".format(", ".join(efforts))
        if spec.efforts is None:
            description += ". The manual does not say which values this model accepts"
        fields["reasoning_effort"] = (
            enum | None,
            Field(description=description, default=None),
        )
    if spec.thinking_switch:
        fields["thinking"] = (
            bool | None,
            Field(
                description="Turn reasoning on or off (chat_template_kwargs.enable_thinking)",
                default=None,
            ),
        )
    return create_model("Options", __base__=SakuraOptions, **fields)


def _reasoning_text(obj):
    # Sakura AI Engine uses "reasoning", other vLLM versions "reasoning_content"
    return getattr(obj, "reasoning", None) or getattr(obj, "reasoning_content", None)


class _Collector:
    "Turns Chat Completions output into LLM stream events, keeping what it saw"

    def __init__(self):
        self.chunks = []
        self.tool_calls = {}
        self.usage = None
        self.finish_reason = None
        self.reasoning = ""
        self.has_text = False
        self.response_json = None

    def add_chunk(self, chunk):
        self.chunks.append(chunk)
        if chunk.usage:
            self.usage = chunk.usage.model_dump()
        if not chunk.choices:
            return
        choice = chunk.choices[0]
        if choice.finish_reason:
            self.finish_reason = choice.finish_reason
        delta = choice.delta
        if delta is None:
            return
        reasoning = _reasoning_text(delta)
        if reasoning:
            self.reasoning += reasoning
            yield StreamEvent(type="reasoning", chunk=reasoning)
        for tool_call in delta.tool_calls or []:
            if tool_call.function.arguments is None:
                tool_call.function.arguments = ""
            idx = _streamed_tool_call_index(self.tool_calls, tool_call)
            if idx not in self.tool_calls:
                self.tool_calls[idx] = tool_call
                yield StreamEvent(
                    type="tool_call_name",
                    chunk=tool_call.function.name or "",
                    tool_call_id=tool_call.id,
                )
            else:
                self.tool_calls[idx].function.arguments += tool_call.function.arguments
            if tool_call.function.arguments:
                yield StreamEvent(
                    type="tool_call_args",
                    chunk=tool_call.function.arguments,
                    tool_call_id=self.tool_calls[idx].id,
                )
        if delta.content:
            self.has_text = True
            yield StreamEvent(type="text", chunk=delta.content)

    def add_completion(self, completion):
        if completion.usage:
            self.usage = completion.usage.model_dump()
        self.response_json = remove_dict_none_values(completion.model_dump())
        choice = completion.choices[0]
        self.finish_reason = choice.finish_reason
        message = choice.message
        reasoning = _reasoning_text(message)
        if reasoning:
            self.reasoning = reasoning
            yield StreamEvent(type="reasoning", chunk=reasoning)
        for idx, tool_call in enumerate(message.tool_calls or []):
            self.tool_calls[idx] = tool_call
            yield StreamEvent(
                type="tool_call_name",
                chunk=tool_call.function.name or "",
                tool_call_id=tool_call.id,
            )
            yield StreamEvent(
                type="tool_call_args",
                chunk=tool_call.function.arguments or "",
                tool_call_id=tool_call.id,
            )
        if message.content:
            self.has_text = True
            yield StreamEvent(type="text", chunk=message.content)

    def finish(self, model, response, messages):
        if self.response_json is None:
            combined = combine_chunks(self.chunks)
            if self.reasoning:
                combined["reasoning"] = self.reasoning
            self.response_json = remove_dict_none_values(combined)
        response.response_json = self.response_json
        for value in self.tool_calls.values():
            response.add_tool_call(
                llm.ToolCall(
                    tool_call_id=value.id,
                    name=value.function.name,
                    arguments=_tool_call_arguments(value.function.arguments),
                )
            )
        model.set_usage(response, self.usage)
        response._prompt_json = redact_data({"messages": messages})
        if self.finish_reason == "length" and not self.has_text and not self.tool_calls:
            # The API returns 200 with an empty answer in this case
            raise llm.ModelError(
                "Reached max_tokens before any answer was written. Reasoning "
                "tokens count towards max_tokens - try a larger -o max_tokens"
            )


class _SharedSakura:
    needs_key = KEY_ALIAS
    key_env_var = KEY_ENV_VAR

    def __init__(self, model_name):
        spec = spec_for(model_name)
        super().__init__(
            model_id="sakura/{}".format(model_name),
            model_name=model_name,
            api_base=API_BASE,
            vision=spec.vision,
            supports_tools=spec.tools,
            # response_format is documented for every model, but whether each
            # one follows a schema is untested
            supports_schema=True,
        )
        self.spec = spec
        self.Options = build_options(spec)

    def __str__(self):
        return "Sakura AI Engine: {}".format(self.model_id)

    def build_kwargs(self, prompt, stream):
        kwargs = super().build_kwargs(prompt, stream)
        effort = kwargs.get("reasoning_effort")
        if isinstance(effort, Enum):
            kwargs["reasoning_effort"] = effort.value
        thinking = kwargs.pop("thinking", None)
        if thinking is not None:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": thinking}
            }
        return kwargs

    def _prepare(self, prompt, conversation):
        messages = self.build_messages(prompt, conversation)
        if self.spec.max_images is not None:
            images = sum(
                1
                for message in messages
                if isinstance(message.get("content"), list)
                for part in message["content"]
                if part.get("type") == "image_url"
            )
            if images > self.spec.max_images:
                raise llm.ModelError(
                    "{} accepts at most {} images per request, got {}".format(
                        self.model_name, self.spec.max_images, images
                    )
                )
        return messages


class SakuraChat(_SharedSakura, Chat):
    def execute(self, prompt, stream, response, conversation=None, key=None):
        messages = self._prepare(prompt, conversation)
        kwargs = self.build_kwargs(prompt, stream)
        client = self.get_client(key)
        collector = _Collector()
        completion = client.chat.completions.create(
            model=self.model_name, messages=messages, stream=stream, **kwargs
        )
        if stream:
            for chunk in completion:
                yield from collector.add_chunk(chunk)
        else:
            yield from collector.add_completion(completion)
        collector.finish(self, response, messages)


class SakuraAsyncChat(_SharedSakura, AsyncChat):
    async def execute(self, prompt, stream, response, conversation=None, key=None):
        messages = self._prepare(prompt, conversation)
        kwargs = self.build_kwargs(prompt, stream)
        client = self.get_client(key, async_=True)
        collector = _Collector()
        completion = await client.chat.completions.create(
            model=self.model_name, messages=messages, stream=stream, **kwargs
        )
        if stream:
            async for chunk in completion:
                for event in collector.add_chunk(chunk):
                    yield event
        else:
            for event in collector.add_completion(completion):
                yield event
        collector.finish(self, response, messages)


class SakuraEmbeddingModel(llm.EmbeddingModel):
    needs_key = KEY_ALIAS
    key_env_var = KEY_ENV_VAR
    # Tested with 100 inputs in one request
    batch_size = 100

    def __init__(self, model_name):
        self.model_id = "sakura/{}".format(model_name)
        self.model_name = model_name
        self.dimensions = KNOWN_EMBEDDING_MODELS.get(model_name)

    def __str__(self):
        return "Sakura AI Engine: {}".format(self.model_id)

    def embed_batch(self, items, *, key=None):
        import openai  # deferred, like LLM does: it slows down CLI startup

        client = openai.OpenAI(base_url=API_BASE, api_key=key)
        results = client.embeddings.create(model=self.model_name, input=list(items))
        for result in sorted(results.data, key=lambda result: result.index):
            yield [float(value) for value in result.embedding]


@llm.hookimpl
def register_models(register):
    for model_name in chat_model_ids(get_model_ids()):
        register(SakuraChat(model_name), SakuraAsyncChat(model_name))


@llm.hookimpl
def register_embedding_models(register):
    for model_name in embedding_model_ids(get_model_ids()):
        register(SakuraEmbeddingModel(model_name))


def _describe(spec):
    if not spec.documented:
        return ["  Not described in the manual - assuming every feature works"]
    lines = []
    if spec.context_length:
        lines.append("  Context length: {:,} tokens".format(spec.context_length))
    features = [
        name
        for name, enabled in (
            ("vision", spec.vision),
            ("tools", spec.tools),
            ("reasoning", spec.reasoning),
        )
        if enabled
    ]
    lines.append("  Features: {}".format(", ".join(features) or "text only"))
    if spec.max_images:
        lines.append("  Images per request: up to {}".format(spec.max_images))
    if spec.reasoning:
        if spec.efforts:
            lines.append("  Reasoning effort: {}".format(", ".join(spec.efforts)))
        elif spec.efforts == ():
            lines.append("  Reasoning effort: cannot be changed")
        if spec.thinking_switch:
            lines.append("  Reasoning can be switched with -o thinking 0/1")
    return lines


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def sakura():
        "Commands relating to the llm-sakura plugin"

    def require_key(key):
        key = get_key(key)
        if not key:
            raise click.ClickException(
                "No key found - set one with 'llm keys set {}' or the {} "
                "environment variable".format(KEY_ALIAS, KEY_ENV_VAR)
            )
        return key

    @sakura.command()
    @click.option("json_", "--json", is_flag=True, help="Output as JSON")
    def models(json_):
        "List Sakura AI Engine models (from the cache, or the manual)"
        ids = get_model_ids()
        chat_ids, embedding_ids = chat_model_ids(ids), embedding_model_ids(ids)
        if json_:
            data = [
                dict(id=id_, type="chat", **asdict(spec_for(id_))) for id_ in chat_ids
            ] + [
                dict(
                    id=id_,
                    type="embedding",
                    dimensions=KNOWN_EMBEDDING_MODELS.get(id_),
                )
                for id_ in embedding_ids
            ]
            click.echo(json.dumps(data, indent=2))
            return
        for id_ in chat_ids:
            click.echo("sakura/{}".format(id_))
            for line in _describe(spec_for(id_)):
                click.echo(line)
        for id_ in embedding_ids:
            click.echo("sakura/{}".format(id_))
            dimensions = KNOWN_EMBEDDING_MODELS.get(id_)
            click.echo(
                "  Embedding model"
                + (", {} dimensions".format(dimensions) if dimensions else "")
            )

    @sakura.command()
    @click.option("--key", help="Sakura AI Engine account token")
    def refresh(key):
        "Refresh the cached list of models from /v1/models"
        path = cache_path()
        before = set()
        if path.is_file():
            try:
                before = set(usable(json.loads(path.read_text())))
            except ValueError:
                pass
        try:
            ids = get_model_ids(key=require_key(key), skip_cache=True)
        except (httpx2.HTTPError, KeyError, TypeError, ValueError) as ex:
            raise click.ClickException("Could not fetch models: {}".format(ex))
        after = set(usable(ids))
        for id_ in sorted(after - before):
            click.echo("Added: sakura/{}".format(id_))
        for id_ in sorted(before - after):
            click.echo("Removed: sakura/{}".format(id_))
        if after == before:
            click.echo("No changes, {} models".format(len(after)))
