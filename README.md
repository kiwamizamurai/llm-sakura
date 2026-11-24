# llm-sakura

[LLM](https://llm.datasette.io/) plugin for the chat and embedding models hosted by Sakura
Internet's [Sakura AI Engine](https://cloud.sakura.ad.jp/products/ai-engine/) (さくらのAI Engine).

> Status: alpha. This is an unofficial plugin, not affiliated with Sakura Internet.
> 非公式のプラグインです。さくらインターネットとは関係ありません。

## Installation / インストール

Install this plugin in the same environment as LLM:

```bash
llm install llm-sakura
```

## Configuration / 設定

Create an account token (`<UUID>:<secret>`) in the
[control panel](https://secure.sakura.ad.jp/ai/) under "アカウントトークン", then:

```bash
llm keys set sakura
# paste the token
```

Or set the `SAKURA_AI_ENGINE_TOKEN` environment variable.

The free plan allows 3,000 chat requests and 10,000 embedding requests per month. HTTP 429
can be returned before that limit is reached; wait a few minutes and try again. The API does
not report how many requests are left, so check the control panel for that.

## Usage / 使い方

```bash
llm models -q sakura
llm -m sakura/preview/GLM-5.3-Flash 'ペリカンの名前を5つ考えて'
llm chat -m sakura/gpt-oss-120b
```

Models are listed from `/v1/models` and the list is cached for an hour. Before a key is set,
the models described in the
[manual](https://manual.sakura.ad.jp/cloud/ai-engine/chat-models.html) are listed instead.

```bash
llm sakura refresh        # fetch the model list again
llm sakura models         # show what each model supports
llm sakura models --json
```

Model names starting with `preview/` can change or be withdrawn without notice. Run
`llm sakura refresh` if a model stops working.

### Reasoning / 思考の過程

Reasoning is streamed before the answer. Hide it with `-R` / `--hide-reasoning`.

```bash
llm -m sakura/gpt-oss-120b 'Why is the sky blue?' -o reasoning_effort low
llm -m sakura/preview/Qwen3.6-35B-A3B 'Hi' -o thinking 0   # switch reasoning off
```

`reasoning_effort` only accepts the values each model supports (see `llm sakura models`).
`preview/Kimi-K2.7-Code` does not accept it, because switching its reasoning off mixes the
reasoning into the answer.

Reasoning tokens count towards `max_tokens`. If `max_tokens` runs out before the model starts
its answer, the API returns an empty answer; this plugin reports it as an error instead.

### Images, tools and schemas / 画像・ツール・スキーマ

```bash
llm -m sakura/preview/GLM-5.3-Flash 'Describe this image' -a image.jpg
llm -m sakura/preview/GLM-5.3-Flash 'What time is it?' -T llm_time --td
llm -m sakura/preview/GLM-5.3-Flash 'Invent a dog' --schema 'name, age int'
```

`preview/Qwen3-VL-30B-A3B-Instruct` accepts up to 3 images per request. `gpt-oss-120b` ignores
images, so this plugin refuses to send them.

### Embeddings / 埋め込み

```bash
llm embed -m sakura/multilingual-e5-large -c 'こんにちは'
llm embed-multi docs --files notes '*.md' -m sakura/preview/Qwen3-Embedding-4B-FP16
llm similar docs -c 'query: 犬の散歩'
```

| Model | Dimensions |
|---|---|
| `sakura/multilingual-e5-large` | 1024 |
| `sakura/preview/Qwen3-Embedding-4B-FP16` | 2560 |

`multilingual-e5-large` works best with `query: ` and `passage: ` prefixes on the text.
The API ignores a requested number of dimensions.

### Options

| Option | Description |
|---|---|
| `-o temperature 0.7` | Sampling temperature, 0.0–2.0 |
| `-o max_tokens 2000` | Maximum tokens to generate, including reasoning |
| `-o json_object 1` | Return a JSON object (the prompt must mention JSON) |
| `-o reasoning_effort low` | Models that support it |
| `-o thinking 0` | Qwen3.6 and Gemma 4: switch reasoning on or off |

Parameters such as `top_p`, `stop` and `seed` are silently ignored by the API, so they are
not offered.

## Development / 開発

```bash
cd llm-sakura
uv run pytest
uv run llm --help
```

Tests that use the API are replayed from `tests/cassettes`, so no token is needed.
To record them again against the real API:

```bash
PYTEST_SAKURA_AI_ENGINE_TOKEN="$(llm keys get sakura)" uv run pytest --record-mode once
```
