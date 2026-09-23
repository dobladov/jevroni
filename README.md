# Jev in ~25 lines of JavaScript, for LM Studio

A JavaScript port of NobodyWho's [Jev in 25 lines of Python](https://www.nobodywho.ai/posts/jev-in-25-lines/):
a "decision model" that takes a prompt with a few choices and returns a probability for each choice,
read straight from the model's next-token distribution. No sampling, no parsing, one forward pass.

Instead of `llama-cpp-python` it talks to [LM Studio](https://lmstudio.ai)'s local OpenAI-compatible
server, so it works with whatever model you have loaded there. Zero dependencies (Node's built-in `fetch`).

## Requirements

- Node.js 18 or newer
- LM Studio with the local server running (Developer tab, or `lms server start`) on `http://localhost:1234`
- A chat model loaded in LM Studio

## Usage

```sh
# The article's example: classify an email as Legitimate / Spam / Phishing
node index.js

# Your own prompt and choices (labels A, B, C, ... are added automatically)
node index.js "Ticket: 'App crashes on launch since the update'" "Billing" "Bug" "Feature request" "Other"
```

Options (flags or environment variables):

| Flag               | Env var        | Default                                          | Meaning                                                                                     |
| ------------------ | -------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| `--model <id>`     | `JEV_MODEL`    | first model loaded in LM Studio                  | Model id as shown by `lms ls` or `GET /v1/models`                                           |
| `--system <text>`  |                | `Choose one option. Reply with only the letter.` | System prompt                                                                               |
| `--prefill <text>` | `JEV_PREFILL`  | auto                                             | Text the assistant turn starts with (see thinking models below). `--prefill ""` disables it |
| `--top <n>`        |                | `10`                                             | `top_logprobs` requested from LM Studio                                                     |
|                    | `LMSTUDIO_URL` | `http://localhost:1234`                          | LM Studio server base URL                                                                   |

As a module:

```js
import { jev } from "./index.js";

const { choices, probabilities } = await jev({
  prompt: "Review: the battery died after two days and support never replied.",
  choices: ["Positive", "Neutral", "Negative"],
  // model: "qwen2.5-coder-1.5b-instruct-mlx",  // optional, defaults to the first loaded model
});
// probabilities -> [0.025, 0.049, 0.925]
```

The result also carries `winner` (the most probable choice), `rawLogprobs` (full-vocabulary log-probabilities
from LM Studio), `logprobs` (renormalised over the choices), `labels`, `sampled` (the token the model would have
produced) and `prefill`.

## Example output

```
$ node index.js --model qwen2.5-coder-1.5b-instruct-mlx
Model: qwen2.5-coder-1.5b-instruct-mlx
Raw logprobs: { Legitimate: -2.805, Spam: -2.633, Phishing: -0.219 }
Log probabilities: { Legitimate: -2.738, Spam: -2.567, Phishing: -0.153 }
Probabilities: { Legitimate: 0.065, Spam: 0.077, Phishing: 0.859 }
Winner: Phishing (85.9%)

$ node index.js --model prism-ml/bonsai-27b
Model: prism-ml/bonsai-27b (prefill "<think>\n\n</think>\n\n")
Raw logprobs: { Legitimate: -4.938, Spam: -3.466, Phishing: -0.194 }
Log probabilities: { Legitimate: -4.789, Spam: -3.318, Phishing: -0.046 }
Probabilities: { Legitimate: 0.008, Spam: 0.036, Phishing: 0.955 }
Winner: Phishing (95.5%)
```

## How it works

1. Build the prompt: a system message, then the user prompt followed by the lettered options
   (`A. Legitimate`, `B. Spam`, ...). LM Studio applies the model's chat template.
2. Request one token from `/v1/chat/completions` with `logprobs: true` and `top_logprobs: 10`.
3. Look up the log-probability of each label token (`A`, `B`, `C`, ...) in the returned top tokens.
4. Renormalise over just those labels (log-softmax) and exponentiate to get probabilities.

The article reads raw logits and softmaxes over the label logits. LM Studio only exposes log-probabilities
over the whole vocabulary, but softmax is shift-invariant, so renormalising the label log-probabilities gives
exactly the same numbers you would get from the logits.

## LM Studio specifics

These were all verified against LM Studio 0.4.25 (MLX runtime) and drive the small deviations from the article:

- **Only the chat endpoint returns logprobs.** `/v1/completions`, the native `/api/v0/completions` and the
  `@lmstudio/sdk` package (whose `logProbs` option is marked as not yet supported) all return none.
- **`top_logprobs` is capped at 10** by the MLX runtime (the API validator accepts up to 20). A label that is not
  among the top tokens gets probability 0. The default system prompt adds "Reply with only the letter." to the
  article's "Choose one option." so the letters reliably land in the top 10; pass `--system "Choose one option."`
  for the original.
- **`max_tokens: 1` returns zero tokens** and no logprobs, so the script asks for 2 and reads the first position.
- **Thinking models** (Qwen3, Gemma, ...) start with a reasoning block. LM Studio moves those tokens into
  `reasoning_content` and returns no logprobs for them. The script detects this and retries once with the
  article's trick: an assistant prefill of `<think>\n\n</think>\n\n`, which LM Studio honours by continuing the
  trailing assistant message. The prefill hurts non-thinking models, so it is only applied when needed.
- **A model must be named** when several are loaded, so the default is the first loaded model
  reported by `/api/v0/models`.
- Tested with `qwen2.5-coder-1.5b-instruct-mlx` and `prism-ml/bonsai-27b`. GGUF models on the llama.cpp
  runtime were not tested; the request shape is the same, but the `top_logprobs` cap may differ.
