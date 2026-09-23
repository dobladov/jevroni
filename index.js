#!/usr/bin/env node
// Jev in (about) 25 lines of JavaScript, for LM Studio.
// Port of https://www.nobodywho.ai/posts/jev-in-25-lines/ — a "decision model" that takes a prompt
// with choices and outputs calibrated probabilities. Zero dependencies: it talks to LM Studio's
// OpenAI-compatible /v1/chat/completions endpoint (logprobs: true) using Node's built-in fetch.
import { fileURLToPath } from "node:url";

const logaddexp = (a, b) =>
  a === -Infinity
    ? b
    : b === -Infinity
      ? a
      : Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)));
const logsumexp = (xs) => xs.reduce(logaddexp, -Infinity);
const THINK_PREFILL = "<think>\n\n</think>\n\n"; // the article's trick to skip thinking on Qwen3-style models
const DEFAULT_URL = process.env.LMSTUDIO_URL ?? "http://localhost:1234";

/** Ask LM Studio to choose between `choices` for `prompt`; returns probabilities over the choices. */
export async function jev({
  prompt,
  choices,
  model,
  system = "Choose one option. Reply with only the letter.",
  prefill,
  topLogprobs = 10,
  baseUrl = DEFAULT_URL,
}) {
  const labels = choices.map((_, i) => String.fromCharCode(65 + i)); // "A", "B", "C", ...
  const options = choices.map((choice, i) => `${labels[i]}. ${choice}`).join("\n");
  model ??= await pickModel(baseUrl);
  const ask = async (assistantPrefill) => {
    const messages = [
      { role: "system", content: system },
      { role: "user", content: `${prompt}\n\n${options}` },
    ];
    if (assistantPrefill) messages.push({ role: "assistant", content: assistantPrefill }); // LM Studio continues a trailing assistant turn
    const res = await fetch(`${baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // max_tokens: 2 because LM Studio returns zero tokens (and no logprobs) for max_tokens: 1; we only read position 0.
      body: JSON.stringify({
        model,
        messages,
        max_tokens: 2,
        temperature: 0,
        logprobs: true,
        top_logprobs: topLogprobs,
      }),
    });
    const data = await res.json();
    if (!res.ok || data.error)
      throw new Error(`LM Studio: ${data.error?.message ?? data.error ?? res.statusText}`);
    return data;
  };
  let data = await ask(prefill ?? "");
  let first = data.choices[0].logprobs?.content?.[0];
  // Thinking models start with a reasoning block; LM Studio moves those tokens to reasoning_content and returns no
  // logprobs for them. Unless the caller chose a prefill, retry once with an empty think block, like the article does.
  if (!first && prefill === undefined && data.choices[0].message?.reasoning_content) {
    data = await ask((prefill = THINK_PREFILL));
    first = data.choices[0].logprobs?.content?.[0];
  }
  if (!first) {
    const reasoning = data.choices[0].message?.reasoning_content;
    throw new Error(
      reasoning
        ? `${data.model} started a reasoning block (${JSON.stringify(reasoning)}) and LM Studio returns no logprobs for reasoning tokens. Pass prefill ${JSON.stringify(THINK_PREFILL)} to skip thinking.`
        : `LM Studio returned no logprobs for ${data.model}. Update LM Studio to a version that supports logprobs.`,
    );
  }
  // Massage the logprobs into probabilities. LM Studio gives full-vocabulary log-probabilities instead of raw
  // logits, but softmax is shift-invariant, so renormalising over the label tokens gives the same result.
  const top = new Map();
  for (const { token, logprob } of first.top_logprobs ?? [])
    top.set(token.trim(), logaddexp(top.get(token.trim()) ?? -Infinity, logprob)); // merge "C" and " C"
  const rawLogprobs = labels.map((label) => top.get(label) ?? -Infinity);
  if (rawLogprobs.every((x) => x === -Infinity)) {
    throw new Error(
      `None of the labels ${labels.join("/")} are among the top-${topLogprobs} next tokens (${[...top.keys()].map((t) => JSON.stringify(t)).join(", ")}); ` +
        `the model sampled ${JSON.stringify(first.token)}. Try a stricter --system prompt or a --prefill.`,
    );
  }
  const logprobs = rawLogprobs.map((x) => x - logsumexp(rawLogprobs));
  const probabilities = logprobs.map(Math.exp);
  const winner = choices[probabilities.indexOf(Math.max(...probabilities))];
  return {
    model: data.model,
    labels,
    choices,
    rawLogprobs,
    logprobs,
    probabilities,
    winner,
    sampled: first.token,
    prefill: prefill || "",
  };
}

/** Default to the first model already loaded in LM Studio (the server refuses requests without a model when several are loaded). */
async function pickModel(baseUrl) {
  const { data } = await (await fetch(`${baseUrl}/api/v0/models`)).json();
  const loaded = data.find((m) => m.state === "loaded" && (m.type === "llm" || m.type === "vlm"));
  if (!loaded)
    throw new Error(
      "No model is loaded in LM Studio. Load one, or pass --model <id> / JEV_MODEL=<id>.",
    );
  return loaded.id;
}

// CLI: node index.js [--model <id>] [--system <text>] [--prefill <text>] [--top <n>] [<prompt> <choice> <choice> ...]
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const flags = { model: process.env.JEV_MODEL, prefill: process.env.JEV_PREFILL };
  const positional = [];
  for (let i = 2; i < process.argv.length; i++) {
    if (process.argv[i].startsWith("--")) flags[process.argv[i].slice(2)] = process.argv[++i];
    else positional.push(process.argv[i]);
  }
  if (flags.prefill != null) flags.prefill = flags.prefill.replace(/\\n/g, "\n"); // lets a shell pass --prefill '<think>\n\n</think>\n\n'
  if (flags.top) flags.topLogprobs = Number(flags.top);
  const [
    prompt = "Email: Payroll asks for your password on a non-company sign-in page.",
    ...choices
  ] = positional;
  try {
    const result = await jev({
      prompt,
      choices: choices.length ? choices : ["Legitimate", "Spam", "Phishing"],
      ...flags,
    });
    const round = (xs) =>
      Object.fromEntries(result.choices.map((c, i) => [c, Number(xs[i].toFixed(3))]));
    console.log(
      `Model: ${result.model}${result.prefill ? ` (prefill ${JSON.stringify(result.prefill)})` : ""}`,
    );
    console.log("Raw logprobs:", round(result.rawLogprobs));
    console.log("Log probabilities:", round(result.logprobs));
    console.log("Probabilities:", round(result.probabilities));
    console.log(`Winner: ${result.winner} (${(Math.max(...result.probabilities) * 100).toFixed(1)}%)`);
  } catch (err) {
    console.error(`Error: ${err.message}${err.cause ? ` (${err.cause.message})` : ""}`);
    if (err.cause)
      console.error(
        `Is the LM Studio server running at ${DEFAULT_URL}? Start it from the Developer tab or with: lms server start`,
      );
    process.exit(1);
  }
}
