"""Model backends: one interface, three implementations.

Every backend has the same two methods:

    backend.chat(model, messages, temperature=..., seed=..., max_tokens=..., ...) -> Reply
    backend.describe(model) -> {"digest": ..., "version": ...}

``messages`` is the whole conversation, a list of ``{"role": "user" or "assistant",
"content": text}``. Nothing is remembered between calls: each call is a fresh conversation.

OllamaBackend    Ollama's native API (``/api/chat``).
OpenAIBackend    Any server with an OpenAI-compatible ``/chat/completions`` endpoint. The base
                 URL and API key come from environment variables named in the config.
MockBackend      For tests only: answers come from a Python function and nothing is sent.

A ``Reply`` keeps the model's thinking text apart from its answer, and gives token
probabilities in one shape whatever the backend:

    reply.logprobs = [{"token": "B", "logprob": -0.1,
                       "top_logprobs": [{"token": "B", "logprob": -0.1}, ...]}, ...]

Fresh evaluation (Ollama)
-------------------------
Ollama keeps the computation for a prompt and reuses it when a later prompt starts the
same way. The reused computation gives slightly different token probabilities from a
computation from scratch: for one prompt here, 0.64 in place of 0.72 for the top letter.
Which queries get reused depends on what was asked before, so results would depend on the
order of the queries.

With ``fresh=True`` the Ollama backend makes every answer a computation from scratch:

1. Before each request it sends a one-token request with an unrelated prompt, so the
   request shares no beginning with the one before it.
2. It checks that Ollama reports no reused tokens for the request, and fails if it does.

Step 2 can only succeed if the server does not bring back older prompts from its second,
larger cache. That cache is switched off by starting the server with the environment
variable ``LLAMA_ARG_CACHE_RAM=0``; the README gives the command. ``describe`` tests this
once per model before a run.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

import requests


class BackendError(RuntimeError):
    """A request failed: the server could not be reached, timed out or returned an error."""


@dataclass
class Reply:
    text: str                              # the answer, without any thinking text
    thinking: str = ""                     # the thinking text, if the model produced any
    logprobs: Optional[list] = None        # one entry per generated token; None if not returned
    finish_reason: Optional[str] = None    # "stop", "length", ... as the backend reports it
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None    # prompt tokens whose computation the server reused
    seconds: float = 0.0                   # time the request took


def _post(url, body, timeout, headers=None):
    """POST JSON and return the decoded JSON reply and the seconds it took."""
    started = time.monotonic()
    try:
        response = requests.post(url, json=body, headers=headers, timeout=timeout)
    except requests.RequestException as err:
        raise BackendError(f"request to {url} failed: {err}") from err
    seconds = time.monotonic() - started
    if response.status_code != 200:
        raise BackendError(f"{url} returned HTTP {response.status_code}: {response.text[:500]}")
    try:
        return response.json(), seconds
    except ValueError as err:
        raise BackendError(f"{url} did not return JSON: {response.text[:200]!r}") from err


def _tidy_logprobs(entries):
    """Keep the token, its log probability and its alternatives; drop everything else."""
    if not entries:
        return None
    return [
        {
            "token": entry["token"],
            "logprob": entry["logprob"],
            "top_logprobs": [{"token": top["token"], "logprob": top["logprob"]}
                             for top in entry.get("top_logprobs") or []],
        }
        for entry in entries
    ]


class OllamaBackend:
    """Ollama's native chat API."""

    name = "ollama"
    RESET_PROMPT = "Say ok."    # the unrelated prompt sent before each request when fresh is on

    def __init__(self, base_url="http://localhost:11434", options=None, timeout=300, fresh=False):
        self.base_url = base_url.rstrip("/")
        self.options = dict(options or {})    # sent with every request, for example {"num_ctx": 8192}
        self.timeout = timeout
        self.fresh = fresh                    # compute every answer from scratch (see the module docstring)

    def chat(self, model, messages, *, temperature, seed, max_tokens, think=None, schema=None,
             top_logprobs=0, extra_body=None):
        if self.fresh:
            reset = {"model": model, "messages": [{"role": "user", "content": self.RESET_PROMPT}], "stream": False,
                     "think": False, "options": {**self.options, "temperature": 0.0, "seed": 0, "num_predict": 1}}
            _post(f"{self.base_url}/api/chat", reset, self.timeout)
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {**self.options, "temperature": temperature, "seed": seed, "num_predict": max_tokens},
        }
        if think is not None:
            body["think"] = think
        if schema is not None:
            body["format"] = schema
        if top_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs
        body.update(extra_body or {})
        data, seconds = _post(f"{self.base_url}/api/chat", body, self.timeout)
        cached = data.get("prompt_eval_cached_count") or 0
        if self.fresh and cached:
            raise BackendError(
                f"Ollama reused {cached} cached prompt tokens, so this answer was not computed from scratch. "
                "Start the server with LLAMA_ARG_CACHE_RAM=0 (see the README), or set backends.ollama.fresh "
                "to false to accept answers that depend on the order of the queries.")
        message = data.get("message") or {}
        return Reply(
            text=message.get("content") or "",
            thinking=message.get("thinking") or "",
            logprobs=_tidy_logprobs(data.get("logprobs")),
            finish_reason=data.get("done_reason"),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            cached_tokens=cached,
            seconds=seconds,
        )

    def describe(self, model):
        """The model's digest and the server's version, to record with every result.

        With ``fresh`` on, this also asks the same question twice and so checks, before a
        run starts, that the server computes a repeated prompt from scratch.
        """
        try:
            tags = requests.get(f"{self.base_url}/api/tags", timeout=30).json()["models"]
            version = requests.get(f"{self.base_url}/api/version", timeout=30).json()["version"]
        except (requests.RequestException, ValueError, KeyError) as err:
            raise BackendError(f"could not reach Ollama at {self.base_url}: {err}") from err
        digests = {entry["name"]: entry["digest"] for entry in tags}
        if model not in digests and f"{model}:latest" not in digests:
            raise BackendError(f"model {model!r} is not installed in Ollama; installed: {sorted(digests)}")
        if self.fresh:
            question = [{"role": "user", "content": "Name one color."}]
            for _ in range(2):    # the second call raises BackendError if the first one was cached
                self.chat(model, question, temperature=0.0, seed=0, max_tokens=1, think=False)
        return {"digest": digests.get(model) or digests[f"{model}:latest"], "version": version}


class OpenAIBackend:
    """Any server that speaks the OpenAI chat completions API."""

    name = "openai"

    def __init__(self, base_url, api_key=None, timeout=300):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.timeout = timeout

    @classmethod
    def from_environment(cls, base_url_env, api_key_env, timeout=300):
        """Build the backend from the two environment variables named in the config."""
        base_url = os.environ.get(base_url_env)
        if not base_url:
            raise BackendError(f"set the environment variable {base_url_env} to the server's base URL, "
                               "for example https://api.openai.com/v1 or http://localhost:11434/v1")
        return cls(base_url, os.environ.get(api_key_env), timeout)

    def chat(self, model, messages, *, temperature, seed, max_tokens, think=None, schema=None,
             top_logprobs=0, extra_body=None):
        if think is not None:
            raise ValueError("think is an Ollama setting; pass the provider's own field in extra_body")
        body = {"model": model, "messages": messages, "temperature": temperature, "seed": seed,
                "max_tokens": max_tokens}
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "strict": True, "schema": schema},
            }
        if top_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs
        body.update(extra_body or {})
        data, seconds = _post(f"{self.base_url}/chat/completions", body, self.timeout, self.headers)
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as err:
            raise BackendError(f"unexpected reply from {self.base_url}: {str(data)[:300]}") from err
        usage = data.get("usage") or {}
        return Reply(
            text=message.get("content") or "",
            thinking=message.get("reasoning") or message.get("reasoning_content") or "",
            logprobs=_tidy_logprobs((choice.get("logprobs") or {}).get("content")),
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            seconds=seconds,
        )

    def describe(self, model):
        """This API has no model digest; the server's own name for the model is in each reply."""
        return {"digest": None, "version": None}


class MockBackend:
    """A stand-in for tests: ``answer(model, messages, **settings)`` supplies every reply.

    ``answer`` returns a Reply, or just the text of one. Every call is kept in ``calls``.
    """

    name = "mock"

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def chat(self, model, messages, **settings):
        self.calls.append({"model": model, "messages": [dict(message) for message in messages], **settings})
        reply = self.answer(model, messages, **settings)
        return reply if isinstance(reply, Reply) else Reply(text=reply)

    def describe(self, model):
        return {"digest": "mock-digest", "version": "mock"}


def make_backends(config):
    """The backends that the config's models need, by name. The mock is never among them."""
    settings = config.settings["backends"]
    timeout = config.run["timeout_seconds"]
    backends = {}
    for name in {model["backend"] for model in config.settings["models"]}:
        if name == "ollama":
            backends[name] = OllamaBackend(settings["ollama"]["base_url"], settings["ollama"]["options"], timeout,
                                           fresh=settings["ollama"]["fresh"])
        elif name == "openai":
            backends[name] = OpenAIBackend.from_environment(
                settings["openai"]["base_url_env"], settings["openai"]["api_key_env"], timeout)
    return backends
