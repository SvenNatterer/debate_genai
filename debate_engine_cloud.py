"""
debate_engine_cloud.py
======================
Debate engine backed by Azure AI Services (Azure OpenAI / Azure AI Foundry).

Required .env entries
---------------------
  CUSTOM_API_BASE_URL  – your Azure endpoint, e.g.
                         https://gpt-course.cognitiveservices.azure.com
  CUSTOM_API_KEY       – your Azure API key
  AZURE_API_VERSION    – optional, default "2024-12-01-preview"

Azure request format
--------------------
  POST {base_url}/openai/deployments/{model}/chat/completions
       ?api-version={api_version}
  Header: api-key: {CUSTOM_API_KEY}

Public interface (identical to original debate_engine.py)
---------------------------------------------------------
  get_client_and_model()
  chat_completion(system_prompt, user_prompt)
  set_active_model(provider, model)
  get_active_model_label()
  build_agents(agent_configs)
  run_debate(topic, rounds, strategies, agent_configs)
  judge_debate(topic, transcript)
  summarize_debate(topic, transcript, judgment)
"""

import json
import os
import random
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

from config import AGENT_LIBRARY, PHILOSOPHER_LIBRARY, SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Runtime model override (set by the UI dropdown)
# ---------------------------------------------------------------------------

_runtime_override: Dict[str, str] = {}


def set_active_model(provider: str, model: str) -> None:
    """Called by the UI to select a model at runtime."""
    _runtime_override["provider"] = provider
    _runtime_override["model"] = model


def get_active_model_label() -> str:
    """Human-readable label for the currently active model."""
    provider = _runtime_override.get("provider", "custom")
    model = _runtime_override.get("model") or os.getenv("CUSTOM_MODEL", "")
    if not model:
        return "No model selected"
    prefix = "Local" if provider == "local" else "Azure"
    return f"{prefix} / {model}"


# ---------------------------------------------------------------------------
# Azure configuration
# ---------------------------------------------------------------------------

_AZURE_API_VERSION = "2024-12-01-preview"


def _get_azure_config(model: str = "") -> Tuple[str, str, str]:
    """
    Returns (endpoint_url, resolved_model, api_key).
    Raises ValueError with a clear message if anything is missing.
    """
    base_url       = os.getenv("CUSTOM_API_BASE_URL", "").rstrip("/")
    api_key        = os.getenv("CUSTOM_API_KEY", "")
    resolved_model = model or _runtime_override.get("model") or os.getenv("CUSTOM_MODEL", "")
    api_version    = os.getenv("AZURE_API_VERSION", _AZURE_API_VERSION)

    if not base_url:
        raise ValueError(
            "CUSTOM_API_BASE_URL is not set. "
            "Add it to your .env file, e.g.: "
            "CUSTOM_API_BASE_URL=https://gpt-course.cognitiveservices.azure.com"
        )
    if not api_key:
        raise ValueError(
            "CUSTOM_API_KEY is not set. "
            "Add your Azure API key to your .env file."
        )
    if not resolved_model:
        raise ValueError(
            "No model selected. Choose one from the dropdowns on the start screen."
        )

    endpoint = (
        f"{base_url}/openai/deployments/{resolved_model}/chat/completions"
        f"?api-version={api_version}"
    )

    return endpoint, resolved_model, api_key


# ---------------------------------------------------------------------------
# Public status helper  (same signature as original get_client_and_model)
# ---------------------------------------------------------------------------

def get_client_and_model() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Connectivity check used by the status bar in app.py.
    Returns (base_url, "ready", None) when Azure credentials are configured,
    or (None, None, "<message>") when not.
    """
    base_url = os.getenv("CUSTOM_API_BASE_URL", "")
    api_key  = os.getenv("CUSTOM_API_KEY", "")
    if not base_url or not api_key:
        return None, None, "Azure not configured — check CUSTOM_API_BASE_URL and CUSTOM_API_KEY in .env"
    return base_url, "ready", None


# ---------------------------------------------------------------------------
# Core chat completion
# ---------------------------------------------------------------------------

def _is_error_response(text: str) -> bool:
    return text.startswith(("[ERROR]", "Azure API", "Configuration error", "No model",
                            "CUSTOM_API_BASE_URL", "CUSTOM_API_KEY",
                            "Ollama"))


def _chat_completion_ollama(system_prompt: str, user_prompt: str, model: str = "") -> str:
    """Send a chat request to a local Ollama instance."""
    base_url       = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    resolved_model = model or _runtime_override.get("model") or os.getenv("OLLAMA_MODEL", "llama3.2:1b")

    payload = {
        "model": resolved_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "stream": False,
    }
    request = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = json.loads(response.read().decode("utf-8"))
            if "error" in raw and raw["error"]:
                return f"Ollama error: {raw['error']}"
            content = raw.get("message", {}).get("content", "")
            return content.strip() if content.strip() else "Ollama returned an empty response."
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8").strip()
        except Exception:
            body = ""
        return f"Ollama HTTP error {exc.code}: {body or exc.reason}"
    except urllib.error.URLError as exc:
        return f"Ollama connection error at {base_url}: {exc.reason}"
    except TimeoutError:
        return f"Ollama request timed out for model '{resolved_model}'."
    except json.JSONDecodeError:
        return "Ollama returned invalid JSON."


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    *,
    provider: str = "",
    model: str = "",
) -> str:
    """
    Sends a chat request to the given provider/model (or falls back to _runtime_override).
    Returns a descriptive error string (never raises) so the UI can surface it.
    """
    resolved_provider = provider or _runtime_override.get("provider", "custom")
    if resolved_provider == "local":
        return _chat_completion_ollama(system_prompt, user_prompt, model=model)

    try:
        endpoint, resolved_model, api_key = _get_azure_config(model=model)
    except ValueError as exc:
        return str(exc)

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.7,
    }

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "api-key":      api_key,   # Azure uses api-key, not Authorization: Bearer
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            raw     = json.loads(response.read().decode("utf-8"))
            choices = raw.get("choices", [])
            if not choices:
                return "Azure API returned no choices in response."
            content = choices[0].get("message", {}).get("content", "")
            return content.strip() if content.strip() else "Azure API returned an empty response."

    except urllib.error.HTTPError as exc:
        try:
            body      = exc.read().decode("utf-8").strip()
            error_msg = json.loads(body).get("error", {}).get("message", body)
        except Exception:
            error_msg = exc.reason
        return f"Azure API HTTP {exc.code}: {error_msg}"

    except urllib.error.URLError as exc:
        return f"Azure API connection error: {exc.reason}"

    except TimeoutError:
        return f"Azure API request timed out for model '{resolved_model}'."

    except json.JSONDecodeError:
        return "Azure API returned invalid JSON."


# ---------------------------------------------------------------------------
# Strategy mapping  (aligned with STRATEGY_OPTIONS in config.py)
# ---------------------------------------------------------------------------

def strategy_to_instructions(strategy: str) -> str:
    mapping = {
        "Direct rebuttal": """
- Directly challenge the opponent's latest point.
- Show why their conclusion does not follow from their premises.
- Stay confrontational but coherent.
""",
        "Evidence-focused": """
- Focus on logic and internal consistency.
- Point out weaknesses or contradictions in the opponent's argument.
- Use precise and structured reasoning.
""",
        "Emotional persuasion": """
- Use emotionally engaging language.
- Emphasize human consequences, hope, fear, or justice.
- Make the argument feel vivid and impactful.
""",
        "Devil's advocate": """
- Challenge assumptions from an unexpected angle.
- Argue positions that complicate the opponent's framing.
- Stay provocative but intellectually honest.
""",
        "Pragmatic trade-off analysis": """
- Acknowledge trade-offs honestly.
- Sound fair and reflective.
- Defend your side without sounding simplistic or extreme.
""",
    }
    return mapping.get(strategy, "- Argue clearly, persuasively, and stay consistent with your side.")


# ---------------------------------------------------------------------------
# DebateAgent
# ---------------------------------------------------------------------------

@dataclass
class DebateAgent:
    key: str
    name: str
    goal: str
    style: str
    philosopher: str = ""
    philosopher_stance: str = ""
    image: str = ""
    side: str = ""
    agent_provider: str = ""
    agent_model: str = ""

    def respond(
        self,
        topic: str,
        transcript: List[Dict[str, str]],
        round_idx: int,
        strategy: str,
        max_words: int = 120,
    ) -> str:
        history = "\n".join(
            f"{turn['speaker']}: {turn['text']}" for turn in transcript[-8:]
        ) or "No prior turns."

        persona_block = ""
        if self.philosopher:
            persona_block = f"""
Philosopher persona: {self.philosopher}
Philosopher stance: {self.philosopher_stance}
Debate side: {self.side}
"""

        prompt = f"""
Agent: {self.name}
Goal: {self.goal}
Style: {self.style}
{persona_block}
Argument strategy: {strategy}

Strategy instructions:
{strategy_to_instructions(strategy)}

Topic:
{topic}

Round:
{round_idx}

Recent transcript:
{history}

Instructions:
- Respond as {self.name}.
- If a philosopher persona is assigned, reflect that philosopher's perspective and tone.
- Stay consistent with your assigned side.
- Address the debate topic directly.
- React to the opponent's most relevant prior point when possible.
- Keep it under {max_words} words.
- Avoid bullet points.
"""
        response = chat_completion(
            SYSTEM_PROMPT, prompt,
            provider=self.agent_provider,
            model=self.agent_model,
        ).strip()
        if _is_error_response(response):
            return f"[ERROR] {response}"
        return response


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def build_agents(agent_configs: List[Dict[str, str]]) -> List[DebateAgent]:
    side_labels = ["For", "Against"]
    side_goals = {
        "For":     "Argue clearly in favor of the topic and defend why it is desirable, useful, or justified.",
        "Against": "Argue clearly against the topic and expose why it is risky, flawed, or unjustified.",
    }
    agents = []
    for idx, cfg in enumerate(agent_configs, start=1):
        philosopher = PHILOSOPHER_LIBRARY[cfg["philosopher_key"]]
        side        = side_labels[idx - 1]
        agents.append(DebateAgent(
            key=f"agent_{idx}",
            name=f"{philosopher['name']} ({side})",
            goal=side_goals[side],
            style=f"{philosopher['style']}; debating from the {side.lower()} side",
            philosopher=philosopher["name"],
            philosopher_stance=philosopher["stance"],
            image=philosopher["image"],
            side=side,
            agent_provider=cfg.get("provider", ""),
            agent_model=cfg.get("model", ""),
        ))
    return agents


# ---------------------------------------------------------------------------
# Debate execution
# ---------------------------------------------------------------------------

def run_debate(
    topic: str,
    rounds: int,
    player_strategies: List[str],
    agent_configs: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    transcript: List[Dict[str, str]] = []
    agents = build_agents(agent_configs)
    for r in range(1, rounds + 1):
        for idx, agent in enumerate(agents):
            transcript.append({
                "speaker": agent.name,
                "text":    agent.respond(topic, transcript, r, player_strategies[idx]),
            })
    return transcript


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def judge_debate(
    topic: str,
    transcript: List[Dict[str, str]],
    *,
    judge_provider: str = "",
    judge_model: str = "",
) -> Dict[str, Any]:
    history = "\n".join(f"{t['speaker']}: {t['text']}" for t in transcript)

    prompt = f"""
Agent: Judge
Goal: {AGENT_LIBRARY['judge']['goal']}
Style: {AGENT_LIBRARY['judge']['style']}

Topic:
{topic}

Transcript:
{history}

Return strict JSON with this schema:
{{
  "winner": "agent name",
  "scores": {{
    "agent name": {{
      "logic": 1-10,
      "relevance": 1-10,
      "rebuttal": 1-10,
      "fairness": 1-10,
      "total": 1-40
    }}
  }},
  "reasoning": "2-4 sentences"
}}
"""
    raw = chat_completion(
        SYSTEM_PROMPT, prompt,
        provider=judge_provider,
        model=judge_model,
    ).strip()

    if _is_error_response(raw):
        speakers = sorted({t["speaker"] for t in transcript})
        return {
            "winner":    "No winner",
            "scores":    {s: {"logic": 0, "relevance": 0, "rebuttal": 0, "fairness": 0, "total": 0} for s in speakers},
            "reasoning": raw,
        }

    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        return json.loads(raw)
    except Exception:
        speakers = sorted({t["speaker"] for t in transcript})
        scores   = {}
        for s in speakers:
            lo, re, rb, fa = (random.randint(6, 9), random.randint(6, 9),
                              random.randint(5, 9), random.randint(6, 9))
            scores[s] = {"logic": lo, "relevance": re, "rebuttal": rb,
                         "fairness": fa, "total": lo + re + rb + fa}
        return {
            "winner":    max(scores, key=lambda k: scores[k]["total"]),
            "scores":    scores,
            "reasoning": "Fallback scoring used because the model did not return valid JSON.",
        }


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------

def summarize_debate(
    topic: str,
    transcript: List[Dict[str, str]],
    judgment: Dict[str, Any],
    *,
    judge_provider: str = "",
    judge_model: str = "",
) -> str:
    history = "\n".join(f"{t['speaker']}: {t['text']}" for t in transcript)

    prompt = f"""
Agent: Summarizer
Goal: {AGENT_LIBRARY['summarizer']['goal']}
Style: {AGENT_LIBRARY['summarizer']['style']}

Topic:
{topic}

Transcript:
{history}

Judgment:
{json.dumps(judgment, indent=2)}

Write a balanced summary in 1-2 paragraphs.
Then add:
Strongest points:
- Pro: ...
- Contra: ...
- Caveat: ...
"""
    summary = chat_completion(
        SYSTEM_PROMPT, prompt,
        provider=judge_provider,
        model=judge_model,
    ).strip()
    if _is_error_response(summary):
        return f"[ERROR] {summary}"
    return summary
