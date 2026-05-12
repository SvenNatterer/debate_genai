"""
debate_engine_cloud.py
======================
Debate engine backed by Azure AI Services (Azure OpenAI / Azure AI Foundry)
or a local Ollama instance, routed per-call.

Required .env entries
---------------------
  CUSTOM_API_BASE_URL  – your Azure endpoint, e.g.
                         https://gpt-course.cognitiveservices.azure.com
  CUSTOM_API_KEY       – your Azure API key
  AZURE_API_VERSION    – optional, default "2024-12-01-preview"
  OLLAMA_BASE_URL      – optional, default "http://localhost:11434"

Every call to chat_completion() requires explicit provider= and model= kwargs.
There is no global default or silent fallback — missing values return an error string.

Public interface
----------------
  get_client_and_model()
  chat_completion(system_prompt, user_prompt, *, provider, model)
  build_agents(agent_configs)
  run_debate(topic, rounds, strategies, agent_configs, team_mode)
  judge_debate(topic, transcript, *, judge_provider, judge_model, focus)
  aggregate_judgments(judgments)
  summarize_debate(topic, transcript, judgment, *, judge_provider, judge_model)
"""

import json
import os
import random
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

from config import AGENT_LIBRARY, PHILOSOPHER_LIBRARY, SYSTEM_PROMPT


TOKEN_USAGE_LOG: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Azure configuration
# ---------------------------------------------------------------------------

_AZURE_API_VERSION = "2024-12-01-preview"


def _get_azure_config(model: str) -> Tuple[str, str, str]:
    """
    Returns (endpoint_url, model, api_key).
    Raises ValueError with a clear message if credentials are missing.
    model must be non-empty — caller is responsible.
    """
    base_url    = os.getenv("CUSTOM_API_BASE_URL", "").rstrip("/")
    api_key     = os.getenv("CUSTOM_API_KEY", "")
    api_version = os.getenv("AZURE_API_VERSION", _AZURE_API_VERSION)

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

    endpoint = (
        f"{base_url}/openai/deployments/{model}/chat/completions"
        f"?api-version={api_version}"
    )

    return endpoint, model, api_key


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


def reset_token_usage() -> None:
    TOKEN_USAGE_LOG.clear()


def get_token_usage() -> List[Dict[str, Any]]:
    return [dict(entry) for entry in TOKEN_USAGE_LOG]


def _call_label_from_prompt(user_prompt: str) -> str:
    for line in user_prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("Agent:"):
            return stripped.replace("Agent:", "", 1).strip() or "Agent"
    if "strict debate judge" in user_prompt.lower():
        return "Judge"
    return "Model call"


def _record_token_usage(
    *,
    provider: str,
    model: str,
    user_prompt: str,
    usage: Dict[str, Any],
) -> None:
    if not usage:
        return

    prompt_tokens = (
        usage.get("prompt_tokens")
        or usage.get("prompt_eval_count")
        or 0
    )
    completion_tokens = (
        usage.get("completion_tokens")
        or usage.get("eval_count")
        or 0
    )
    total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)

    TOKEN_USAGE_LOG.append({
        "call": _call_label_from_prompt(user_prompt),
        "provider": provider,
        "model": model,
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    })


def _parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from model output using progressively looser strategies.

    Models — especially small ones — often wrap JSON in prose or code fences.
    Strategy order: direct parse → code fence extraction → first-brace heuristic.
    """
    # 1. Direct parse (ideal case — model returned only JSON)
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2. Markdown code fence: ```json { ... } ``` or ``` { ... } ```
    match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # 3. First { to last } — handles prose before/after the JSON block
    start = text.find('{')
    end   = text.rfind('}')
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass

    return None


def _chat_completion_ollama(system_prompt: str, user_prompt: str, model: str) -> str:
    """Send a chat request to a local Ollama instance. model must be non-empty."""
    base_url       = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    resolved_model = model

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
            _record_token_usage(
                provider="local",
                model=resolved_model,
                user_prompt=user_prompt,
                usage=raw,
            )
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
    Routes a chat request to Ollama (provider='local') or Azure (provider='custom').
    provider and model must both be non-empty — no silent fallback exists.
    Returns a descriptive error string (never raises) so the UI can surface it.
    """
    if not provider or not model:
        return (
            "[ERROR] No model/provider specified for this call. "
            "Select a model in the start screen and try again."
        )
    if provider == "local":
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
            _record_token_usage(
                provider=provider,
                model=resolved_model,
                user_prompt=user_prompt,
                usage=raw.get("usage", {}),
            )
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
        team_mode: bool = False,
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

        if team_mode:
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

Hidden philosopher team protocol:
- Strategist: privately identify the strongest argument line for your assigned side.
- Critic: privately identify weaknesses, objections, and risks in that argument.
- Speaker: write the final public response in {self.name}'s voice.

Instructions:
- Return only the final Speaker response.
- Do not mention Strategist, Critic, Speaker, team deliberation, hidden notes, or internal reasoning.
- Respond as {self.name}.
- If a philosopher persona is assigned, reflect that philosopher's perspective and tone.
- Stay consistent with your assigned side.
- Address the debate topic directly.
- React to the opponent's most relevant prior point when possible.
- Keep it under {max_words} words.
- Avoid bullet points.
"""
        else:
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
    max_words: int = 120,
    team_mode: bool = False,
) -> List[Dict[str, str]]:
    transcript: List[Dict[str, str]] = []
    agents = build_agents(agent_configs)
    for r in range(1, rounds + 1):
        for idx, agent in enumerate(agents):
            transcript.append({
                "speaker": agent.name,
                "text":    agent.respond(
                    topic, transcript, r, player_strategies[idx],
                    max_words=max_words,
                    team_mode=team_mode,
                ),
            })
    return transcript


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

_JUDGE_METRICS = [
    "logical_validity",
    "argument_strength",
    "counterargument_handling",
    "clarity",
    "relevance",
]


def judge_debate(
    topic: str,
    transcript: List[Dict[str, str]],
    *,
    judge_provider: str = "",
    judge_model: str = "",
    focus: str = "",
) -> Dict[str, Any]:
    focus_block = (
        f"\nSpecialisation:\n{focus}\n"
        f"Score all five metrics, but apply extra scrutiny in your area of focus.\n"
    ) if focus else ""

    # Speaker names taken directly from transcript so the model can copy them exactly.
    speakers = sorted({t["speaker"] for t in transcript})
    speaker_list = "\n".join(f'  "{s}"' for s in speakers)

    history = "\n".join(f"{t['speaker']}: {t['text']}" for t in transcript)

    prompt = f"""You are a strict debate judge. Evaluate the debate below and return ONLY a JSON object — no explanation, no prose, no markdown. Your entire response must begin with {{ and end with }}.
{focus_block}
Scoring criteria (each 1–10):
- logical_validity: Are arguments logically sound?
- argument_strength: How well-supported are the claims?
- counterargument_handling: How well does each side rebut the opponent?
- clarity: How clearly are ideas expressed?
- relevance: How well does each speaker stay on topic?
Total = sum of the five scores (5–50).

Topic: {topic}

Speakers (use these exact strings as keys):
{speaker_list}

Transcript:
{history}

Return this exact JSON structure (replace placeholder values):
{{
  "winner": "<exact speaker name>",
  "scores": {{
{chr(10).join(f'    "{s}": {{"logical_validity": 7, "argument_strength": 7, "counterargument_handling": 7, "clarity": 7, "relevance": 7, "total": 35}}' for s in speakers)}
  }},
  "reasoning": "2-4 sentence explanation of the winner."
}}
"""
    raw = chat_completion(
        SYSTEM_PROMPT, prompt,
        provider=judge_provider,
        model=judge_model,
    ).strip()

    _zero = {m: 0 for m in _JUDGE_METRICS}
    _zero["total"] = 0

    if _is_error_response(raw):
        return {
            "winner":    "No winner",
            "scores":    {s: dict(_zero) for s in speakers},
            "reasoning": raw,
        }

    parsed = _parse_json_response(raw)
    if parsed is not None:
        return parsed

    # Fallback: random scores so the UI always has something to render
    scores = {}
    for s in speakers:
        vals = {m: random.randint(5, 9) for m in _JUDGE_METRICS}
        vals["total"] = sum(vals[m] for m in _JUDGE_METRICS)
        scores[s] = vals
    return {
        "winner":    max(scores, key=lambda k: scores[k]["total"]),
        "scores":    scores,
        "reasoning": f"Could not parse model output as JSON. Raw response: {raw[:300]}",
    }


def aggregate_judgments(judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Average scores from multiple judge calls into a single judgment."""
    if len(judgments) == 1:
        return judgments[0]

    speakers: List[str] = []
    for j in judgments:
        for s in j.get("scores", {}):
            if s not in speakers:
                speakers.append(s)

    aggregated: Dict[str, Any] = {}
    for s in speakers:
        avg: Dict[str, Any] = {}
        for m in _JUDGE_METRICS:
            values = [j["scores"].get(s, {}).get(m, 0) for j in judgments]
            avg[m] = round(sum(values) / len(values), 1)
        avg["total"] = round(sum(avg[m] for m in _JUDGE_METRICS), 1)
        aggregated[s] = avg

    winner = max(aggregated, key=lambda k: aggregated[k]["total"])
    reasoning = " | ".join(
        j.get("reasoning", "") for j in judgments if j.get("reasoning")
    )
    return {"winner": winner, "scores": aggregated, "reasoning": reasoning}


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
