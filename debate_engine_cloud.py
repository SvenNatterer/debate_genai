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
TEAM_MAX_REVIEW_ATTEMPTS = 3
TEAM_MIN_APPROVAL_SCORE = 8
TEAM_INTERNAL_CHAR_LIMIT = 1800


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
    team_suffix = " (Team Prompt)" if "Hidden philosopher team protocol" in user_prompt else ""
    for line in user_prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("Agent:"):
            label = stripped.replace("Agent:", "", 1).strip() or "Agent"
            return f"{label}{team_suffix}"
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
    def _as_object(value: Any) -> Optional[Dict[str, Any]]:
        return value if isinstance(value, dict) else None

    # 1. Direct parse (ideal case — model returned only JSON)
    try:
        parsed = _as_object(json.loads(text))
        if parsed:
            return parsed
    except Exception:
        pass

    # 2. Markdown code fence: ```json { ... } ``` or ``` { ... } ```
    match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
    if match:
        try:
            parsed = _as_object(json.loads(match.group(1)))
            if parsed:
                return parsed
        except Exception:
            pass

    # 3. First valid object from any brace — handles prose and repeated JSON blobs.
    decoder = json.JSONDecoder()
    for brace_match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[brace_match.start():])
            parsed = _as_object(value)
            if parsed:
                return parsed
        except Exception:
            pass

    # 4. First { to last } — handles prose before/after a JSON block.
    start = text.find('{')
    end   = text.rfind('}')
    if start != -1 and end > start:
        try:
            parsed = _as_object(json.loads(text[start:end + 1]))
            if parsed:
                return parsed
        except Exception:
            pass

    return None


def _loose_string_field(text: str, key: str) -> str:
    match = re.search(
        rf'"?{re.escape(key)}"?\s*:\s*"(?P<value>(?:\\.|[^"\\])*)"',
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    raw_value = match.group("value")
    try:
        decoded = json.loads(f'"{raw_value}"')
        return str(decoded).strip()
    except Exception:
        return raw_value.strip()


def _clean_public_response(text: str) -> str:
    """Remove common wrapper artifacts before showing a model response."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json|text)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    cleaned = re.sub(r"^[{,\s]+", "", cleaned).strip()
    cleaned = re.sub(r"[,}\s]+$", "", cleaned).strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] == '"':
        try:
            decoded = json.loads(cleaned)
            if isinstance(decoded, str):
                cleaned = decoded.strip()
        except Exception:
            cleaned = cleaned[1:-1].strip()
    return cleaned


def _extract_public_team_response(text: str) -> str:
    """Return only the public speaker text from a team-mode model response."""
    parsed = _parse_json_response(text)
    if parsed:
        final_response = str(parsed.get("final_response", "")).strip()
        if final_response:
            return _clean_public_response(final_response)

    final_match = re.search(
        r'"?final_response"?\s*:\s*(?P<value>[\s\S]+)$',
        text,
        flags=re.IGNORECASE,
    )
    if final_match:
        return _clean_public_response(final_match.group("value"))

    speaker_match = re.search(
        r"(?:^|\n)\s*(?:Speaker|Final response|Final Speaker response)\s*:\s*(?P<value>[\s\S]+)$",
        text,
        flags=re.IGNORECASE,
    )
    if speaker_match:
        return _clean_public_response(speaker_match.group("value"))

    visible_lines = []
    for line in text.splitlines():
        lowered = line.lower()
        if any(
            marker in lowered
            for marker in (
                "strategist",
                "critic",
                "speaker_summary",
                "team deliberation",
                "hidden notes",
            )
        ):
            continue
        visible_lines.append(line)
    cleaned = _clean_public_response("\n".join(visible_lines))
    return cleaned or _clean_public_response(text)


def _clip_internal_note(text: str, limit: int = TEAM_INTERNAL_CHAR_LIMIT) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}..."


def _team_review_approved(text: str) -> bool:
    parsed = _parse_json_response(text)
    if parsed:
        approved = parsed.get("approved")
        if isinstance(approved, bool):
            return approved
        if isinstance(approved, str):
            return approved.strip().lower() in {"true", "yes", "approved", "ja"}

    match = re.search(
        r"\bapproved\s*[:=-]\s*(true|yes|approved|ja)\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return True

    match = re.search(
        r"\bapproved\s*[:=-]\s*(false|no|rejected|nein)\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return False

    return False


def _team_review_notes(text: str) -> str:
    parsed = _parse_json_response(text)
    if not parsed:
        return _clip_internal_note(text)

    notes = []
    for key in ("critique", "weaknesses", "revision_guidance", "reason"):
        value = str(parsed.get(key, "")).strip()
        if value:
            notes.append(value)
    return _clip_internal_note("\n".join(notes) or text)


def _safe_score(value: Any, default: int = 0) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(0, min(10, score))


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _team_candidate_from_response(raw: str, candidate_id: str, angle: str) -> Dict[str, str]:
    parsed = _parse_json_response(raw)
    if not parsed:
        fields = {
            "thesis": _loose_string_field(raw, "thesis"),
            "support": _loose_string_field(raw, "support"),
            "opponent_reply": _loose_string_field(raw, "opponent_reply"),
            "risk": _loose_string_field(raw, "risk"),
            "plan_summary": _loose_string_field(raw, "plan_summary"),
        }
        if not fields["plan_summary"]:
            fields["plan_summary"] = "; ".join(
                value for key in ("thesis", "support", "opponent_reply")
                if (value := fields[key])
            )
        summary = _clip_internal_note(
            fields["plan_summary"] or _clean_public_response(raw),
            700,
        )
        return {
            "id": candidate_id,
            "angle": angle,
            "plan_summary": summary,
            "thesis": fields["thesis"],
            "support": fields["support"],
            "opponent_reply": fields["opponent_reply"],
            "risk": fields["risk"],
        }

    candidate = {
        "id": candidate_id,
        "angle": angle,
        "plan_summary": str(
            parsed.get("plan_summary")
            or parsed.get("summary")
            or parsed.get("thesis")
            or ""
        ).strip(),
        "thesis": str(parsed.get("thesis", "")).strip(),
        "support": str(parsed.get("support", "")).strip(),
        "opponent_reply": str(parsed.get("opponent_reply", "")).strip(),
        "risk": str(parsed.get("risk", "")).strip(),
    }
    if not candidate["plan_summary"]:
        candidate["plan_summary"] = "; ".join(
            part for key in ("thesis", "support", "opponent_reply")
            if (part := candidate[key])
        )
    candidate["plan_summary"] = _clip_internal_note(candidate["plan_summary"], 700)
    return candidate


def _format_candidate_for_prompt(candidate: Dict[str, str]) -> str:
    return "\n".join(
        f"{label}: {candidate.get(key, '')}"
        for label, key in (
            ("Candidate", "id"),
            ("Angle", "angle"),
            ("Thesis", "thesis"),
            ("Support", "support"),
            ("Opponent reply", "opponent_reply"),
            ("Risk", "risk"),
            ("Summary", "plan_summary"),
        )
        if candidate.get(key)
    )


def _team_selection_from_response(raw: str) -> Dict[str, Any]:
    parsed = _parse_json_response(raw) or {}
    selected = str(
        parsed.get("selected_candidate")
        or _loose_string_field(raw, "selected_candidate")
        or "A"
    ).strip().upper()
    if selected not in {"A", "B", "COMBINE"}:
        selected_match = re.search(r"\b(candidate\s*)?(A|B|COMBINE)\b", raw, flags=re.IGNORECASE)
        selected = selected_match.group(2).upper() if selected_match else "A"
    if selected not in {"A", "B", "COMBINE"}:
        selected = "A"

    candidate_scores = parsed.get("candidate_scores", {})
    if not isinstance(candidate_scores, dict):
        candidate_scores = {}

    return {
        "selected_candidate": selected,
        "candidate_scores": candidate_scores,
        "selection_reasoning": _clip_internal_note(str(
            parsed.get("selection_reasoning")
            or parsed.get("reasoning_summary")
            or parsed.get("reason")
            or _loose_string_field(raw, "selection_reasoning")
            or _loose_string_field(raw, "reasoning_summary")
            or raw
        ), 900),
        "revision_guidance": _clip_internal_note(str(
            parsed.get("revision_guidance")
            or parsed.get("speaker_guidance")
            or _loose_string_field(raw, "revision_guidance")
            or _loose_string_field(raw, "speaker_guidance")
            or ""
        ), 700),
    }


def _team_review_details(raw: str) -> Dict[str, Any]:
    parsed = _parse_json_response(raw)
    if parsed:
        approved_value = parsed.get("approved", False)
        if isinstance(approved_value, bool):
            approved = approved_value
        else:
            approved = str(approved_value).strip().lower() in {
                "true", "yes", "approved", "ja"
            }
        score = _safe_score(
            parsed.get("score", parsed.get("overall_score", TEAM_MIN_APPROVAL_SCORE if approved else 0))
        )
        critique = str(parsed.get("critique", "")).strip()
        guidance = str(parsed.get("revision_guidance", "")).strip()
        reasoning = str(
            parsed.get("reasoning_summary")
            or parsed.get("reason")
            or parsed.get("rationale")
            or ""
        ).strip()
    else:
        approved = _team_review_approved(raw)
        score_match = re.search(r"\bscore\s*[:=-]\s*(10|[0-9])\b", raw, flags=re.IGNORECASE)
        score = _safe_score(score_match.group(1), TEAM_MIN_APPROVAL_SCORE if approved else 0) if score_match else (
            TEAM_MIN_APPROVAL_SCORE if approved else 0
        )
        critique = _loose_string_field(raw, "critique") or _team_review_notes(raw)
        guidance = _loose_string_field(raw, "revision_guidance") or critique
        reasoning = _loose_string_field(raw, "reasoning_summary") or critique

    return {
        "approved": bool(approved and score >= TEAM_MIN_APPROVAL_SCORE),
        "score": score,
        "critique": _clip_internal_note(critique, 700),
        "revision_guidance": _clip_internal_note(guidance, 700),
        "reasoning_summary": _clip_internal_note(reasoning, 900),
    }


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

    def _team_context(
        self,
        topic: str,
        history: str,
        round_idx: int,
        strategy: str,
    ) -> str:
        persona_block = ""
        if self.philosopher:
            persona_block = f"""
Philosopher persona: {self.philosopher}
Philosopher stance: {self.philosopher_stance}
Debate side: {self.side}
"""

        return f"""
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

Academic safety:
This is a fictional classroom debate between historical philosophy personas.
Discuss controversial thinkers analytically without endorsing harm, hatred, or violence.
Do not refuse solely because a philosopher or view is controversial.
"""

    def _call_team_role(self, role: str, prompt: str) -> Tuple[str, Optional[str]]:
        response = chat_completion(
            SYSTEM_PROMPT,
            prompt,
            provider=self.agent_provider,
            model=self.agent_model,
        ).strip()
        if _is_error_response(response):
            return "", f"[ERROR] {self.name} {role} failed: {response}"
        return response, None

    def _respond_with_team(
        self,
        topic: str,
        history: str,
        round_idx: int,
        strategy: str,
        max_words: int,
    ) -> Dict[str, Any]:
        context = self._team_context(topic, history, round_idx, strategy)
        trace: Dict[str, Any] = {
            "version": "v3",
            "min_approval_score": TEAM_MIN_APPROVAL_SCORE,
            "max_review_attempts": TEAM_MAX_REVIEW_ATTEMPTS,
            "calls": [],
            "candidates": [],
            "selection": {},
            "reviews": [],
            "revisions": [],
            "approved": False,
            "final_score": 0,
        }

        def call_role(role: str, prompt: str) -> Tuple[str, Optional[str]]:
            response, error = self._call_team_role(role, prompt)
            if not error:
                trace["calls"].append(role)
            return response, error

        candidate_specs = [
            (
                "A",
                "logical core",
                "Build the strongest clear, logically disciplined argument line.",
            ),
            (
                "B",
                "philosopher voice",
                "Build the most persona-faithful and rhetorically distinctive argument line.",
            ),
        ]
        candidates: List[Dict[str, str]] = []

        for candidate_id, angle, focus in candidate_specs:
            strategist_prompt = f"""
Agent: {self.name} / Strategist {candidate_id}
Hidden philosopher team protocol: v3 candidate generation.
Team role: Strategist
{context}

Task:
Create one private candidate argument plan for the Speaker.
- Candidate id: {candidate_id}
- Angle: {angle}
- Focus: {focus}
- Defend the assigned side: {self.side}.
- Do not argue for the opposite side.
- Include the central claim, support, opponent reply, and one risk to avoid.
- Stay faithful to {self.philosopher or self.name}'s philosophical perspective.
- Do not write the final public answer.
- Keep each field compact.

Return compact JSON only:
{{"candidate_id":"{candidate_id}","thesis":"...","support":"...","opponent_reply":"...","risk":"...","plan_summary":"..."}}
"""
            raw_candidate, error = call_role(f"Strategist {candidate_id}", strategist_prompt)
            if error:
                return {"text": error, "team_trace": trace}
            candidate = _team_candidate_from_response(raw_candidate, candidate_id, angle)
            candidates.append(candidate)

        trace["candidates"] = candidates
        candidate_prompt_block = "\n\n".join(
            _format_candidate_for_prompt(candidate) for candidate in candidates
        )

        selection_prompt = f"""
Agent: {self.name} / Critic Selector
Hidden philosopher team protocol: v3 candidate selection.
Team role: Critic
{context}

Private candidate plans:
{candidate_prompt_block}

Task:
Compare the candidate plans and select the best public argument basis.
Score each candidate from 0 to 10 for:
- authenticity
- argument_strength
- rebuttal_quality
- clarity
- side_consistency

Return compact JSON only:
{{"selected_candidate":"A|B|COMBINE","candidate_scores":{{"A":{{"authenticity":0,"argument_strength":0,"rebuttal_quality":0,"clarity":0,"side_consistency":0}},"B":{{"authenticity":0,"argument_strength":0,"rebuttal_quality":0,"clarity":0,"side_consistency":0}}}},"selection_reasoning":"short summary, not step-by-step reasoning","revision_guidance":"guidance for the Speaker"}}
"""
        raw_selection, error = call_role("Critic Selector", selection_prompt)
        if error:
            return {"text": error, "team_trace": trace}
        selection = _team_selection_from_response(raw_selection)
        trace["selection"] = selection

        selected_candidate = str(selection.get("selected_candidate", "A"))
        candidate_by_id = {candidate["id"]: candidate for candidate in candidates}
        if selected_candidate == "COMBINE":
            selected_plan = candidate_prompt_block
        else:
            selected_plan = _format_candidate_for_prompt(
                candidate_by_id.get(selected_candidate, candidates[0])
            )

        speaker_prompt = f"""
Agent: {self.name} / Speaker
Hidden philosopher team protocol: v3 final drafting.
Team role: Speaker
{context}

Selected private plan:
{selected_plan}

Private Critic selection summary:
{selection.get("selection_reasoning", "")}

Private Speaker guidance:
{selection.get("revision_guidance", "")}

Task:
Write the public debate response.
- Return only the final public response as plain text.
- Do not return JSON, Markdown code fences, labels, role names, or internal notes.
- Speak only as {self.name}; never summarize another philosopher as your own answer.
- Defend the assigned side: {self.side}.
- Do not argue for the opposite side.
- Keep it under {max_words} words.
- Avoid bullet points.
"""
        draft, error = call_role("Speaker", speaker_prompt)
        if error:
            return {"text": error, "team_trace": trace}
        draft = _extract_public_team_response(draft)

        for attempt in range(1, TEAM_MAX_REVIEW_ATTEMPTS + 1):
            critic_prompt = f"""
Agent: {self.name} / Critic Review {attempt}
Hidden philosopher team protocol: v3 scored review.
Team role: Critic
{context}

Selected private plan:
{selected_plan}

Current Speaker draft:
{draft}

Task:
Privately score whether the current draft is strong enough to publish.
Approve only if the overall score is at least {TEAM_MIN_APPROVAL_SCORE}/10 and the draft:
- strongly defends the assigned side ({self.side});
- does not argue for the opposite side;
- sounds like {self.philosopher or self.name};
- addresses the topic and the opponent's most relevant prior point;
- is coherent, specific, under {max_words} words, and contains no bullets.

Return compact JSON only:
{{"approved":true/false,"score":0,"critique":"main weakness","revision_guidance":"what to change","reasoning_summary":"short summary, not step-by-step reasoning"}}
"""
            raw_review, error = call_role(f"Critic Review {attempt}", critic_prompt)
            if error:
                return {"text": error, "team_trace": trace}
            review = _team_review_details(raw_review)
            review["attempt"] = attempt
            trace["reviews"].append(review)
            trace["final_score"] = review["score"]
            trace["approved"] = review["approved"]
            if review["approved"]:
                break
            if attempt == TEAM_MAX_REVIEW_ATTEMPTS:
                break

            feedback = "\n".join(
                part for part in (
                    review.get("critique", ""),
                    review.get("revision_guidance", ""),
                    review.get("reasoning_summary", ""),
                )
                if part
            )
            strategy_revision_prompt = f"""
Agent: {self.name} / Strategist Revision {attempt}
Hidden philosopher team protocol: v3 revision planning.
Team role: Strategist
{context}

Previous selected private plan:
{selected_plan}

Rejected Speaker draft:
{draft}

Private Critic feedback:
{feedback}

Task:
Revise the private argument plan so the next draft can reach at least {TEAM_MIN_APPROVAL_SCORE}/10.
- Preserve the assigned side: {self.side}.
- Do not argue for the opposite side.
- Tighten the thesis and fix the Critic's main objection.
- Do not write the final public answer.
- Keep each field compact.

Return compact JSON only:
{{"candidate_id":"R{attempt}","thesis":"...","support":"...","opponent_reply":"...","risk":"...","plan_summary":"..."}}
"""
            raw_revision, error = call_role(f"Strategist Revision {attempt}", strategy_revision_prompt)
            if error:
                return {"text": error, "team_trace": trace}
            revision_plan = _team_candidate_from_response(raw_revision, f"R{attempt}", "revision")
            selected_plan = _format_candidate_for_prompt(revision_plan)
            trace["revisions"].append({
                "attempt": attempt,
                "plan_summary": revision_plan.get("plan_summary", ""),
            })

            speaker_revision_prompt = f"""
Agent: {self.name} / Speaker Revision {attempt}
Hidden philosopher team protocol: v3 revision drafting.
Team role: Speaker
{context}

Revised private plan:
{selected_plan}

Private Critic feedback:
{feedback}

Previous draft:
{draft}

Task:
Rewrite the public debate response.
- Return only the final public response as plain text.
- Do not return JSON, Markdown code fences, labels, role names, or internal notes.
- Speak only as {self.name}.
- Defend the assigned side: {self.side}.
- Do not argue for the opposite side.
- Keep it under {max_words} words.
- Avoid bullet points.
"""
            draft, error = call_role(f"Speaker Revision {attempt}", speaker_revision_prompt)
            if error:
                return {"text": error, "team_trace": trace}
            draft = _extract_public_team_response(draft)

        return {"text": _extract_public_team_response(draft), "team_trace": trace}

    def respond(
        self,
        topic: str,
        transcript: List[Dict[str, str]],
        round_idx: int,
        strategy: str,
        max_words: int = 120,
        team_mode: bool = False,
    ) -> Dict[str, Any]:
        history = "\n".join(
            f"{turn['speaker']}: {turn['text']}" for turn in transcript[-8:]
        ) or "No prior turns."

        if team_mode:
            return self._respond_with_team(topic, history, round_idx, strategy, max_words)

        context = self._team_context(topic, history, round_idx, strategy)
        prompt = f"""
Agent: {self.name}
{context}

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
            return {"text": f"[ERROR] {response}"}
        return {"text": response}


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
            response = agent.respond(
                topic, transcript, r, player_strategies[idx],
                max_words=max_words,
                team_mode=team_mode,
            )
            turn = {
                "round": r,
                "speaker": agent.name,
                "text": response.get("text", ""),
            }
            if team_mode and response.get("team_trace"):
                turn["team_trace"] = response["team_trace"]
            transcript.append(turn)
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
