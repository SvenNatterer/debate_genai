import json
import os
import random
import urllib.error
import urllib.request
from urllib.parse import urljoin
from dataclasses import dataclass
from typing import Any, Dict, List

from config import AGENT_LIBRARY, PHILOSOPHER_LIBRARY, SYSTEM_PROMPT


def can_reach_ollama(base_url: str, timeout: float = 2.0) -> bool:
    health_url = base_url.rstrip("/") + "/api/tags"

    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        return False


def get_ollama_status(base_url: str, model: str, timeout: float = 2.0) -> tuple[bool, str]:
    health_url = base_url.rstrip("/") + "/api/tags"

    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as response:
            if response.status != 200:
                return False, f"Ollama error: /api/tags returned HTTP {response.status}."

            raw = json.loads(response.read().decode("utf-8"))
            models = raw.get("models", [])
            installed_names = {item.get("name", "") for item in models if isinstance(item, dict)}

            if model not in installed_names:
                if installed_names:
                    available = ", ".join(sorted(installed_names))
                    return False, f"Ollama reachable, but model '{model}' is not installed. Installed models: {available}."
                return False, f"Ollama reachable, but no models are installed. Missing model: '{model}'."

            return True, "ok"
    except urllib.error.HTTPError as exc:
        return False, f"Ollama HTTP error: {exc.code} {exc.reason}."
    except urllib.error.URLError as exc:
        return False, f"Ollama not reachable at {base_url}. Reason: {exc.reason}."
    except TimeoutError:
        return False, f"Ollama request timed out for {base_url}."
    except json.JSONDecodeError:
        return False, "Ollama returned invalid JSON for /api/tags."
    except ValueError as exc:
        return False, f"Invalid Ollama URL '{base_url}': {exc}."


def get_client_and_model() -> tuple[str | None, str | None, str | None]:
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    ok, status_message = get_ollama_status(ollama_base_url, ollama_model)
    if not ok:
        return None, None, status_message

    return ollama_base_url.rstrip("/"), ollama_model, None


def mock_response(agent_name: str, prompt: str) -> str:
    seed = abs(hash((agent_name, prompt))) % (10**6)
    rng = random.Random(seed)

    bank = {
        "Judge": [
            "The stronger side engaged the opponent directly and defended its claims more clearly.",
            "Scoring should reward relevance, rebuttal quality, and argumentative consistency.",
        ],
        "Summarizer": [
            "The debate showed both strengths and weaknesses, with neither side being completely unchallenged.",
            "A balanced reading suggests the topic involves real benefits, but also important trade-offs and risks.",
        ],
    }

    lower_prompt = prompt.lower()

    if agent_name not in bank:
        if "debate side: for" in lower_prompt or "argue clearly in favor" in lower_prompt:
            bank[agent_name] = [
                f"{agent_name} argues that the proposal creates structure, clarity, and practical value.",
                f"{agent_name} emphasizes that guided debate helps people compare reasons instead of accepting a single answer.",
            ]
        elif "debate side: against" in lower_prompt or "risk" in lower_prompt:
            bank[agent_name] = [
                f"{agent_name} warns that persuasive phrasing can hide weak reasoning or biased assumptions.",
                f"{agent_name} argues that apparent balance may still reproduce blind spots from the same model family.",
            ]
        else:
            bank[agent_name] = [
                f"{agent_name} questions the assumptions behind the proposal and asks for stronger justification.",
                f"{agent_name} pushes the debate toward clearer definitions, evidence, and trade-offs.",
            ]

    parts = bank.get(agent_name, ["Mock mode response."])
    return " ".join(rng.sample(parts, k=min(2, len(parts))))


def fallback_or_status_message(user_prompt: str, status_message: str | None) -> str:
    if status_message:
        return status_message

    for role in ["Judge", "Summarizer"]:
        if role in user_prompt:
            return mock_response(role, user_prompt)
    if "Agent:" in user_prompt:
        first_line = [line for line in user_prompt.splitlines() if line.startswith("Agent:")]
        if first_line:
            agent_name = first_line[0].replace("Agent:", "").strip()
            return mock_response(agent_name, user_prompt)
    return "Mock mode response."


def is_ollama_error_response(text: str) -> bool:
    prefixes = (
        "Ollama ",
        "Invalid Ollama ",
    )
    return text.startswith(prefixes)


def chat_completion(system_prompt: str, user_prompt: str) -> str:
    base_url, model, status_message = get_client_and_model()
    if base_url is None or model is None:
        return fallback_or_status_message(user_prompt, status_message)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "stream": False,
    }

    request = urllib.request.Request(
        urljoin(base_url + "/", "api/chat"),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = json.loads(response.read().decode("utf-8"))
            if "error" in raw and raw["error"]:
                return f"Ollama chat error: {raw['error']}"

            message = raw.get("message", {})
            content = message.get("content", "")
            if content.strip():
                return content.strip()
            return "Ollama returned an empty response."
    except urllib.error.HTTPError as exc:
        try:
            error_body = exc.read().decode("utf-8").strip()
        except Exception:
            error_body = ""
        if error_body:
            return f"Ollama HTTP error {exc.code}: {error_body}"
        return f"Ollama HTTP error {exc.code}: {exc.reason}"
    except urllib.error.URLError as exc:
        return f"Ollama connection error at {base_url}: {exc.reason}"
    except TimeoutError:
        return f"Ollama request timed out for model '{model}'."
    except json.JSONDecodeError:
        return "Ollama returned invalid JSON for /api/chat."
    except ValueError as exc:
        return f"Invalid Ollama request configuration: {exc}"


def strategy_to_instructions(strategy: str) -> str:
    mapping = {
        "Logical Rebuttal": """
- Focus on logic and internal consistency.
- Point out weaknesses or contradictions in the opponent's argument.
- Use precise and structured reasoning.
""",
        "Emotional Appeal": """
- Use emotionally engaging language.
- Emphasize human consequences, hope, fear, or justice.
- Make the argument feel vivid and impactful.
""",
        "Counterargument": """
- Directly challenge the opponent's latest point.
- Show why their conclusion does not follow from their premises.
- Stay confrontational but still coherent.
""",
        "Examples and Analogies": """
- Use concrete examples or analogies.
- Make abstract claims easier to understand.
- Keep the examples relevant to the topic.
""",
        "Balanced": """
- Acknowledge trade-offs.
- Sound fair and reflective.
- Defend your side without sounding simplistic or extreme.
""",
    }
    return mapping.get(strategy, "- Argue clearly, persuasively, and stay consistent with your side.")


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

    def respond(
        self,
        topic: str,
        transcript: List[Dict[str, str]],
        round_idx: int,
        strategy: str,
        max_words: int = 120,
    ) -> str:
        history = "\n".join(
            [f"{turn['speaker']}: {turn['text']}" for turn in transcript[-8:]]
        ) or "No prior turns."

        persona_block = ""
        if self.philosopher:
            persona_block = f"""
Philosopher persona: {self.philosopher}
Philosopher stance: {self.philosopher_stance}
Debate side: {self.side}
"""

        strategy_instructions = strategy_to_instructions(strategy)

        prompt = f"""
Agent: {self.name}
Goal: {self.goal}
Style: {self.style}
{persona_block}
Argument strategy: {strategy}

Strategy instructions:
{strategy_instructions}

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
        response = chat_completion(SYSTEM_PROMPT, prompt).strip()
        if is_ollama_error_response(response):
            return f"[ERROR] {response}"
        return response


def build_agents(agent_configs: List[Dict[str, str]]) -> List[DebateAgent]:
    agents: List[DebateAgent] = []
    side_labels = ["For", "Against"]
    side_goals = {
        "For": "Argue clearly in favor of the topic and defend why it is desirable, useful, or justified.",
        "Against": "Argue clearly against the topic and expose why it is risky, flawed, or unjustified.",
    }

    for idx, cfg in enumerate(agent_configs, start=1):
        philosopher = PHILOSOPHER_LIBRARY[cfg["philosopher_key"]]
        side = side_labels[idx - 1]
        agents.append(
            DebateAgent(
                key=f"agent_{idx}",
                name=f"{philosopher['name']} ({side})",
                goal=side_goals[side],
                style=f"{philosopher['style']}; debating from the {side.lower()} side",
                philosopher=philosopher["name"],
                philosopher_stance=philosopher["stance"],
                image=philosopher["image"],
                side=side,
            )
        )
    return agents


def run_debate(
    topic: str,
    rounds: int,
    player_strategies: List[str],
    agent_configs: List[Dict[str, str]],
    max_words: int = 120,
) -> List[Dict[str, str]]:
    transcript: List[Dict[str, str]] = []
    agents = build_agents(agent_configs)

    for r in range(1, rounds + 1):
        for idx, agent in enumerate(agents):
            strategy = player_strategies[idx]
            transcript.append(
                {
                    "speaker": agent.name,
                    "text": agent.respond(topic, transcript, r, strategy, max_words=max_words),
                }
            )

    return transcript


def judge_debate(topic: str, transcript: List[Dict[str, str]]) -> Dict[str, Any]:
    history = "\n".join([f"{turn['speaker']}: {turn['text']}" for turn in transcript])

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

    raw = chat_completion(SYSTEM_PROMPT, prompt).strip()

    if is_ollama_error_response(raw):
        speakers = sorted({t["speaker"] for t in transcript})
        scores = {
            s: {
                "logic": 0,
                "relevance": 0,
                "rebuttal": 0,
                "fairness": 0,
                "total": 0,
            }
            for s in speakers
        }
        return {
            "winner": "No winner",
            "scores": scores,
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
        scores = {}
        for s in speakers:
            logic = random.randint(6, 9)
            relevance = random.randint(6, 9)
            rebuttal = random.randint(5, 9)
            fairness = random.randint(6, 9)
            total = logic + relevance + rebuttal + fairness
            scores[s] = {
                "logic": logic,
                "relevance": relevance,
                "rebuttal": rebuttal,
                "fairness": fairness,
                "total": total,
            }
        winner = max(scores, key=lambda k: scores[k]["total"])
        return {
            "winner": winner,
            "scores": scores,
            "reasoning": "Fallback scoring used because the model did not return valid JSON.",
        }


def summarize_debate(topic: str, transcript: List[Dict[str, str]], judgment: Dict[str, Any]) -> str:
    history = "\n".join([f"{turn['speaker']}: {turn['text']}" for turn in transcript])

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
    summary = chat_completion(SYSTEM_PROMPT, prompt).strip()
    if is_ollama_error_response(summary):
        return f"[ERROR] {summary}"
    return summary
