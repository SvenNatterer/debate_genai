import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List

from config import AGENT_LIBRARY, PHILOSOPHER_LIBRARY, SYSTEM_PROMPT

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def get_client_and_model():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

    if OpenAI is None:
        return None, None

    if provider == "ollama":
        return OpenAI(base_url=ollama_base_url, api_key="ollama"), ollama_model

    if os.getenv("OPENAI_API_KEY"):
        return OpenAI(), openai_model

    return None, None


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


def chat_completion(system_prompt: str, user_prompt: str) -> str:
    client, model = get_client_and_model()
    if client is None or model is None:
        for role in ["Judge", "Summarizer"]:
            if role in user_prompt:
                return mock_response(role, user_prompt)
        if "Agent:" in user_prompt:
            first_line = [line for line in user_prompt.splitlines() if line.startswith("Agent:")]
            if first_line:
                agent_name = first_line[0].replace("Agent:", "").strip()
                return mock_response(agent_name, user_prompt)
        return "Mock mode response."

    response = client.chat.completions.create(
        model=model,
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


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
        return chat_completion(SYSTEM_PROMPT, prompt).strip()


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
) -> List[Dict[str, str]]:
    transcript: List[Dict[str, str]] = []
    agents = build_agents(agent_configs)

    for r in range(1, rounds + 1):
        for idx, agent in enumerate(agents):
            strategy = player_strategies[idx]
            transcript.append(
                {
                    "speaker": agent.name,
                    "text": agent.respond(topic, transcript, r, strategy),
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
    return chat_completion(SYSTEM_PROMPT, prompt).strip()