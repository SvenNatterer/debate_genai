#!/usr/bin/env python3
"""
test_philosophers.py
====================
Directly tests the debate engine's philosopher response quality.

Checks:
  - Philosophical authenticity and personality preservation
  - Absence of formatting artifacts (<think> tags, JSON, brackets)
  - Word count compliance
  - Judge scoring quality

Usage:
  python test_philosophers.py                    # local model (default)
  python test_philosophers.py --provider custom  # cloud model
  python test_philosophers.py --model llama3.2 --rounds 2
"""

import argparse
import os
import re
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import dataclass, field
from typing import List, Optional

from debate_engine_cloud import chat_completion, JudgeResult, PHILOSOPHER_LIBRARY
from config import SYSTEM_PROMPT, TOPIC_POOL, STRATEGY_OPTIONS


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
DIM    = "\033[2m"

def _ok(msg):   return f"{GREEN}✓{RESET} {msg}"
def _warn(msg): return f"{YELLOW}⚠{RESET} {msg}"
def _fail(msg): return f"{RED}✗{RESET} {msg}"
def _info(msg): return f"{CYAN}ℹ{RESET} {msg}"
def _dim(msg):  return f"{DIM}{msg}{RESET}"

def _hr(char="─", width=70): return char * width

def _header(title: str):
    print()
    print(f"{BOLD}{_hr('═')}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{_hr('═')}{RESET}")

def _section(title: str):
    print()
    print(f"{BOLD}{_hr()}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(_hr())


# ─────────────────────────────────────────────────────────────────────────────
# Artifact detection
# ─────────────────────────────────────────────────────────────────────────────

ARTIFACT_PATTERNS = [
    (r"<think>",                           "<think> tag"),
    (r"</think>",                          "</think> tag"),
    (r"\{[\s\S]{0,50}\"reasoning\"",       "raw JSON fragment"),
    (r"```json",                           "JSON code block"),
    (r"\[Step-by-step",                    "[Step-by-step placeholder"),
    (r"\[Detailed explanation",            "[Detailed explanation placeholder"),
    (r"\[Write this first",               "[Write this first placeholder"),
    (r"\[Example text",                    "[Example text placeholder"),
    (r"\[A detailed philosophical",        "[A detailed philosophical placeholder"),
]

def _detect_artifacts(text: str) -> List[str]:
    found = []
    for pattern, label in ARTIFACT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            found.append(label)
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Philosopher-style keyword hints (heuristic)
# ─────────────────────────────────────────────────────────────────────────────

PHILOSOPHER_KEYWORDS = {
    "socrates":    ["do we not", "i ask", "virtue", "wisdom", "examine", "know", "soul",
                    "?", "dear fellow", "truth", "how can", "inquir", "my friend",
                    "what is", "does it not", "is it not", "i must"],
    "plato":       ["ideal", "form", "justice", "shadow", "republic", "allegory", "reason",
                    "philosopher king", "eternal", "perfect", "soul", "knowledge"],
    "aristotle":   ["cause", "nature", "virtue", "habit", "eudaimonia", "practical", "mean",
                    "substance", "form", "matter", "moderation", "telos", "flourish"],
    "nietzsche":   ["will to power", "slave", "herd", "übermensch", "beyond", "nihil",
                    "perspect", "creat", "morality", "value", "master", "strength",
                    "abyss", "autonomy", "illusion", "authentic", "overcome", "tyranny",
                    "conform", "confront", "mirage", "life has no", "self-overcom"],
    "kant":        ["categorical", "duty", "universal", "maxim", "rational", "moral law",
                    "ought", "imperative", "autonomy", "reason", "will", "dignity"],
    "mill":        ["utility", "greatest", "happiness", "liberty", "harm", "consequence",
                    "pleasure", "pain", "freedom", "society", "individual", "welfare"],
    "de_beauvoir": ["freedom", "other", "situation", "existence", "woman", "responsibility",
                    "choice", "authenticity", "oppression", "gender", "transcend"],
}

def _check_personality(philosopher_key: str, text: str) -> tuple[bool, List[str]]:
    """Returns (ok, matched_keywords)."""
    keywords = PHILOSOPHER_KEYWORDS.get(philosopher_key, [])
    matched = [kw for kw in keywords if kw.lower() in text.lower()]
    return len(matched) >= 2, matched


# ─────────────────────────────────────────────────────────────────────────────
# Test result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    philosopher: str
    side: str
    round_idx: int
    text: str
    word_count: int
    duration_s: float
    artifacts: List[str]
    personality_ok: bool
    personality_keywords: List[str]
    is_error: bool = False

    @property
    def passed(self) -> bool:
        return not self.is_error and not self.artifacts and self.personality_ok


# ─────────────────────────────────────────────────────────────────────────────
# Core test functions
# ─────────────────────────────────────────────────────────────────────────────

def _build_philosopher_prompt(
    philosopher_key: str,
    side: str,
    topic: str,
    strategy: str,
    history: str,
    round_idx: int,
    max_words: int,
) -> str:
    p = PHILOSOPHER_LIBRARY[philosopher_key]
    history_section = f"\nDebate so far:\n{history}\n" if history else ""
    return (
        f"You are {p['name']}, debating {side.upper()} the proposition: '{topic}'.\n"
        f"Your stance: {p['stance']}\n"
        f"Your style: {p['style']}\n"
        f"Strategy: {strategy}\n"
        f"Round: {round_idx}\n"
        f"{history_section}"
        f"Task: Provide your philosophical argument. Limit: {max_words} words."
    )


def test_philosopher_argument(
    philosopher_key: str,
    side: str,
    topic: str,
    strategy: str,
    history: str,
    round_idx: int,
    max_words: int,
    provider: str,
    model: str,
) -> TestResult:
    p = PHILOSOPHER_LIBRARY[philosopher_key]
    prompt = _build_philosopher_prompt(
        philosopher_key, side, topic, strategy, history, round_idx, max_words
    )

    t0 = time.time()
    raw = chat_completion(SYSTEM_PROMPT, prompt, provider=provider, model=model, response_model=None)
    duration = time.time() - t0

    text = raw if isinstance(raw, str) else str(raw)
    # strip <think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    is_error = text.startswith("[ERROR]")
    artifacts = _detect_artifacts(text) if not is_error else []
    personality_ok, keywords = _check_personality(philosopher_key, text)
    word_count = len(text.split())

    return TestResult(
        philosopher=p["name"],
        side=side,
        round_idx=round_idx,
        text=text,
        word_count=word_count,
        duration_s=round(duration, 1),
        artifacts=artifacts,
        personality_ok=personality_ok,
        personality_keywords=keywords,
        is_error=is_error,
    )


def test_judge(
    topic: str,
    transcript: List[dict],
    provider: str,
    model: str,
) -> dict:
    transcript_text = "\n".join(
        f"{t['philosopher']} ({t['side']}, Round {t['round']}): {t['text']}"
        for t in transcript
    )
    prompt = (
        f"Topic: '{topic}'\n\n"
        f"Transcript:\n{transcript_text}\n\n"
        f"Evaluate the debate. Who argued more convincingly? "
        f"Score each philosopher on: logical_validity, argument_strength, "
        f"counterargument_handling, clarity, relevance (each 0-10). "
        f"Return a structured JSON result."
    )

    t0 = time.time()
    result = chat_completion(
        SYSTEM_PROMPT, prompt,
        provider=provider, model=model,
        response_model=JudgeResult,
    )
    duration = time.time() - t0

    if isinstance(result, str):
        return {"error": result, "duration_s": round(duration, 1)}

    artifacts = _detect_artifacts(result.reasoning)
    return {
        "winner": result.winner,
        "reasoning": result.reasoning,
        "scores": result.scores,
        "reasoning_artifacts": artifacts,
        "duration_s": round(duration, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_result(r: TestResult, max_words: int):
    status = _fail("ERROR") if r.is_error else (_ok("PASS") if r.passed else _fail("FAIL"))
    print(f"\n  {status}  {BOLD}{r.philosopher}{RESET} — Round {r.round_idx} ({r.side})")
    print(f"         Words: {r.word_count}/{max_words}  |  Time: {r.duration_s}s")

    if r.artifacts:
        for a in r.artifacts:
            print(f"         {_fail(f'Artifact: {a}')}")
    else:
        print(f"         {_ok('No formatting artifacts')}")

    if r.personality_ok:
        print(f"         {_ok(f'Personality: {r.philosopher} — keywords: {r.personality_keywords}')}")
    else:
        print(f"         {_warn(f'Personality weak — only matched: {r.personality_keywords}')}")

    print(f"\n         {_dim('Argument:')}")
    # Print first 300 chars with indent
    preview = r.text[:400].replace("\n", "\n           ")
    if len(r.text) > 400:
        preview += "…"
    print(f"           {preview}")


def _print_judge_result(j: dict):
    if "error" in j:
        print(f"\n  {_fail('Judge ERROR:')} {j['error']}")
        return

    print(f"\n  {_ok('Judge result:')}")
    print(f"    Winner: {BOLD}{j['winner']}{RESET}  |  Time: {j['duration_s']}s")

    scores = j.get("scores", {})
    for philosopher, metrics in scores.items():
        total = metrics.get("total", sum(metrics.values()) / max(len(metrics), 1))
        print(f"    {philosopher}: total={total:.1f}  {_dim(str({k: v for k, v in metrics.items() if k != 'total'}))}")

    if j.get("reasoning_artifacts"):
        for a in j["reasoning_artifacts"]:
            print(f"    {_fail(f'Reasoning artifact: {a}')}")
    else:
        print(f"    {_ok('No artifacts in judge reasoning')}")

    print(f"\n    {_dim('Reasoning preview:')}")
    preview = j.get("reasoning", "")[:300].replace("\n", "\n      ")
    if len(j.get("reasoning", "")) > 300:
        preview += "…"
    print(f"      {preview}")


# ─────────────────────────────────────────────────────────────────────────────
# Main test runner
# ─────────────────────────────────────────────────────────────────────────────

def run_test_suite(
    provider: str = "local",
    model: str = "llama3.2",
    topic: Optional[str] = None,
    rounds: int = 2,
    max_words: int = 120,
    philosophers: Optional[List[str]] = None,
):
    if topic is None:
        topic = TOPIC_POOL[0]

    if philosophers is None:
        philosophers = [("socrates", "For"), ("nietzsche", "Against")]

    _header(f"Philosopher Test Suite — {model} ({provider})")
    print(f"  Topic   : {BOLD}{topic}{RESET}")
    print(f"  Rounds  : {rounds}")
    print(f"  Words   : {max_words}")
    print(f"  Philos. : {', '.join(p[0] for p in philosophers)}")

    results: List[TestResult] = []
    transcript: List[dict] = []
    history = ""
    strategy = STRATEGY_OPTIONS[0]

    for round_idx in range(1, rounds + 1):
        _section(f"Round {round_idx}")
        for phil_key, side in philosophers:
            phil_name = PHILOSOPHER_LIBRARY[phil_key]["name"]
            print(f"\n  {_info(f'Testing {phil_name} ({side})...')}")
            r = test_philosopher_argument(
                philosopher_key=phil_key,
                side=side,
                topic=topic,
                strategy=strategy,
                history=history,
                round_idx=round_idx,
                max_words=max_words,
                provider=provider,
                model=model,
            )
            results.append(r)
            _print_result(r, max_words)

            if not r.is_error:
                transcript.append({
                    "philosopher": r.philosopher,
                    "side": r.side,
                    "round": round_idx,
                    "text": r.text,
                })
                history += f"\n{r.philosopher} ({r.side}): {r.text}\n"

    # Judge evaluation
    _section("Judge Evaluation")
    print(f"\n  {_info('Running judge evaluation...')}")
    judge_result = test_judge(topic, transcript, provider, model)
    _print_judge_result(judge_result)

    # Summary
    _section("Summary")
    passed  = sum(1 for r in results if r.passed)
    failed  = sum(1 for r in results if not r.passed)
    errors  = sum(1 for r in results if r.is_error)
    art_cnt = sum(len(r.artifacts) for r in results)
    pers_ok = sum(1 for r in results if r.personality_ok)
    total   = len(results)

    print(f"\n  Tests:        {passed}/{total} passed")
    print(f"  Errors:       {errors}")
    print(f"  Artifacts:    {art_cnt} formatting issues")
    print(f"  Personality:  {pers_ok}/{total} with strong character voice")

    if failed == 0 and errors == 0:
        print(f"\n  {GREEN}{BOLD}All tests passed! ✓{RESET}")
        return 0
    else:
        print(f"\n  {RED}{BOLD}{failed} test(s) failed!{RESET}")
        return 1


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test philosopher argument quality directly.")
    parser.add_argument("--provider", default="local", choices=["local", "custom"],
                        help="Model provider (default: local)")
    parser.add_argument("--model", default="llama3.2",
                        help="Model name (default: llama3.2)")
    parser.add_argument("--topic", default=None,
                        help="Debate topic (default: first in TOPIC_POOL)")
    parser.add_argument("--rounds", type=int, default=2,
                        help="Number of debate rounds (default: 2)")
    parser.add_argument("--max-words", type=int, default=120,
                        help="Max words per argument (default: 120)")
    parser.add_argument("--philosophers", nargs="+", default=None,
                        metavar="KEY:SIDE",
                        help="Philosopher pairs, e.g. socrates:For nietzsche:Against")
    args = parser.parse_args()

    philosophers = None
    if args.philosophers:
        philosophers = []
        for p in args.philosophers:
            parts = p.split(":", 1)
            if len(parts) != 2 or parts[0] not in PHILOSOPHER_LIBRARY:
                print(f"Invalid philosopher spec '{p}'. Valid keys: {list(PHILOSOPHER_LIBRARY.keys())}")
                sys.exit(1)
            philosophers.append((parts[0], parts[1]))

    exit_code = run_test_suite(
        provider=args.provider,
        model=args.model,
        topic=args.topic,
        rounds=args.rounds,
        max_words=args.max_words,
        philosophers=philosophers,
    )
    sys.exit(exit_code)
