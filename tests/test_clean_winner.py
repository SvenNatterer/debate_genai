import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _speaker_aliases(speaker: str) -> list[str]:
    base_name = re.sub(r"\s*\([^)]*\)", "", speaker).strip()
    aliases = [speaker]
    if base_name and base_name != speaker:
        aliases.append(base_name)
    return aliases

def extract_winner(text: str, transcript: list) -> str:
    speakers = []
    for turn in transcript:
        s = turn.get("speaker")
        if s and s not in speakers:
            speakers.append(s)

    match = re.search(r"(?:winner|gewinner):\s*\*?\*?([^\n\*\#]+)", text, re.IGNORECASE)
    if match:
        winner_candidate = match.group(1).strip()
        winner_candidate = re.sub(r"[\*\#\(\)]+", "", winner_candidate).strip()
        for speaker in speakers:
            for alias in _speaker_aliases(speaker):
                if alias.lower() in winner_candidate.lower() or winner_candidate.lower() in alias.lower():
                    return speaker

    low_text = text.lower()
    for speaker in speakers:
        for alias in _speaker_aliases(speaker):
            low_alias = alias.lower()
            if (f"winner is {low_alias}" in low_text or 
                f"{low_alias} is the winner" in low_text or 
                f"{low_alias} wins" in low_text or 
                f"{low_alias} was the winner" in low_text or
                f"gewinner ist {low_alias}" in low_text or
                f"{low_alias} gewinnt" in low_text or
                f"{low_alias} ist der gewinner" in low_text):
                return speaker

    found_speakers = []
    for speaker in speakers:
        for alias in _speaker_aliases(speaker):
            if alias.lower() in low_text:
                found_speakers.append(speaker)
                break
    found_speakers = list(set(found_speakers))
    if len(found_speakers) == 1:
        return found_speakers[0]

    return "N/A"

transcript = [
    {"speaker": "Socrates (For)"},
    {"speaker": "Friedrich Nietzsche (Against)"}
]

test_cases = [
    ("winner: Friedrich Nietzsche (Against)", "Friedrich Nietzsche (Against)"),
    ("winner: **Friedrich Nietzsche (Against)**", "Friedrich Nietzsche (Against)"),
    ("winner: Friedrich Nietzsche", "Friedrich Nietzsche (Against)"),
    ("winner: **Friedrich Nietzsche**", "Friedrich Nietzsche (Against)"),
    ("Winner: Socrates", "Socrates (For)"),
    ("winner: Socrates (For)", "Socrates (For)"),
    ("Winner is Socrates.", "Socrates (For)"),
    ("Friedrich Nietzsche wins this round.", "Friedrich Nietzsche (Against)"),
    ("gewinner: Friedrich Nietzsche (Against)", "Friedrich Nietzsche (Against)"),
    ("gewinner: **Friedrich Nietzsche**", "Friedrich Nietzsche (Against)"),
    ("Gewinner ist Socrates.", "Socrates (For)"),
    ("Friedrich Nietzsche gewinnt diesen Kampf.", "Friedrich Nietzsche (Against)"),
    ("Socrates ist der Gewinner.", "Socrates (For)"),
    ("We have no winner today.", "N/A")
]

for text, expected in test_cases:
    res = extract_winner(text, transcript)
    assert res == expected, f"Failed for '{text}': got '{res}', expected '{expected}'"

print("All extract_winner test cases passed successfully!")


def clean_judge_reasoning(text: str, transcript: list[dict]) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL).strip()
    
    # 1. Try robust extraction first: find all Begründung/Justification/Reasoning blocks
    pattern = r"(?is)\b(Begr(?:ü|u)ndung|Justification|Reasoning)\s*:\s*(.*?)(?=\n\s*[A-Z][a-zA-Z\s]+(?:\([^)]*\))?\s*:\s*\d+/\d+|\n\s*Winner\s*:|$)"
    matches = re.findall(pattern, cleaned)
    
    if matches:
        blocks = []
        for marker, content in matches:
            content_cleaned = content.strip()
            content_cleaned = re.sub(r"(?is)\n?\s*winner\s*:.*$", "", content_cleaned).strip()
            blocks.append(f"{marker}: {content_cleaned}")
        return "\n\n".join(blocks).strip()

    # 2. Fallback to original speaker-based clean logic
    # Identify and extract speaker names/aliases to find their score segments
    speakers = []
    for turn in transcript:
        speaker = str(turn.get("speaker", "")).strip()
        if speaker and speaker not in speakers:
            speakers.append(speaker)

    # Find and remove the score block for each speaker
    ranges = []
    for speaker in speakers:
        start_match = None
        for alias in _speaker_aliases(speaker):
            match = re.search(rf"{re.escape(alias)}\s*(?:\([^)]*\))?\s*:", cleaned, re.IGNORECASE)
            if match and (start_match is None or match.start() < start_match.start()):
                start_match = match
        if not start_match:
            continue

        start = start_match.end()
        end = len(cleaned)
        for other in speakers:
            if other == speaker:
                continue
            for alias in _speaker_aliases(other):
                match = re.search(rf"{re.escape(alias)}\s*(?:\([^)]*\))?\s*:", cleaned[start:], re.IGNORECASE)
                if match:
                    end = min(end, start + match.start())

        marker = re.search(
            r"\b(?:Begr(?:ü|u)ndung|Justification|Reasoning|Winner)\s*:",
            cleaned[start:],
            re.IGNORECASE,
        )
        if marker:
            end = min(end, start + marker.start())

        ranges.append((start_match.start(), end))

    # Sort ranges in descending order to avoid offset shifting issues during removal
    ranges.sort(key=lambda r: r[0], reverse=True)
    for start, end in ranges:
        cleaned = cleaned[:start] + cleaned[end:]

    # Now search for the main reasoning marker and slice from it
    marker = re.search(
        r"\b(?:Begr(?:ü|u)ndung|Justification|Reasoning)\s*:",
        cleaned,
        re.IGNORECASE,
    )
    if marker:
        cleaned = cleaned[marker.start():]
    else:
        aliases = [
            alias.lower()
            for turn in transcript
            for alias in _speaker_aliases(str(turn.get("speaker", "")))
            if alias
        ]
        lines = []
        for line in cleaned.splitlines():
            lower_line = line.lower()
            has_score = bool(re.search(r"\d+(?:[\.,]\d+)?\s*/\s*\d+", line))
            has_speaker = any(alias in lower_line for alias in aliases)
            if has_score and has_speaker:
                continue
            lines.append(line)
        cleaned = "\n".join(lines).strip()

    cleaned = re.sub(r"(?is)\n?\s*winner\s*:.*$", "", cleaned).strip()
    return cleaned


# Test clean_judge_reasoning
raw_judge_text_multiple = """Socrates (For): 7/10, logical validity: 5/10, argument strength: 8/10, counterargument handling: 8/10, clarity: 9/10, relevance: 8/10, overall total: 37/50. Begründung: This score reflects Socrates' ability to articulate a nuanced and balanced perspective that acknowledges the importance of individual autonomy while still prioritizing the common good.

Friedrich Nietzsche (Against): 8/10, logical validity: 9/10, argument strength: 9/10, counterargument handling: 9/10, clarity: 8/10, relevance: 9/10, overall total: 53/60. Begründung: This score highlights Nietzsche's compelling and well-supported arguments that effectively challenge Socrates' views on individual autonomy and the common good.
Winner: Friedrich Nietzsche (Against)"""

expected_cleaned_multiple = """Begründung: This score reflects Socrates' ability to articulate a nuanced and balanced perspective that acknowledges the importance of individual autonomy while still prioritizing the common good.

Begründung: This score highlights Nietzsche's compelling and well-supported arguments that effectively challenge Socrates' views on individual autonomy and the common good."""

cleaned_res = clean_judge_reasoning(raw_judge_text_multiple, transcript)
assert cleaned_res == expected_cleaned_multiple, f"Multiple reasoning blocks failed! Got:\n{cleaned_res}"

# Test simple single justification without scores
simple_text = "Begründung: Die Entscheidung basiert auf der Überlegung."
assert clean_judge_reasoning(simple_text, transcript) == "Begründung: Die Entscheidung basiert auf der Überlegung.", "Simple reasoning block cleaning failed!"

# Test text with no Begründung marker at all
no_marker_text = "Nietzsche was better."
assert clean_judge_reasoning(no_marker_text, transcript) == "Nietzsche was better.", "No marker cleaning failed!"

print("All clean_judge_reasoning test cases passed successfully!")

