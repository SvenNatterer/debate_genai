# Debate AI / Philosopher Arena

A lightweight multi-agent debate game built around local LLM inference with **Ollama**.  
Two agents argue **for** and **against** a given topic, a **Judge** scores the debate, and a **Summarizer** produces a balanced wrap-up.

## Project overview

This project is designed as a small debate engine with a game-like frontend.  
The core idea is:

- choose two philosopher-inspired personas
- assign them opposite debate sides
- let them debate a predefined topic over multiple rounds
- evaluate the result with a judge agent
- generate a concise summary

The current backend is structured so that it can run with **Ollama only**, while still providing fallback behavior when Ollama is not reachable.

---

## File structure

### `app.py`
Main application entry point, typically the Streamlit frontend.

Expected responsibilities:

- page layout and UI flow
- topic selection
- character / philosopher selection
- strategy selection
- starting and displaying debates
- showing judge results and final summary
- surfacing connection status and errors to the user

In short: `app.py` is the interface layer that calls the backend logic.

---

### `debate_engine.py`
Core backend logic for the debate system.

This file currently contains:

- **Ollama connectivity checks**
  - `can_reach_ollama(...)`
  - `get_ollama_status(...)`

- **Model configuration**
  - `get_client_and_model()`

- **Fallback / mock behavior**
  - `mock_response(...)`
  - `fallback_or_status_message(...)`

- **LLM request handling**
  - `chat_completion(...)`

- **Strategy mapping**
  - `strategy_to_instructions(...)`

- **Debate agent implementation**
  - `DebateAgent` dataclass
  - `DebateAgent.respond(...)`

- **Agent construction**
  - `build_agents(...)`

- **Debate execution**
  - `run_debate(...)`

- **Evaluation**
  - `judge_debate(...)`

- **Summary generation**
  - `summarize_debate(...)`

#### Main purpose
`debate_engine.py` is the orchestration layer for the whole debate pipeline.  
It builds agents, formats prompts, sends them to Ollama, collects responses, scores the debate, and creates a final summary.

---

### `config.py`
Central configuration file for reusable constants and project-wide settings.

Based on the imports in `debate_engine.py`, this file is expected to define:

- `SYSTEM_PROMPT`
- `AGENT_LIBRARY`
- `PHILOSOPHER_LIBRARY`

#### Recommended contents

**`SYSTEM_PROMPT`**
- Defines the general behavior of the model
- Keeps all agents concise, structured, and on-topic

**`AGENT_LIBRARY`**
- Stores metadata for special roles such as:
  - Judge
  - Summarizer
  - optionally Pro / Contra templates

**`PHILOSOPHER_LIBRARY`**
- Stores philosopher-specific metadata, for example:
  - display name
  - debate style
  - philosophical stance
  - image path

This file acts as the project's central registry for personas and role definitions.

---

### `requirements.txt`
Lists Python dependencies required to run the project.

Typical dependencies may include:

- `streamlit`
- `requests` or standard-library networking alternatives
- optionally `openai` if older versions supported OpenAI-compatible APIs
- any UI helper libraries you added

Install with:

```bash
pip install -r requirements.txt
```

---

### `.env` (optional)
Optional environment configuration file for local setup.

Typical variables:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b
```

If you use Streamlit locally, this can help keep your configuration clean and separate from source code.

---

### `README.md`
Project documentation.

This file should explain:

- what the project does
- how the files are organized
- how to install dependencies
- how to run the app
- how the debate flow works
- common errors and troubleshooting steps

---

## How the system works

### 1. Topic is chosen
A predefined debate topic is selected in the frontend.

### 2. Two philosopher personas are assigned
Each side gets one philosopher persona from `PHILOSOPHER_LIBRARY`.

### 3. Strategies are selected
Each player or side receives a rhetorical strategy such as:

- Logical Rebuttal
- Emotional Appeal
- Counterargument
- Examples and Analogies
- Balanced

These strategies are converted into prompt instructions by `strategy_to_instructions(...)`.

### 4. Debate runs over multiple rounds
`run_debate(...)` alternates between both agents and stores every response in a transcript.

### 5. Judge evaluates the debate
`judge_debate(...)` requests a strict JSON result containing:

- winner
- per-agent score breakdown
- reasoning

### 6. Summarizer creates the final overview
`summarize_debate(...)` generates a balanced textual summary and highlights the strongest points from both sides.

---

## Ollama setup

This project is configured to use **Ollama** as the local inference backend.

### Start Ollama
Make sure Ollama is running locally.

Example:

```bash
ollama serve
```

### Check installed models

```bash
ollama list
```

### Example model pull

```bash
ollama pull llama3.2:1b
```

### Default environment values
If no environment variables are set, the backend falls back to:

- `OLLAMA_BASE_URL=http://localhost:11434`
- `OLLAMA_MODEL=llama3.2:1b`

---

## Running the project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama

```bash
ollama serve
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## Important functions in `debate_engine.py`

### `get_ollama_status(base_url, model)`
Checks whether:

- Ollama is reachable
- the `/api/tags` endpoint responds correctly
- the required model is installed

This is the main health check before any debate starts.

### `chat_completion(system_prompt, user_prompt)`
Sends a chat request to Ollama via `/api/chat`.

It returns:

- the model response if successful
- a detailed error message if something fails

### `strategy_to_instructions(strategy)`
Maps a strategy label to concrete prompting instructions.  
This keeps the UI-friendly strategy name separate from the actual model prompt.

### `DebateAgent.respond(...)`
Builds the prompt for one agent turn using:

- name
- goal
- style
- philosopher persona
- side
- strategy
- transcript history

### `run_debate(...)`
Controls the full debate loop and produces the transcript.

### `judge_debate(...)`
Requests structured evaluation from the model.  
If the response is invalid JSON, it falls back to randomly generated scores.

### `summarize_debate(...)`
Creates the end-of-debate summary based on topic, transcript, and judgment.

---

## Error handling and fallback behavior

The backend includes several fallback layers.

### If Ollama is not reachable
The system returns a readable status message instead of crashing.

### If the requested model is missing
The system explains which models are installed and which one is missing.

### If judging JSON is invalid
`judge_debate(...)` generates fallback scores so the UI can still continue.

### If summarization or agent calls fail
The returned error text is passed through and can be shown in the UI.

This makes the app more robust during development and demos.

---

## Suggested philosopher config format

Example structure for `PHILOSOPHER_LIBRARY` in `config.py`:

```python
PHILOSOPHER_LIBRARY = {
    "nietzsche": {
        "name": "Friedrich Nietzsche",
        "style": "provocative, sharp, aphoristic",
        "stance": "questions morality, power, and conformity",
        "image": "assets/nietzsche.png",
    },
    "socrates": {
        "name": "Socrates",
        "style": "questioning, dialectical, precise",
        "stance": "seeks truth through questioning and contradiction",
        "image": "assets/socrates.png",
    },
}
```

---

## Suggested role config format

Example structure for `AGENT_LIBRARY`:

```python
AGENT_LIBRARY = {
    "judge": {
        "goal": "Evaluate which side argued more clearly, fairly, and consistently.",
        "style": "neutral, structured, analytical",
    },
    "summarizer": {
        "goal": "Summarize the debate in a balanced way.",
        "style": "clear, balanced, concise",
    },
}
```

---

## Possible extensions

Good next steps for the project:

- persistent match history
- multiple debate stages / screens
- improved judge rubric
- animated topic selection
- coin-flip side assignment
- stronger transcript memory across rounds
- downloadable debate logs
- tournament mode
- different local models for debaters vs judge
- explicit score visualization in the UI

---

## Troubleshooting

### The UI still shows mock mode
Check that:

- Ollama is actually running
- the `OLLAMA_BASE_URL` is correct
- the selected model is installed
- the frontend is really calling the updated backend code

### Ollama is reachable but no answer is returned
Possible causes:

- wrong model name
- invalid JSON returned by the endpoint
- timeout during `/api/chat`
- empty message content from the model

### Debate quality is weak
Possible reasons:

- model is too small
- prompts are too short or too generic
- debate history window is too limited
- strategy instructions are not distinctive enough

---

## Summary

This project separates the debate system into:

- **UI layer** → `app.py`
- **logic/orchestration layer** → `debate_engine.py`
- **configuration/persona layer** → `config.py`

That structure makes it easier to extend the game with new philosophers, strategies, stages, and evaluation logic.

