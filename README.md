# Philosopher Arena

An arcade-style AI debate game where historical philosophers argue for and against a topic. Pick two philosophers, choose a debate strategy, watch them clash — then let a panel of AI judges score the winner.

---

## What it does

- Two philosopher personas debate a topic over multiple rounds
- Each side can use a different AI model and rhetorical strategy
- A configurable panel of 1–3 AI judges scores the debate
- A neutral AI summarizer wraps up the strongest arguments
- Text-to-speech voices read the debate aloud (can be toggled off)

Three game modes are available:

| Mode | Description |
|---|---|
| **Single Philosopher** | Classic 1v1 debate — one philosopher per side |
| **Team Philosopher** | 3-agent teams (Agent A, Agent B, Reviewer) per side |
| **Free Topic** | Enter your own topic; one philosopher team explores it without a fixed pro/contra frame |

---

## Prerequisites

- **Python 3.11 or newer**
- **pip**
- **Ollama** (only needed if you want to run a local model — see below)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/SvenNatterer/debate_genai.git
cd debate_genai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

- **Windows:** `.venv\Scripts\activate`
- **macOS / Linux:** `source .venv/bin/activate`

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your environment

Create a `.env` file in the project root (copy the template below). Values you leave blank will simply disable that backend.

```env
# ── Local model (Ollama) ──────────────────────────────────────────────────────
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3.2

# ── Cloud / Azure AI Services ─────────────────────────────────────────────────
CUSTOM_API_BASE_URL=https://your-resource.cognitiveservices.azure.com
CUSTOM_API_KEY=your-api-key-here
```

### 5. Run the app

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

---

## Model options

The model selector in the UI lists all available options. They fall into two groups:

### Local model — Ollama (no API key needed)

| Model in UI | Ollama model name | Notes |
|---|---|---|
| Llama 3.2 3b (Local) | `llama3.2` | Default local model — runs on most laptops |

**To use the local model:**

1. Install Ollama from [ollama.com](https://ollama.com)
2. Start the Ollama server:
   ```bash
   ollama serve
   ```
3. Pull the model (one-time download, ~2 GB):
   ```bash
   ollama pull llama3.2
   ```
4. Make sure `OLLAMA_BASE_URL` is set in your `.env` (default: `http://localhost:11434/v1`)

The local model works fully offline — no API key or internet connection required after the initial download. Response quality and speed depend on your hardware.

---

### Cloud models — Azure AI Services (API key required)

The following models are available when `CUSTOM_API_BASE_URL` and `CUSTOM_API_KEY` are set in your `.env`:

| Model in UI | Model ID |
|---|---|
| GPT-4.1 mini | `gpt-4.1-mini` |
| GPT-5-chat | `gpt-5-chat` |
| DeepSeek-V3.2 | `DeepSeek-V3.2` |
| Mistral Large 3 | `mistral-Large-3` |
| Mistral Small | `mistral-small-2503` |
| Llama 4 Maverick | `Llama-4-Maverick-17B-128E-Instruct-FP8` |
| Llama 3.3 70B | `Llama-3.3-70B-Instruct` |

Cloud models produce noticeably better debate quality than the local 3B model. Each debater, judge, and summarizer can use a different model independently.

---

## Project structure

```
debate_genai/
├── app.py                  # Entry point — Streamlit page config and stage router
├── ui.py                   # All UI rendering and stage logic
├── debate_engine_cloud.py  # LLM calls, debate loop, judging, summarization
├── audio_engine.py         # Text-to-speech via Edge TTS + pygame playback
├── config.py               # Philosopher library, topic pool, strategies
├── styles.py               # Arcade CSS injected into Streamlit
├── philosophers_guide.md   # Optional philosopher bio texts shown in the UI
├── images/                 # Philosopher portrait images
├── requirements.txt        # Python dependencies
└── .env                    # Your local config (not committed)
```

---

## Troubleshooting

**"Azure not configured — check your .env file"**
The app loaded but the cloud backend is not set up. Either fill in `CUSTOM_API_BASE_URL` and `CUSTOM_API_KEY` in your `.env`, or use the local Ollama model instead.

**Local model selected but no response**
- Make sure Ollama is running: `ollama serve`
- Confirm the model is downloaded: `ollama list`
- Check `OLLAMA_BASE_URL` in your `.env` matches the running Ollama port

**Debate quality is weak**
The local 3B model is intentionally small and fast. For better arguments, switch any agent or judge to a cloud model in the character selection screen.

**Audio does not play**
Audio requires `pygame` to be installed correctly (included in `requirements.txt`) and a working audio output device. You can turn audio off in Settings on the start screen.
