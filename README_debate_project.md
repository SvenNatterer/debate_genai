
# Debate Project Starter

Files:
- `debate_agents_notebook.ipynb` — notebook prototype
- `app.py` — Streamlit website

## Run the website

```bash
pip install streamlit openai
streamlit run app.py
```

## OpenAI mode

```bash
export OPENAI_API_KEY=your_key
export OPENAI_MODEL=gpt-4o-mini
streamlit run app.py
```

## Ollama mode

```bash
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3.1:8b
export OLLAMA_BASE_URL=http://localhost:11434/v1
streamlit run app.py
```

## Suggested next project steps
- add source retrieval
- show citations per agent
- compare strategies side by side
- store logs for evaluation
- add a fairness warning / hallucination disclaimer
