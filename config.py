import random

SYSTEM_PROMPT = (
    "You are part of a structured multi-agent debate system. "
    "Be concise, logically consistent, and stay on topic."
)

PHILOSOPHER_LIBRARY = {
    "socrates": {
        "name": "Socrates",
        "stance": "questioning assumptions through dialogue",
        "style": "probing, humble, dialectical",
        "image": "images/Socrates.png",
    },
    "plato": {
        "name": "Plato",
        "stance": "reasoning about ideals, justice, and order",
        "style": "structured, abstract, principled",
        "image": "images/Plato.png",
    },
    "aristotle": {
        "name": "Aristotle",
        "stance": "practical reasoning focused on causes and consequences",
        "style": "systematic, empirical, balanced",
        "image": "images/Aristotle.png",
    },
    "nietzsche": {
        "name": "Friedrich Nietzsche",
        "stance": "challenging norms, morality, and herd thinking",
        "style": "provocative, critical, intense",
        "image": "images/Nietzsche.png",
    },
    "kant": {
        "name": "Immanuel Kant",
        "stance": "duty, universal rules, and rational consistency",
        "style": "formal, rigorous, principled",
        "image": "images/Kant.png",
    },
    "mill": {
        "name": "John Stuart Mill",
        "stance": "weighing outcomes, liberty, and social utility",
        "style": "analytical, liberal, consequence-aware",
        "image": "images/Mill.png",
    },
    "de_beauvoir": {
        "name": "Simone de Beauvoir",
        "stance": "freedom, responsibility, and social structures shaping human choice",
        "style": "reflective, critical, existential",
        "image": "images/Beauvoir.png",
    },
}

AGENT_LIBRARY = {
    "summarizer": {
        "name": "Summarizer",
        "goal": "Provide a balanced summary and strongest arguments from each side.",
        "style": "clear, neutral",
    },
}

TOPIC_POOL = [
    "Human life has an objective meaning.",
    "AI will reduce human creativity overall.",
    "Human nature is fundamentally good.",
    "Humans have genuine free will.",
    "Happiness should be the highest goal in life.",
    "It is sometimes morally acceptable to lie.",
    "Justice is more important than freedom.",
    "Morality is universal rather than culturally relative.",
    "Suffering is necessary for personal growth.",
    "The individual should submit to the common good.",
    "Certain knowledge is possible.",
    "Technological progress does more good than harm.",
]


STRATEGY_OPTIONS = [
    "Direct rebuttal",
    "Evidence-focused",
    "Emotional persuasion",
    "Devil's advocate",
    "Pragmatic trade-off analysis",
]


def random_topic() -> str:
    return random.choice(TOPIC_POOL)
