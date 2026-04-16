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
    "judge": {
        "name": "Judge",
        "goal": "Evaluate arguments on logic, relevance, rebuttal quality, and fairness.",
        "style": "formal, objective",
    },
    "summarizer": {
        "name": "Summarizer",
        "goal": "Provide a balanced summary and strongest arguments from each side.",
        "style": "clear, neutral",
    },
    "moderator": {
        "name": "Moderator",
        "goal": "Keep the debate structured and focused.",
        "style": "neutral, organized, brief",
    },
}

TOPIC_POOL = [
    "What is the meaning of life?",
    "Will AI kill human creativity?",
    "Is human nature fundamentally good or selfish?",
    "Do humans truly have free will, or is everything determined?",
    "Is happiness the highest goal in life?",
    "Is it ever morally acceptable to lie?",
    "Is justice more important than freedom?",
    "Does life have an objective meaning?",
    "Are morality and ethics universal, or culturally relative?",
    "Is suffering necessary for personal growth?",
    "Should the individual submit to the common good?",
    "Can we ever have certain knowledge?",
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