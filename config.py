import random

SYSTEM_PROMPT = (
    "You are part of a structured multi-agent debate system. "
    "Be logically consistent and stay on topic. "
    "Always respect the word limit given in the task."
)

PHILOSOPHER_LIBRARY = {
    # ── THE ANCIENT GREEKS ────────────────────────────────────────────────────

    "socrates": {
        "name": "Socrates",
        "stance": "questioning assumptions through dialogue to expose ignorance and pursue truth",
        "style": "probing, humble, dialectical",
        "image": "images/Socrates.png",
        "audio_provider": "edge",
        "voice_id": "en-GB-ThomasNeural",
        "stereo_panning": -0.5,
    },
    "plato": {
        "name": "Plato",
        "stance": "reasoning about ideal Forms, justice, and the philosopher-king's ordered society",
        "style": "structured, abstract, principled",
        "image": "images/Plato.png",
        "audio_provider": "edge",
        "voice_id": "en-US-ChristopherNeural",
        "stereo_panning": 0.5,
    },
    "aristotle": {
        "name": "Aristotle",
        "stance": "practical reasoning focused on causes, virtue, and flourishing in the real world",
        "style": "systematic, empirical, balanced",
        "image": "images/Aristotle.png",
        "audio_provider": "edge",
        "voice_id": "en-GB-RyanNeural",
        "stereo_panning": -0.3,
    },
    "hypatia": {
        "name": "Hypatia of Alexandria",
        "stance": "mathematics and Neoplatonic philosophy reveal the rational order underlying reality",
        "style": "serene, rigorous, Neoplatonic",
        "image": "images/Hypatia.png",
        "audio_provider": "edge",
        "voice_id": "en-GB-LibbyNeural",
        "stereo_panning": 0.0,
    },

    # ── THE MEDIEVAL WORLD ────────────────────────────────────────────────────

    "aquinas": {
        "name": "Thomas Aquinas",
        "stance": "faith and reason are compatible; natural law reflects divine reason accessible to all",
        "style": "scholastic, systematic, theistic",
        "image": "images/Aquinas.png",
        "audio_provider": "edge",
        "voice_id": "en-IE-ConnorNeural",
        "stereo_panning": 0.7,
    },

    # ── THE EARLY MODERN PERIOD ───────────────────────────────────────────────

    "descartes": {
        "name": "René Descartes",
        "stance": "radical doubt as a method; the thinking self (cogito) is the only certain foundation",
        "style": "methodical, dualist, searching",
        "image": "images/Descartes.png",
        "audio_provider": "edge",
        "voice_id": "en-AU-WilliamNeural",       # distinct from Socrates (en-GB-ThomasNeural)
        "stereo_panning": 0.4,
    },
    "locke": {
        "name": "John Locke",
        "stance": "mind begins as a blank slate; government derives legitimacy from consent and natural rights",
        "style": "empirical, moderate, civic-minded",
        "image": "images/Locke.png",
        "audio_provider": "edge",
        "voice_id": "en-CA-LiamNeural",          # distinct from Aristotle (en-GB-RyanNeural)
        "stereo_panning": 0.6,
    },
    "leibniz": {
        "name": "Gottfried Wilhelm Leibniz",
        "stance": "the universe is composed of mind-like monads; we live in the best of all possible worlds",
        "style": "optimistic, rationalist, mathematically-minded",
        "image": "images/Leibniz.png",
        "audio_provider": "edge",
        "voice_id": "en-NZ-MitchellNeural",      # distinct from Kant (en-GB-OliverNeural)
        "stereo_panning": -0.4,
    },

    # ── THE ENLIGHTENMENT & 19TH CENTURY ─────────────────────────────────────

    "wollstonecraft": {
        "name": "Mary Wollstonecraft",
        "stance": "women possess the same rational capacities as men; political and educational equality is a moral necessity",
        "style": "passionate, egalitarian, morally urgent",
        "image": "images/Wollstonecraft.png",
        "audio_provider": "edge",
        "voice_id": "en-GB-SoniaNeural",
        "stereo_panning": -0.5,
    },
    "kant": {
        "name": "Immanuel Kant",
        "stance": "grounding ethics in duty and the categorical imperative; knowledge shaped by the mind's own structure",
        "style": "formal, rigorous, principled",
        "image": "images/Kant.png",
        "audio_provider": "edge",
        "voice_id": "en-GB-OliverNeural",
        "stereo_panning": -0.8,
    },
    "hegel": {
        "name": "Georg Wilhelm Friedrich Hegel",
        "stance": "reality unfolds through dialectical contradiction; history is Spirit coming to self-knowledge",
        "style": "dense, dialectical, grand-systemic",
        "image": "images/Hegel.png",
        "audio_provider": "edge",
        "voice_id": "en-US-RogerNeural",
        "stereo_panning": -0.7,
    },
    "mill": {
        "name": "John Stuart Mill",
        "stance": "maximising overall happiness while protecting individual liberty from social tyranny",
        "style": "analytical, liberal, consequence-aware",
        "image": "images/Mill.png",
        "audio_provider": "edge",
        "voice_id": "en-US-BrianNeural",
        "stereo_panning": 0.8,
    },
    "marx": {
        "name": "Karl Marx",
        "stance": "history is driven by class struggle; capitalism alienates labour and must be overcome",
        "style": "polemical, materialist, structural",
        "image": "images/Marx.png",
        "audio_provider": "edge",
        "voice_id": "en-US-EricNeural",
        "stereo_panning": -0.6,
    },

    # ── THE 20TH CENTURY ──────────────────────────────────────────────────────

    "nietzsche": {
        "name": "Friedrich Nietzsche",
        "stance": "challenging herd morality, championing the will to power and self-overcoming",
        "style": "provocative, critical, intense",
        "image": "images/Nietzsche.png",
        "audio_provider": "edge",
        "voice_id": "en-US-GuyNeural",
        "stereo_panning": 0.3,
    },
    "wittgenstein": {
        "name": "Ludwig Wittgenstein",
        "stance": "philosophical problems arise from language misuse; meaning is use within a form of life",
        "style": "terse, precise, enigmatic",
        "image": "images/Wittgenstein.png",
        "audio_provider": "edge",
        "voice_id": "en-US-SteffanNeural",
        "stereo_panning": -0.2,
    },
    "arendt": {
        "name": "Hannah Arendt",
        "stance": "political life requires active participation; totalitarianism stems from the 'banality of evil'",
        "style": "lucid, political, historically-grounded",
        "image": "images/Arendt.png",
        "audio_provider": "edge",
        "voice_id": "en-US-AriaNeural",
        "stereo_panning": 0.5,
    },
    "sartre": {
        "name": "Jean-Paul Sartre",
        "stance": "existence precedes essence; humans are radically free and wholly responsible for themselves",
        "style": "existential, confrontational, freedom-obsessed",
        "image": "images/Sartre.png",
        "audio_provider": "edge",
        "voice_id": "en-ZA-LukeNeural",          # distinct from Nietzsche (en-US-GuyNeural)
        "stereo_panning": 0.2,
    },
    "de_beauvoir": {
        "name": "Simone de Beauvoir",
        "stance": "freedom and responsibility are fundamental; gender is socially constructed, not biologically fixed",
        "style": "reflective, critical, existential",
        "image": "images/Beauvoir.png",
        "audio_provider": "edge",
        "voice_id": "en-US-MichelleNeural",
        "stereo_panning": 0.0,
    },
    "foot": {
        "name": "Philippa Foot",
        "stance": "morality is grounded in human nature and virtue, not abstract principles or mere sentiment",
        "style": "precise, virtue-focused, commonsensical",
        "image": "images/Foot.png",
        "audio_provider": "edge",
        "voice_id": "en-GB-MaisieNeural",
        "stereo_panning": -0.3,
    },
    "butler": {
        "name": "Judith Butler",
        "stance": "gender is performative, not innate; identity categories are effects of repeated social acts",
        "style": "deconstructive, theoretical, challenging",
        "image": "images/Butler.png",
        "audio_provider": "edge",
        "voice_id": "en-US-JennyNeural",
        "stereo_panning": 0.3,
    },
}

# ── VOICE MAP (for quick reference & collision checks) ────────────────────────
# en-AU-WilliamNeural   → Descartes
# en-CA-LiamNeural      → Locke
# en-GB-LibbyNeural     → Hypatia
# en-GB-MaisieNeural    → Foot
# en-GB-OliverNeural    → Kant
# en-GB-RyanNeural      → Aristotle
# en-GB-SoniaNeural     → Wollstonecraft
# en-GB-ThomasNeural    → Socrates
# en-IE-ConnorNeural    → Aquinas
# en-NZ-MitchellNeural  → Leibniz
# en-US-AriaNeural      → Arendt
# en-US-BrianNeural     → Mill
# en-US-ChristopherNeural → Plato
# en-US-EricNeural      → Marx
# en-US-GuyNeural       → Nietzsche
# en-US-JennyNeural     → Butler
# en-US-MichelleNeural  → de Beauvoir
# en-US-RogerNeural     → Hegel
# en-US-SteffanNeural   → Wittgenstein
# en-ZA-LukeNeural      → Sartre

AGENT_LIBRARY = {
    "summarizer": {
        "name": "Summarizer",
        "goal": "Provide a balanced summary and strongest arguments from each side.",
        "style": "clear, neutral",
    },
}


TOPIC_POOL = [
    # ── GENERAL STATEMENTS ──────────────────────────────────────────────────
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

    # ── MIND, IDENTITY & CONSCIOUSNESS ───────────────────────────────────────
    "A sufficiently advanced AI deserves moral consideration.",
    "Personal identity is an illusion — there is no continuous self.",
    "Consciousness cannot be fully explained by physical processes.",
    "We have a moral obligation to enhance human intelligence artificially.",
    "Memory defines who we are more than character does.",
    "A person who has completely changed their beliefs and values is no longer the same person.",

    # ── ETHICS & MORAL PHILOSOPHY ─────────────────────────────────────────────
    "It is wrong to bring children into a world of certain suffering.",
    "Animals have rights that humans are morally obligated to respect.",
    "The ends never justify the means.",
    "Moral progress is real and measurable.",
    "Wealth beyond a certain point is morally indefensible.",
    "We owe more to people alive today than to future generations.",
    "Forgiveness is always a virtue.",
    "A just society requires equal outcomes, not merely equal opportunities.",

    # ── POLITICS, SOCIETY & POWER ─────────────────────────────────────────────
    "Democracy is the worst form of government except for all the others.",
    "The nation-state is an obstacle to human flourishing.",
    "Radical transparency — no privacy for anyone, including governments — would make society more just.",
    "Civil disobedience is sometimes not only justified but morally required.",
    "The free market is structurally incapable of solving climate change.",
    "Political leaders should be chosen by sortition — random selection from citizens — rather than election.",

    # ── KNOWLEDGE, SCIENCE & REALITY ──────────────────────────────────────────
    "Science tells us how the world is, but cannot tell us how it ought to be.",
    "Objective truth exists independently of human perception.",
    "The simulation hypothesis is a serious philosophical position, not science fiction.",
    "Mathematical structures exist independently of human minds.",
    "We are morally obligated to take low-probability, high-consequence risks seriously.",

    # ── EXISTENCE, TIME & THE HUMAN CONDITION ─────────────────────────────────
    "Death gives life its meaning — immortality would make existence pointless.",
    "Boredom is a deeper philosophical problem than pain.",
    "The past should constrain the present less than it currently does.",
    "We have a duty to remember historical atrocities, even ones we had no part in.",
    "A life fully devoted to art is more valuable than a life devoted to reducing suffering.",
    "Language does not merely describe reality — it constructs it.",

    # ── TECHNOLOGY & THE FUTURE ───────────────────────────────────────────────
    "Social media has made collective reasoning worse, not better.",
    "Humanity's long-term survival requires leaving Earth.",
    "The pursuit of artificial general intelligence is reckless.",
    "Digital relationships can be as meaningful as physical ones.",
    "Automation makes a universal basic income not just desirable but necessary.",
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
