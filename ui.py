import base64
import html
import json
import re
import concurrent.futures
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

from config import PHILOSOPHER_LIBRARY, STRATEGY_OPTIONS, random_topic
from debate_engine_cloud import (
    aggregate_judgments,
    get_token_usage,
    judge_debate,
    reset_token_usage,
    run_debate,
    summarize_debate,
)

# ── Philosopher bio guide ─────────────────────────────────────────────────────

_PHILOSOPHER_BIOS: dict[str, str] | None = None


def _load_philosopher_bios() -> dict[str, str]:
    global _PHILOSOPHER_BIOS
    if _PHILOSOPHER_BIOS is not None:
        return _PHILOSOPHER_BIOS
    guide_path = Path(__file__).parent / "philosophers_guide.md"
    if not guide_path.exists():
        _PHILOSOPHER_BIOS = {}
        return _PHILOSOPHER_BIOS
    text = guide_path.read_text(encoding="utf-8")
    name_to_key = {v["name"]: k for k, v in PHILOSOPHER_LIBRARY.items()}
    bios: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    def _flush() -> None:
        if current_key and current_lines:
            bios[current_key] = "\n".join(current_lines).strip()

    for line in text.splitlines():
        if line.startswith("### "):
            _flush()
            current_key = None
            current_lines = []
            heading_text = line[4:]
            for name, key in name_to_key.items():
                if heading_text.startswith(name):
                    current_key = key
                    break
        elif line.startswith("## ") or line.strip() == "---":
            _flush()
            current_key = None
            current_lines = []
        elif current_key is not None:
            current_lines.append(line)
    _flush()
    _PHILOSOPHER_BIOS = bios
    return bios


# Focus instructions injected into the judge prompt per specialisation.
# Index corresponds to judge number (0-based) for a given panel size.
JUDGE_ROLE_LABELS = {
    1: ["General Evaluation"],
    2: ["Logic & Reasoning", "Debate & Communication"],
    3: ["Logic", "Debate", "Clarity"],
}

JUDGE_FOCUS = {
    1: [""],
    2: [
        "You are the Logic & Reasoning Judge. Emphasise Logical Validity and Argument Strength above the other metrics.",
        "You are the Debate & Communication Judge. Emphasise Counterargument Handling, Clarity, and Relevance above the other metrics.",
    ],
    3: [
        "You are the Logic Judge. Emphasise Logical Validity and Argument Strength above the other metrics.",
        "You are the Debate Judge. Emphasise Counterargument Handling and Relevance above the other metrics.",
        "You are the Clarity Judge. Emphasise Clarity and how well each side communicates their reasoning above the other metrics.",
    ],
}

METRIC_LABELS = {
    "logical_validity":         "Logical Validity",
    "argument_strength":        "Argument Strength",
    "counterargument_handling": "Counterarg. Handling",
    "clarity":                  "Clarity",
    "relevance":                "Relevance",
    "total":                    "Total",
}


# Available models shown in the UI selector
MODEL_OPTIONS = {
    "Llama3.2:3b (Local)":            ("local", "llama3.2"),
    "GPT-4.1 mini  (Platform)":       ("custom", "gpt-4.1-mini"),
    "GPT-5-chat  (Platform)":         ("custom", "gpt-5-chat"),
    "DeepSeek-V3.2  (Platform)":      ("custom", "DeepSeek-V3.2"),
    "Mistral Large 3  (Platform)":    ("custom", "mistral-Large-3"),
    "Mistral Small  (Platform)":      ("custom", "mistral-small-2503"),
    "Llama 4 Maverick  (Platform)":   ("custom", "Llama-4-Maverick-17B-128E-Instruct-FP8"),
    "Llama 3.3 70B  (Platform)":      ("custom", "Llama-3.3-70B-Instruct"),
}

MODEL_DEFAULTS_VERSION = 2

DEFAULT_AGENT_MAX_WORDS = 120
MIN_AGENT_MAX_WORDS = 40
MAX_AGENT_MAX_WORDS = 300
DEFAULT_ARGUMENT_ROUNDS = 2
MIN_ARGUMENT_ROUNDS = 1
MAX_ARGUMENT_ROUNDS = 5
TEAM_SIDE_KEYS = ("for", "against")
TEAM_CONFIG_SIDE_KEYS = ("for", "against", "free")
TEAM_SIDE_LABELS = {
    "for": "Pro Team",
    "against": "Contra Team",
    "free": "Free topic team",
}
TEAM_ROLE_DEFS = (
    ("agent_a", "Support", "socrates"),
    ("agent_b", "Counter", "nietzsche"),
    ("reviewer", "Critic", "aristotle"),
)
TEAM_DEFAULTS_VERSION = 5


def _local_model_label() -> str:
    return next(
        (label for label, (provider, _) in MODEL_OPTIONS.items() if provider == "local"),
        list(MODEL_OPTIONS.keys())[0],
    )


def image_to_data_uri(image_path: str) -> str:
    image_bytes = Path(image_path).read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

_DEBATE_PARAM_KEYS = [
    "_dp_phil1", "_dp_phil2", "_dp_model1", "_dp_model2",
    "_dp_strat1", "_dp_strat2", "_dp_num_judges",
    "_dp_judge1_model", "_dp_judge2_model", "_dp_judge3_model",
    "_dp_agent_configs", "_dp_agent_max_words", "_dp_rounds",
    "_dp_developer_mode", "_dp_audio_transcription", "_dp_speaker_speed",
    "_dp_philosopher_teams", "_dp_team_configs", "_dp_start_mode",
    "_character_stage_seeded",
    "_loading_screen_primed",
    "_dp_max_review_attempts",
]


def reset_game() -> None:
    st.session_state["stage"] = 0
    st.session_state["selected_topic"] = random_topic()
    st.session_state["rounds"] = DEFAULT_ARGUMENT_ROUNDS
    st.session_state["player1_strategy"] = STRATEGY_OPTIONS[0]
    st.session_state["player2_strategy"] = STRATEGY_OPTIONS[0]
    st.session_state["ui_agent1_philosopher_name"] = PHILOSOPHER_LIBRARY["socrates"]["name"]
    st.session_state["ui_agent2_philosopher_name"] = PHILOSOPHER_LIBRARY["nietzsche"]["name"]
    st.session_state["agent1_philosopher_name"] = PHILOSOPHER_LIBRARY["socrates"]["name"]
    st.session_state["agent2_philosopher_name"] = PHILOSOPHER_LIBRARY["nietzsche"]["name"]
    default_model = _local_model_label()
    st.session_state["agent1_model_label"] = default_model
    st.session_state["agent2_model_label"] = default_model
    st.session_state["judge1_model_label"] = default_model
    st.session_state["judge2_model_label"] = default_model
    st.session_state["judge3_model_label"] = default_model
    st.session_state["summary_model_label"] = default_model
    st.session_state["_pref_summary_model"] = default_model
    st.session_state["_model_defaults_version"] = MODEL_DEFAULTS_VERSION
    st.session_state.setdefault("num_judges", 1)
    st.session_state["agent_max_words"] = DEFAULT_AGENT_MAX_WORDS
    st.session_state["developer_mode"] = False
    st.session_state["audio_transcription"] = False
    st.session_state["speaker_speed"] = 1.0
    st.session_state["philosopher_teams"] = False
    st.session_state["start_mode"] = "single"
    st.session_state["team_max_review_attempts"] = 3
    st.session_state["free_topic_input"] = ""
    _reset_team_mode_defaults()

    # summary is always enabled — no preference needed
    st.session_state["_pref_agent_max_words"] = DEFAULT_AGENT_MAX_WORDS
    st.session_state["_pref_rounds"] = DEFAULT_ARGUMENT_ROUNDS
    st.session_state["_pref_developer_mode"] = False
    st.session_state["_pref_audio_transcription"] = False
    st.session_state["_pref_speaker_speed"] = 1.0
    st.session_state["_pref_philosopher_teams"] = False
    st.session_state["_pref_start_mode"] = "single"
    st.session_state["_pref_team_configs"] = _team_configs_from_state()

    for key in [
        "transcript", "judgment", "judge_results", "summary", "topic", "agent_configs",
        "token_usage", "team_traces", "team_mode_active", "team_configs", "judge_step",
        *_DEBATE_PARAM_KEYS,
    ]:
        st.session_state.pop(key, None)


def _valid_model_label(value: object, fallback: object | None = None) -> str:
    model_keys = list(MODEL_OPTIONS.keys())
    if isinstance(value, str) and value in MODEL_OPTIONS:
        return value
    if isinstance(fallback, str) and fallback in MODEL_OPTIONS:
        return fallback
    return model_keys[0]


def _save_debate_params(
    agent1_name: str | None = None,
    agent2_name: str | None = None,
    agent1_model_label: str | None = None,
    agent2_model_label: str | None = None,
    judge_model_labels: list[str] | None = None,
) -> None:
    """Copy live widget values to stable keys before a stage transition.

    Widget-bound keys are disowned by Streamlit when their widget isn't rendered,
    so we snapshot them here while they are guaranteed to be current.
    """
    default_m = list(MODEL_OPTIONS.keys())[0]
    resolved_agent1_name = agent1_name or st.session_state.get("ui_agent1_philosopher_name") or st.session_state.get("agent1_philosopher_name")
    resolved_agent2_name = agent2_name or st.session_state.get("ui_agent2_philosopher_name") or st.session_state.get("agent2_philosopher_name")
    start_mode = _current_start_mode()

    st.session_state["_dp_phil1"]       = _valid_philosopher_name(resolved_agent1_name, "socrates")
    st.session_state["_dp_start_mode"]  = start_mode
    if start_mode == "free":
        st.session_state["_dp_phil2"] = ""
        st.session_state["_dp_agent_configs"] = _agent_configs_from_names(
            st.session_state["_dp_phil1"],
            None,
            single=True,
        )
    else:
        st.session_state["_dp_phil2"] = _valid_philosopher_name(resolved_agent2_name, "nietzsche")
        st.session_state["_dp_agent_configs"] = _agent_configs_from_names(
            st.session_state["_dp_phil1"],
            st.session_state["_dp_phil2"],
        )
    st.session_state["_dp_model1"]      = _valid_model_label(
        agent1_model_label, st.session_state.get("agent1_model_label", default_m)
    )
    st.session_state["_dp_model2"]      = _valid_model_label(
        agent2_model_label, st.session_state.get("agent2_model_label", default_m)
    )
    st.session_state["_dp_strat1"]      = st.session_state.get("player1_strategy", STRATEGY_OPTIONS[0])
    st.session_state["_dp_strat2"]      = st.session_state.get("player2_strategy", STRATEGY_OPTIONS[0])
    st.session_state["_dp_agent_max_words"] = _valid_agent_max_words(
        st.session_state.get("_pref_agent_max_words")
        or st.session_state.get("agent_max_words")
    )
    st.session_state["_dp_rounds"] = _valid_argument_rounds(
        st.session_state.get("_pref_rounds")
        or st.session_state.get("rounds")
    )
    st.session_state["_dp_developer_mode"] = bool(
        st.session_state.get("_pref_developer_mode")
        or st.session_state.get("developer_mode", False)
    )
    st.session_state["_dp_audio_transcription"] = _session_bool(
        "_pref_audio_transcription",
        "audio_transcription",
        default=False,
    )
    st.session_state["_dp_speaker_speed"] = float(
        st.session_state.get("_pref_speaker_speed")
        or st.session_state.get("speaker_speed", 1.0)
        or 1.0
    )
    st.session_state["_dp_philosopher_teams"] = bool(
        st.session_state.get("_pref_philosopher_teams")
        or st.session_state.get("philosopher_teams", False)
    )
    st.session_state["_dp_max_review_attempts"] = int(
        st.session_state.get("_pref_team_max_review_attempts")
        or st.session_state.get("team_max_review_attempts", 3)
    )
    team_configs = _team_configs_from_state()
    st.session_state["_pref_team_configs"] = team_configs
    st.session_state["_dp_team_configs"] = team_configs
    num_j = 0 if start_mode == "free" else st.session_state.get("num_judges", 1)
    st.session_state["_dp_num_judges"]  = num_j
    st.session_state["_dp_summary_model"] = st.session_state.get("summary_model_label", default_m)
    judge_model_labels = judge_model_labels or []
    for i in range(3):
        live_judge_model = judge_model_labels[i] if i < len(judge_model_labels) else None
        st.session_state[f"_dp_judge{i+1}_model"] = _valid_model_label(
            live_judge_model,
            st.session_state.get(f"judge{i+1}_model_label", default_m),
        )


def ensure_session_state() -> None:
    if "stage" not in st.session_state:
        reset_game()
        return

    st.session_state.setdefault("selected_topic", random_topic())
    st.session_state.setdefault("rounds", DEFAULT_ARGUMENT_ROUNDS)
    st.session_state.setdefault("player1_strategy", STRATEGY_OPTIONS[0])
    st.session_state.setdefault("player2_strategy", STRATEGY_OPTIONS[0])
    st.session_state.setdefault(
        "ui_agent1_philosopher_name", PHILOSOPHER_LIBRARY["socrates"]["name"]
    )
    st.session_state.setdefault(
        "ui_agent2_philosopher_name", PHILOSOPHER_LIBRARY["nietzsche"]["name"]
    )
    st.session_state.setdefault(
        "agent1_philosopher_name", st.session_state["ui_agent1_philosopher_name"]
    )
    st.session_state.setdefault(
        "agent2_philosopher_name", st.session_state["ui_agent2_philosopher_name"]
    )
    default_model = _local_model_label()
    model_label_keys = [
        "agent1_model_label",
        "agent2_model_label",
        "judge1_model_label",
        "judge2_model_label",
        "judge3_model_label",
        "summary_model_label",
        "_pref_summary_model",
    ]
    if st.session_state.get("_model_defaults_version") != MODEL_DEFAULTS_VERSION:
        for key in model_label_keys:
            st.session_state[key] = default_model
        st.session_state["_model_defaults_version"] = MODEL_DEFAULTS_VERSION
    for key in model_label_keys:
        st.session_state.setdefault(key, default_model)
    st.session_state.setdefault("num_judges", 1)
    # include_summary removed — summary is always generated
    st.session_state.setdefault("agent_max_words", DEFAULT_AGENT_MAX_WORDS)
    st.session_state.setdefault("developer_mode", False)
    st.session_state.setdefault("audio_transcription", False)
    st.session_state.setdefault("speaker_speed", 1.0)
    st.session_state.setdefault("philosopher_teams", False)
    st.session_state.setdefault("start_mode", "single")
    st.session_state.setdefault("team_max_review_attempts", 3)
    st.session_state.setdefault("free_topic_input", "")
    if (
        st.session_state.get("philosopher_teams")
        and st.session_state.get("_team_defaults_version") != TEAM_DEFAULTS_VERSION
    ):
        _reset_team_mode_defaults()
    else:
        _ensure_team_mode_defaults()
    # summary is always enabled — no preference key needed
    st.session_state.setdefault("_pref_agent_max_words", DEFAULT_AGENT_MAX_WORDS)
    st.session_state.setdefault("_pref_rounds", DEFAULT_ARGUMENT_ROUNDS)
    st.session_state.setdefault("_pref_developer_mode", False)
    st.session_state.setdefault("_pref_audio_transcription", False)
    st.session_state.setdefault("_pref_speaker_speed", 1.0)
    st.session_state.setdefault("_pref_philosopher_teams", False)
    st.session_state.setdefault("_pref_team_max_review_attempts", 3)
    st.session_state.setdefault("_pref_start_mode", "single")
    st.session_state.setdefault("_pref_team_configs", _team_configs_from_state())


# Helper function to validate philosopher names
def _valid_philosopher_name(value: str | None, fallback_key: str) -> str:
    valid_names = {v["name"] for v in PHILOSOPHER_LIBRARY.values()}
    fallback_name = PHILOSOPHER_LIBRARY[fallback_key]["name"]

    if value in valid_names:
        return value

    return fallback_name


def get_judge_image_path(role: str) -> str:
    role_lower = str(role or "").lower()
    if "logic" in role_lower:
        return "images/Judge_Logic.png"
    elif "debate" in role_lower:
        return "images/Judge_Debate.png"
    elif "clarity" in role_lower:
        return "images/Judge_Clarity.png"
    else:
        return "images/Judge_General.png"


def _current_start_mode() -> str:
    mode = (
        st.session_state.get("_dp_start_mode")
        or st.session_state.get("_pref_start_mode")
        or st.session_state.get("start_mode")
        or "single"
    )
    return mode if mode in {"single", "team", "free"} else "single"


def _session_bool(*keys: str, default: bool = False) -> bool:
    for key in keys:
        if key in st.session_state:
            return bool(st.session_state[key])
    return default


def _philosopher_name_from_key(key: str) -> str:
    return PHILOSOPHER_LIBRARY.get(key, PHILOSOPHER_LIBRARY["socrates"])["name"]


def _philosopher_key_from_name(name: str | None, fallback_key: str) -> str:
    philosopher_options = {v["name"]: k for k, v in PHILOSOPHER_LIBRARY.items()}
    valid_name = _valid_philosopher_name(name, fallback_key)
    return philosopher_options[valid_name]


def _team_widget_key(side: str, role_key: str, field: str) -> str:
    return f"team_v{TEAM_DEFAULTS_VERSION}_{side}_{role_key}_{field}"


def _team_role_defaults(side: str, role_key: str, fallback_key: str) -> tuple[str, str]:
    if side == "against" and role_key == "agent_a":
        fallback_key = "nietzsche"
    if side == "against" and role_key == "agent_b":
        fallback_key = "socrates"
    default_strategy = "Evidence-focused" if side == "free" else STRATEGY_OPTIONS[0]
    return _philosopher_name_from_key(fallback_key), default_strategy


def _ensure_team_mode_defaults() -> None:
    for side in TEAM_CONFIG_SIDE_KEYS:
        for role_key, _role_label, fallback_key in TEAM_ROLE_DEFS:
            default_name, default_strategy = _team_role_defaults(side, role_key, fallback_key)
            st.session_state.setdefault(_team_widget_key(side, role_key, "philosopher"), default_name)
            st.session_state.setdefault(_team_widget_key(side, role_key, "strategy"), default_strategy)


def _reset_team_mode_defaults() -> None:
    for side in TEAM_CONFIG_SIDE_KEYS:
        for role_key, _role_label, fallback_key in TEAM_ROLE_DEFS:
            default_name, default_strategy = _team_role_defaults(side, role_key, fallback_key)
            st.session_state[_team_widget_key(side, role_key, "philosopher")] = default_name
            st.session_state[_team_widget_key(side, role_key, "strategy")] = default_strategy
    st.session_state["_team_defaults_version"] = TEAM_DEFAULTS_VERSION


def _team_configs_from_state() -> dict:
    _ensure_team_mode_defaults()
    configs: dict = {}
    for side in TEAM_CONFIG_SIDE_KEYS:
        side_config: dict = {}
        for role_key, role_label, fallback_key in TEAM_ROLE_DEFS:
            philosopher_name = st.session_state.get(_team_widget_key(side, role_key, "philosopher"))
            strategy = st.session_state.get(_team_widget_key(side, role_key, "strategy"), STRATEGY_OPTIONS[0])
            if strategy not in STRATEGY_OPTIONS:
                strategy = STRATEGY_OPTIONS[0]
            side_config[role_key] = {
                "label": role_label,
                "philosopher_key": _philosopher_key_from_name(philosopher_name, fallback_key),
                "philosopher_name": _valid_philosopher_name(philosopher_name, fallback_key),
                "strategy": strategy,
            }
        configs[side] = side_config
    return configs


def _valid_agent_max_words(value: int | str | None) -> int:
    try:
        words = int(value)
    except (TypeError, ValueError):
        words = DEFAULT_AGENT_MAX_WORDS

    return max(MIN_AGENT_MAX_WORDS, min(MAX_AGENT_MAX_WORDS, words))


def _valid_argument_rounds(value: int | str | None) -> int:
    try:
        rounds = int(value)
    except (TypeError, ValueError):
        rounds = DEFAULT_ARGUMENT_ROUNDS

    return max(MIN_ARGUMENT_ROUNDS, min(MAX_ARGUMENT_ROUNDS, rounds))


def _agent_configs_from_names(
    agent1_name: str | None,
    agent2_name: str | None,
    single: bool = False,
):
    philosopher_options = {v["name"]: k for k, v in PHILOSOPHER_LIBRARY.items()}

    agent1_name = _valid_philosopher_name(agent1_name, "socrates")
    agent2_name = _valid_philosopher_name(agent2_name, "nietzsche")

    configs = [
        {
            "philosopher_key": philosopher_options[agent1_name],
            "philosopher_name": agent1_name,
        }
    ]
    if single:
        return configs
    configs.append(
        {
            "philosopher_key": philosopher_options[agent2_name],
            "philosopher_name": agent2_name,
        }
    )
    return configs


def _seed_character_stage_widgets() -> None:
    if st.session_state.get("_character_stage_seeded"):
        return

    st.session_state["ui_agent1_philosopher_name"] = _valid_philosopher_name(
        st.session_state.get("agent1_philosopher_name")
        or st.session_state.get("ui_agent1_philosopher_name"),
        "socrates",
    )
    st.session_state["ui_agent2_philosopher_name"] = _valid_philosopher_name(
        st.session_state.get("agent2_philosopher_name")
        or st.session_state.get("ui_agent2_philosopher_name"),
        "nietzsche",
    )
    st.session_state["_character_stage_seeded"] = True


def _clear_stale_stage_tail(slots: int = 30) -> None:
    for _ in range(slots):
        st.empty()


def current_agent_configs(use_saved_params: bool = False):
    start_mode = _current_start_mode()
    if use_saved_params:
        saved_configs = st.session_state.get("_dp_agent_configs")
        if isinstance(saved_configs, list):
            if start_mode == "free" and len(saved_configs) >= 1:
                return saved_configs[:1]
            if len(saved_configs) >= 2:
                return saved_configs[:2]

    if use_saved_params:
        raw_agent1_name = (
            st.session_state.get("_dp_phil1")
            or st.session_state.get("agent1_philosopher_name")
            or st.session_state.get("ui_agent1_philosopher_name")
        )
        raw_agent2_name = (
            st.session_state.get("_dp_phil2")
            or st.session_state.get("agent2_philosopher_name")
            or st.session_state.get("ui_agent2_philosopher_name")
        )
    else:
        raw_agent1_name = st.session_state.get("ui_agent1_philosopher_name") or st.session_state.get("agent1_philosopher_name")
        raw_agent2_name = st.session_state.get("ui_agent2_philosopher_name") or st.session_state.get("agent2_philosopher_name")

    return _agent_configs_from_names(
        raw_agent1_name,
        raw_agent2_name,
        single=start_mode == "free",
    )


def render_header() -> None:
    st.markdown('<div class="arcade-title">Philosopher Arena</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="arcade-subtitle">Arcade Debate System</div>',
        unsafe_allow_html=True,
    )


def render_mode_status(cloud_active: bool, status_message: str | None = None) -> None:
    import html as _html
    if cloud_active:
        st.markdown('<div class="small-status">Debate engine ready</div>', unsafe_allow_html=True)
    else:
        text = status_message or "Azure not configured — check your .env file"
        st.markdown(f'<div class="small-status">{_html.escape(text)}</div>', unsafe_allow_html=True)


def render_top_bar() -> None:
    if st.session_state.get("stage") == 3:
        return

    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        st.empty()
    with top_col2:
        if st.button("New Game", use_container_width=True):
            reset_game()
            st.rerun()


def render_topic_panel(topic: str, small: bool = False) -> None:
    if small:
        st.markdown(
            f"""
            <div class="topic-panel small-topic-panel">
                <div class="topic-label small-topic-label">Debate Topic:</div>
                <div class="topic-text small-topic-text">{topic}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="topic-panel">
                <div class="topic-label">Debate Topic:</div>
                <div class="topic-text">{topic}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_fighter_card(philosopher_key: str, side_label: str) -> None:
    philosopher = PHILOSOPHER_LIBRARY[philosopher_key]
    image_src = image_to_data_uri(philosopher["image"])
    portrait_path = Path(philosopher["image"])
    sprite_path = portrait_path.with_name(f"{portrait_path.stem}_idle_strip.png")
    if sprite_path.exists():
        sprite_src = image_to_data_uri(str(sprite_path))
        fighter_media = (
            f'<div class="fighter-animation-stage">'
            f'<div class="fighter-sprite-bob">'
            f'<div class="fighter-sprite-idle" style="background-image:url({sprite_src});" '
            f'aria-label="{html.escape(philosopher["name"])} idle animation"></div>'
            f'</div>'
            f'<div class="fighter-idle-shadow"></div>'
            f'</div>'
        )
    else:
        fighter_media = f'<img src="{image_src}" class="fighter-image" alt="{html.escape(philosopher["name"])}">'

    bio_html = ""
    raw_bio = _load_philosopher_bios().get(philosopher_key, "")
    if raw_bio:
        bio_escaped = html.escape(raw_bio)
        bio_escaped = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', bio_escaped)
        bio_escaped = re.sub(r'\*(.+?)\*', r'<em>\1</em>', bio_escaped)
        bio_html = (
            f'<details class="fighter-bio-details">'
            f'<summary>About {html.escape(philosopher["name"])}</summary>'
            f'<div class="fighter-bio-text">{bio_escaped}</div>'
            f'</details>'
        )

    st.markdown(
        f'<div class="fighter-card">'
        f'{fighter_media}'
        f'<div class="fighter-name">{philosopher["name"]}</div>'
        f'<div class="fighter-side">{side_label}</div>'
        f'<div class="fighter-stance">{philosopher["stance"]}</div>'
        f'{bio_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

def _build_scores_html(scores: dict) -> str:
    """Return a score table as an HTML string.

    Handles both the new metric keys (logical_validity, …) and any legacy or
    unexpected format the model might return — nothing is silently dropped.
    """
    _fallback_labels = {
        "logic": "Logic", "relevance": "Relevance",
        "rebuttal": "Rebuttal", "fairness": "Fairness",
    }
    if not scores:
        return ""

    def score_text(value: object, key: str) -> str:
        if value in (None, ""):
            return "-"
        if isinstance(value, (int, float)):
            denominator = 50 if key == "total" else 10
            number = int(value) if float(value).is_integer() else value
            return f"{number}/{denominator}"
        value_str = str(value).strip()
        if not value_str:
            return "-"
        if "/" in value_str:
            return html.escape(value_str)
        if re.fullmatch(r"\d+(?:[.,]\d+)?", value_str):
            denominator = 50 if key == "total" else 10
            return f"{html.escape(value_str)}/{denominator}"
        return html.escape(value_str)

    metric_keys = list(METRIC_LABELS.keys())
    header_cells = "".join(
        f'<th style="padding:9px 10px;text-align:left;border-bottom:1px solid rgba(255,255,255,0.16);">{label}</th>'
        for label in ["Speaker", *METRIC_LABELS.values()]
    )
    rows = []
    for name, vals in scores.items():
        if not isinstance(vals, dict):
            rows.append(
                f'<tr>'
                f'<td style="padding:9px 10px;font-weight:800;color:white;">{html.escape(str(name))}</td>'
                f'<td colspan="{len(metric_keys)}" style="padding:9px 10px;">{html.escape(str(vals))}</td>'
                f'</tr>'
            )
            continue

        has_new_keys = any(k in METRIC_LABELS for k in vals)
        if has_new_keys:
            # Auto-calculate total if the model didn't provide one
            score_keys = [k for k in metric_keys if k != "total"]
            if not vals.get("total"):
                def _extract_num(v: object) -> float | None:
                    if isinstance(v, (int, float)):
                        return float(v)
                    s = str(v or "").strip()
                    m = re.match(r"(\d+(?:[.,]\d+)?)", s)
                    return float(m.group(1).replace(",", ".")) if m else None
                nums = [_extract_num(vals.get(k)) for k in score_keys]
                if all(n is not None for n in nums):
                    vals = dict(vals)  # don't mutate the original
                    vals["total"] = int(sum(nums))  # type: ignore[assignment]
            cells = "".join(
                f'<td style="padding:9px 10px;">{score_text(vals.get(key), key)}</td>'
                for key in metric_keys
            )
        else:
            # Fall back to whatever the model returned (old or unexpected format)
            fallback_text = "<br>".join(
                f'{html.escape(_fallback_labels.get(k, k.replace("_", " ").title()))}: {html.escape(str(v))}'
                for k, v in vals.items()
            )
            cells = f'<td colspan="{len(metric_keys)}" style="padding:9px 10px;">{fallback_text}</td>'
        rows.append(
            f'<tr style="border-bottom:1px solid rgba(255,255,255,0.10);">'
            f'<td style="padding:9px 10px;font-weight:800;color:white;">{html.escape(str(name))}</td>'
            f'{cells}'
            f'</tr>'
        )

    body_rows = "".join(rows)
    return (
        '<div style="overflow-x:auto;margin-top:12px;">'
        '<table style="width:100%;border-collapse:collapse;color:#e8ecff;font-size:0.9rem;'
        'background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.14);">'
        f'<thead><tr style="color:#ffd54a;background:rgba(255,255,255,0.05);">{header_cells}</tr></thead>'
        f'<tbody>{body_rows}</tbody>'
        '</table>'
        '</div>'
    )


def _format_score_value(score: str, denominator: str) -> str:
    score = score.replace(",", ".").strip()
    try:
        score_num = float(score)
        score = str(int(score_num)) if score_num.is_integer() else str(score_num)
    except ValueError:
        pass
    return f"{score}/{denominator}"


def _speaker_aliases(speaker: str) -> list[str]:
    base_name = re.sub(r"\s*\([^)]*\)", "", speaker).strip()
    aliases = [speaker]
    if base_name and base_name != speaker:
        aliases.append(base_name)
    return aliases


def _alias_pattern(alias: str) -> str:
    # Base-name aliases (no parens) must NOT match role-suffixed names like
    # "Socrates (Against):" — omit the optional-parens group so "Socrates"
    # only matches bare "Socrates:", preventing cross-speaker segment collisions.
    if "(" in alias:
        return rf"{re.escape(alias)}\s*(?:\([^)]*\))?\s*[:\-–—=]"
    return rf"{re.escape(alias)}\s*[:\-–—=]"


def _speaker_score_segment(text: str, speaker: str, speakers: list[str]) -> str:
    start_match = None
    for alias in _speaker_aliases(speaker):
        match = re.search(_alias_pattern(alias), text, re.IGNORECASE)
        if match and (start_match is None or match.start() < start_match.start()):
            start_match = match
    if not start_match:
        return ""

    start = start_match.end()
    end = len(text)
    for other in speakers:
        if other == speaker:
            continue
        for alias in _speaker_aliases(other):
            match = re.search(_alias_pattern(alias), text[start:], re.IGNORECASE)
            if match:
                end = min(end, start + match.start())

    marker = re.search(
        r"\b(?:Begr(?:ü|u)ndung|Justification|Reasoning|Winner)\s*:",
        text[start:],
        re.IGNORECASE,
    )
    if marker:
        end = min(end, start + marker.start())

    return text[start:end]


def _parse_scores_from_segment(segment: str) -> dict:
    metric_aliases = {
        "logical_validity": ["logical validity", "logische gültigkeit", "logische gueltigkeit", "logic"],
        "argument_strength": ["argument strength", "argumentationsstärke", "argumentationsstaerke", "argument strength"],
        "counterargument_handling": ["counterargument handling", "counterargument", "gegenargument"],
        "clarity": ["clarity", "klarheit"],
        "relevance": ["relevance", "relevanz"],
    }
    scores = {}
    for key, aliases in metric_aliases.items():
        for alias in aliases:
            label_re = re.escape(alias).replace(r"\ ", r"\s+")
            patterns = [
                rf"(\d+(?:[\.,]\d+)?)\s*(?:/\s*(10|50))?\s*(?:[-–—:=]|\bfor\b|\bin\b|\bof\b|\()?\s*{label_re}",
                rf"{label_re}\s*(?:[-–—:=]|\bfor\b|\bin\b|\bof\b|\()?\s*(\d+(?:[\.,]\d+)?)\s*(?:/\s*(10|50))?\b",
            ]
            match = None
            for pattern in patterns:
                m = re.search(pattern, segment, re.IGNORECASE)
                if m:
                    score = m.group(1)
                    denom = m.group(2) or "10"
                    scores[key] = _format_score_value(score, denom)
                    match = m
                    break
            if match:
                break

    # Robust total parser allowing totaling/totalling/overall/etc.
    total_match = re.search(
        r"\b(?:total(?:ing|ling|ed)?|overall|gesamt(?:punktzahl)?)\s*(?:[-–—:=]|\bfor\b|\bin\b|\bof\b|\()?\s*(\d+(?:[\.,]\d+)?)\s*(?:/\s*(50|10))?\b",
        segment,
        re.IGNORECASE,
    )
    if not total_match:
        total_match = re.search(
            r"(\d+(?:[\.,]\d+)?)\s*(?:/\s*(50|10))?\s*(?:[-–—:=]|\bfor\b|\bin\b|\bof\b|\()?\s*\b(?:total(?:ing|ling|ed)?|overall|gesamt(?:punktzahl)?)\b",
            segment,
            re.IGNORECASE,
        )
    if total_match:
        scores["total"] = _format_score_value(total_match.group(1), total_match.group(2) or "50")

    # Fallback: if no individual metric scores were found, OR only one was found (which often happens when a single score is followed by a list of metrics like "8/10 for logical validity, argument strength..."),
    # try to assign matches or propagate.
    individual_keys = [k for k in METRIC_LABELS if k != "total"]
    found_keys = [k for k in individual_keys if k in scores]
    
    if len(found_keys) <= 1:
        # Find all numbers in the segment that could be scores
        matches = list(re.finditer(r"(\d+(?:[\.,]\d+)?)(?:\s*/\s*(10|50)\b)?", segment))
        valid_matches = []
        for m in matches:
            val_str = m.group(1).replace(",", ".")
            try:
                val = float(val_str)
                denom = m.group(2) or "10"
                # Exclude total scores if they are explicitly /50
                if denom == "50":
                    continue
                if 0 <= val <= 10:
                    valid_matches.append(m)
            except ValueError:
                pass
        
        # If there's only one unique score number, propagate it to all individual metrics
        if len(valid_matches) == 1:
            single_val = valid_matches[0].group(1)
            for key in individual_keys:
                scores[key] = _format_score_value(single_val, "10")
            if "total" not in scores:
                try:
                    val_num = float(single_val.replace(",", "."))
                    scores["total"] = _format_score_value(str(val_num * 5), "50")
                except ValueError:
                    scores["total"] = _format_score_value(single_val, "10")
        elif len(valid_matches) > 1:
            # Map multiple numbers positionally
            for key, match in zip(individual_keys, valid_matches):
                scores[key] = _format_score_value(match.group(1), "10")

    return scores


def parse_judge_scores(text: str, transcript: list[dict]) -> dict:
    speakers = []
    for turn in transcript:
        speaker = str(turn.get("speaker", "")).strip()
        if speaker and speaker not in speakers:
            speakers.append(speaker)

    parsed = {}
    for speaker in speakers:
        segment = _speaker_score_segment(text, speaker, speakers)
        if not segment:
            continue
        scores = _parse_scores_from_segment(segment)
        if scores:
            parsed[speaker] = scores
    return parsed


def clean_judge_reasoning(text: str, transcript: list[dict]) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL).strip()
    
    # 1. Try robust extraction first: find all Begründung/Justification/Reasoning blocks
    pattern = r"(?is)\b(Begr(?:ü|u)ndung|Justification|Reasoning)\s*:\s*(.*?)(?=\n\s*[A-Z][a-zA-Z\s]+(?:\([^)]*\))?\s*[:\-–—=]|\n\s*Winner\s*:|$)"
    matches = re.findall(pattern, cleaned)
    
    if matches:
        blocks = []
        for _, content in matches:
            content_cleaned = content.strip()
            content_cleaned = re.sub(r"(?is)\n?\s*winner\s*:.*$", "", content_cleaned).strip()
            blocks.append(content_cleaned)
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
            match = re.search(_alias_pattern(alias), cleaned, re.IGNORECASE)
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
                match = re.search(_alias_pattern(alias), cleaned[start:], re.IGNORECASE)
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
        cleaned = cleaned[marker.end():].strip()
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



def render_judge_result_body(judgment: dict) -> None:
    reasoning = judgment.get("reasoning", "")
    scores_html = _build_scores_html(judgment.get("scores", {}))
    reasoning_html = (
        f'<div style="color:#e8ecff;line-height:1.6;font-size:0.95rem;margin-bottom:12px;">'
        f'{_markdown_to_html(reasoning)}'
        f'</div>'
        if reasoning else ""
    )
    st.markdown(f"{reasoning_html}{scores_html}", unsafe_allow_html=True)


def _build_token_usage_html(entries: list[dict]) -> str:
    import html as _html

    if not entries:
        return (
            '<div class="score-card">'
            '<div style="color:#aaa; font-size:0.9rem;">'
            'No token usage metadata was returned by the selected backend.'
            '</div>'
            '</div>'
        )

    total_prompt = sum(int(entry.get("prompt_tokens", 0)) for entry in entries)
    total_completion = sum(int(entry.get("completion_tokens", 0)) for entry in entries)
    total = sum(int(entry.get("total_tokens", 0)) for entry in entries)
    has_estimates = any(entry.get("estimated") for entry in entries)

    def token_label(value: int, estimated: bool = False) -> str:
        prefix = "≈" if estimated else ""
        return f"{prefix}{int(value)}"

    rows = "".join(
        "<tr>"
        f"<td>{_html.escape(str(entry.get('call', 'Model call')))}</td>"
        f"<td>{_html.escape(str(entry.get('model', '')))}</td>"
        f"<td>{token_label(entry.get('prompt_tokens', 0), bool(entry.get('estimated')))}</td>"
        f"<td>{token_label(entry.get('completion_tokens', 0), bool(entry.get('estimated')))}</td>"
        f"<td>{token_label(entry.get('total_tokens', 0), bool(entry.get('estimated')))}</td>"
        "</tr>"
        for entry in entries
    )

    return (
        f'<div class="score-card">'
        f'<div class="score-grid" style="margin-bottom:12px;">'
        f'<div>Prompt: {token_label(total_prompt, has_estimates)}</div>'
        f'<div>Completion: {token_label(total_completion, has_estimates)}</div>'
        f'<div>Total: {token_label(total, has_estimates)}</div>'
        f'<div>Calls: {len(entries)}</div>'
        f'</div>'
        f'<table style="width:100%; border-collapse:collapse; color:#e8ecff; font-size:0.82rem;">'
        f'<thead><tr style="color:#8ecbff; text-align:left;">'
        f'<th style="padding:6px 4px;">Call</th>'
        f'<th style="padding:6px 4px;">Model</th>'
        f'<th style="padding:6px 4px;">Prompt</th>'
        f'<th style="padding:6px 4px;">Completion</th>'
        f'<th style="padding:6px 4px;">Total</th>'
        f'</tr></thead>'
        f'<tbody>{rows}</tbody>'
        f'</table>'
        f'</div>'
    )


def _dev_text(value: object, limit: int = 900) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        text = f"{text[:limit].rstrip()}..."
    return text


def _candidate_score_rows(scores: object) -> list[dict]:
    if not isinstance(scores, dict) or not scores:
        return []

    rows = []
    for candidate_id, metrics in scores.items():
        if isinstance(metrics, dict):
            row = {"Candidate": candidate_id}
            row.update({
                str(key).replace("_", " ").title(): value
                for key, value in metrics.items()
            })
        else:
            row = {"Candidate": candidate_id, "Score": metrics}
        rows.append(row)
    return rows


def render_team_traces_panel(entries: list[dict]) -> None:
    if not entries:
        st.caption("No team reasoning traces were captured for this run.")
        return

    st.markdown("#### Team Reasoning")
    for entry in entries:
        trace = entry.get("trace", {})
        candidates = trace.get("candidates", []) if isinstance(trace, dict) else []
        selection = trace.get("selection", {}) if isinstance(trace, dict) else {}
        reviews = trace.get("reviews", []) if isinstance(trace, dict) else []
        revisions = trace.get("revisions", []) if isinstance(trace, dict) else []
        team_roles = trace.get("team_roles", {}) if isinstance(trace, dict) else {}
        selected = selection.get("selected_candidate", "-") if isinstance(selection, dict) else "-"
        final_score = int(trace.get("final_score", 0) or 0) if isinstance(trace, dict) else 0
        approved = bool(trace.get("approved", False)) if isinstance(trace, dict) else False
        calls = trace.get("calls", []) if isinstance(trace, dict) else []
        status = "approved" if approved else "best effort"
        title = (
            f"{entry.get('speaker', 'Agent')} · Round {int(entry.get('round', 0) or 0)} · "
            f"V3 · {len(calls)} calls · selected {selected} · "
            f"score {final_score}/10 · {status}"
        )

        with st.expander(title, expanded=True):
            if isinstance(team_roles, dict) and team_roles:
                role_summary = " | ".join(
                    f"{role.get('label', key)}: {role.get('philosopher', '-')}"
                    f" / {role.get('strategy', '-')}"
                    for key, role in team_roles.items()
                    if isinstance(role, dict)
                )
                st.caption(role_summary)
            st.markdown("**Candidate reasoning**")
            if candidates:
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    cid = _dev_text(candidate.get("id", "Candidate"), 60)
                    angle = _dev_text(candidate.get("angle", ""), 80)
                    role_name = _dev_text(candidate.get("role_philosopher", ""), 80)
                    role_strategy = _dev_text(candidate.get("role_strategy", ""), 80)
                    summary = _dev_text(candidate.get("plan_summary", ""), 700)
                    role_bits = f" · {role_name}" if role_name else ""
                    strategy_bits = f" · {role_strategy}" if role_strategy else ""
                    st.markdown(f"**{cid}** · {angle}{role_bits}{strategy_bits}")
                    st.write(summary or "No candidate summary captured.")
            else:
                st.caption("No candidate summaries captured.")

            score_rows = _candidate_score_rows(
                selection.get("candidate_scores", {}) if isinstance(selection, dict) else {}
            )
            if score_rows:
                st.table(score_rows)

            selection_reasoning = selection.get("selection_reasoning", "") if isinstance(selection, dict) else ""
            if selection_reasoning:
                st.markdown("**Selection**")
                st.write(_dev_text(selection_reasoning, 900))

            st.markdown("**Review reasoning**")
            if reviews:
                for review in reviews:
                    if not isinstance(review, dict):
                        continue
                    review_status = "approved" if review.get("approved") else "revision requested"
                    st.markdown(
                        f"**Review {int(review.get('attempt', 0) or 0)}** · "
                        f"{int(review.get('score', 0) or 0)}/10 · {review_status}"
                    )
                    st.write(f"Critique: {_dev_text(review.get('critique', ''), 700)}")
                    guidance = _dev_text(review.get("revision_guidance", ""), 700)
                    if guidance:
                        st.write(f"Guidance: {guidance}")
                    reasoning = _dev_text(review.get("reasoning_summary", ""), 700)
                    if reasoning:
                        st.write(f"Reasoning: {reasoning}")
            else:
                st.caption("No review summaries captured.")

            if revisions:
                st.markdown("**Revisions**")
                for revision in revisions:
                    if not isinstance(revision, dict):
                        continue
                    st.markdown(f"**Revision {int(revision.get('attempt', 0) or 0)}**")
                    role_name = _dev_text(revision.get("role_philosopher", ""), 80)
                    role_strategy = _dev_text(revision.get("role_strategy", ""), 80)
                    if role_name or role_strategy:
                        st.caption(f"{role_name} / {role_strategy}")
                    st.write(_dev_text(revision.get("plan_summary", ""), 700))


def render_development_panel() -> None:
    if not st.session_state.get("_pref_developer_mode", False):
        return

    team_mode = bool(st.session_state.get("team_mode_active", False))
    team_status = "Enabled" if team_mode else "Disabled"
    
    # Only show the internals label if we are in team mode
    internals_label = " • Team Internals Active" if team_mode else ""

    # Build the panel HTML carefully to avoid markdown triggers
    # We use a very simple structure to ensure it doesn't break
    panel_html = (
        f'<div class="arcade-panel" style="margin-top:8px; margin-bottom:18px;">'
        f'<div class="topic-label">Development{internals_label}</div>'
        f'<div style="color:#aaa; font-size:0.9rem; margin-bottom:12px;">'
        f'Team Mode: <span style="color:#fff;">{team_status}</span>'
        f'</div>'
        f'{_build_token_usage_html(st.session_state.get("token_usage", []))}'
        f'</div>'
    )
    st.markdown(panel_html, unsafe_allow_html=True)
    if team_mode:
        render_team_traces_panel(st.session_state.get("team_traces", []))


def _html_text(value: object) -> str:
    return html.escape(str(value)).replace("\n", "<br>")

def _markdown_to_html(text: str) -> str:
    text_str = str(text)
    try:
        from markdown_it import MarkdownIt
        md = MarkdownIt("commonmark", {"breaks": True, "html": False})
        return md.render(text_str)
    except ImportError:
        return html.escape(text_str).replace("\n", "<br>")


def _json_for_script(value: object) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def render_debate_text_panel(topic: str, transcript: list[dict] | None) -> None:
    st.markdown(
        f"""
        <div class="arcade-panel" style="margin-bottom:18px;">
        """,
        unsafe_allow_html=True,
    )

    if transcript:
        for turn in transcript:
            st.markdown(
                f"""
                <div class="score-card">
                    <div class="score-name">{_html_text(turn['speaker'])}</div>
                    <div class="turn-text" style="color:#e8ecff; line-height:1.6;">{_markdown_to_html(turn['text'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="score-card">
                <div style="color:#e8ecff; line-height:1.6;">Debate is being prepared...</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_team_character_selection() -> tuple[str, str]:
    _ensure_team_mode_defaults()
    philosopher_names = [v["name"] for v in PHILOSOPHER_LIBRARY.values()]
    st.markdown("**Team Philosopher Selection**")

    tabs = st.tabs([TEAM_SIDE_LABELS[side] for side in TEAM_SIDE_KEYS])
    for side, tab in zip(TEAM_SIDE_KEYS, tabs):
        with tab:
            role_cols = st.columns(len(TEAM_ROLE_DEFS))
            for role_col, (role_key, role_label, fallback_key) in zip(role_cols, TEAM_ROLE_DEFS):
                with role_col:
                    philosopher_key = _team_widget_key(side, role_key, "philosopher")
                    strategy_key = _team_widget_key(side, role_key, "strategy")
                    current_name = _valid_philosopher_name(
                        st.session_state.get(philosopher_key),
                        fallback_key,
                    )
                    current_strategy = st.session_state.get(strategy_key, STRATEGY_OPTIONS[0])
                    if current_strategy not in STRATEGY_OPTIONS:
                        current_strategy = STRATEGY_OPTIONS[0]
                    selected_name = st.selectbox(
                        f"{role_label} Philosopher",
                        options=philosopher_names,
                        index=philosopher_names.index(current_name),
                        key=philosopher_key,
                    )
                    st.selectbox(
                        f"{role_label} Strategy",
                        options=STRATEGY_OPTIONS,
                        index=STRATEGY_OPTIONS.index(current_strategy),
                        key=strategy_key,
                    )
                    selected_key = _philosopher_key_from_name(selected_name, fallback_key)
                    render_fighter_card(
                        selected_key,
                        f"{role_label} • {TEAM_SIDE_LABELS[side]}",
                    )

    team_configs = _team_configs_from_state()
    agent1_name = team_configs["for"]["reviewer"]["philosopher_name"]
    agent2_name = team_configs["against"]["reviewer"]["philosopher_name"]
    st.session_state["agent1_philosopher_name"] = agent1_name
    st.session_state["agent2_philosopher_name"] = agent2_name
    st.session_state["ui_agent1_philosopher_name"] = agent1_name
    st.session_state["ui_agent2_philosopher_name"] = agent2_name
    st.session_state["player1_strategy"] = team_configs["for"]["reviewer"]["strategy"]
    st.session_state["player2_strategy"] = team_configs["against"]["reviewer"]["strategy"]
    st.session_state["_pref_team_configs"] = team_configs
    return agent1_name, agent2_name


def render_free_team_character_selection() -> str:
    _ensure_team_mode_defaults()
    side = "free"
    philosopher_names = [v["name"] for v in PHILOSOPHER_LIBRARY.values()]
    st.markdown("**Free Topic Philosopher Team**")

    role_cols = st.columns(len(TEAM_ROLE_DEFS))
    for role_col, (role_key, role_label, fallback_key) in zip(role_cols, TEAM_ROLE_DEFS):
        with role_col:
            philosopher_key = _team_widget_key(side, role_key, "philosopher")
            strategy_key = _team_widget_key(side, role_key, "strategy")
            current_name = _valid_philosopher_name(
                st.session_state.get(philosopher_key),
                fallback_key,
            )
            current_strategy = st.session_state.get(strategy_key, STRATEGY_OPTIONS[0])
            if current_strategy not in STRATEGY_OPTIONS:
                current_strategy = STRATEGY_OPTIONS[0]
            selected_name = st.selectbox(
                f"{role_label} Philosopher",
                options=philosopher_names,
                index=philosopher_names.index(current_name),
                key=philosopher_key,
            )
            st.selectbox(
                f"{role_label} Strategy",
                options=STRATEGY_OPTIONS,
                index=STRATEGY_OPTIONS.index(current_strategy),
                key=strategy_key,
            )
            selected_key = _philosopher_key_from_name(selected_name, fallback_key)
            render_fighter_card(selected_key, f"{role_label} • {TEAM_SIDE_LABELS[side]}")

    team_configs = _team_configs_from_state()
    agent_name = team_configs["free"]["reviewer"]["philosopher_name"]
    st.session_state["agent1_philosopher_name"] = agent_name
    st.session_state["agent2_philosopher_name"] = ""
    st.session_state["ui_agent1_philosopher_name"] = agent_name
    st.session_state["player1_strategy"] = team_configs["free"]["reviewer"]["strategy"]
    st.session_state["_pref_team_configs"] = team_configs
    return agent_name


def _save_start_preferences(philosopher_teams: bool, start_mode: str) -> None:
    if philosopher_teams:
        _reset_team_mode_defaults()
    st.session_state["philosopher_teams"] = philosopher_teams
    st.session_state["start_mode"] = start_mode
    st.session_state["_pref_start_mode"] = start_mode
    # summary is always enabled — model preference still tracked for model selector on character stage
    st.session_state["_pref_summary_model"] = st.session_state.get("summary_model_label", list(MODEL_OPTIONS.keys())[0])
    st.session_state["_pref_agent_max_words"] = _valid_agent_max_words(
        st.session_state.get("agent_max_words")
    )
    st.session_state["_pref_rounds"] = _valid_argument_rounds(
        st.session_state.get("rounds")
    )
    st.session_state["_pref_developer_mode"] = st.session_state.get("developer_mode", False)
    st.session_state["_pref_audio_transcription"] = _session_bool(
        "audio_transcription",
        default=False,
    )
    st.session_state["_pref_speaker_speed"] = float(
        st.session_state.get("speaker_speed", 1.0) or 1.0
    )
    st.session_state["_pref_philosopher_teams"] = philosopher_teams
    st.session_state["_pref_team_max_review_attempts"] = int(
        st.session_state.get("team_max_review_attempts", 3)
    )
    st.session_state["_pref_team_configs"] = _team_configs_from_state()


def render_start_screen() -> None:
    st.markdown(
        """
        <div class="arcade-panel" style="text-align:center;">
            <div style="font-size:2rem; font-weight:800; margin-bottom:1rem;">Choose Game Mode</div>
            <div style="font-size:1.1rem; color:#cfd8ff; margin-bottom:0.7rem;">Select your debate format</div>
            <div style="font-size:0.95rem; font-weight:800; color:#ffd54a; text-transform:uppercase; letter-spacing:0.08em;">Debate engine ready</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    mode_col1, mode_col2, mode_col3 = st.columns(3)
    with mode_col1:
        if st.button("Single-Philosopher Mode", type="primary", use_container_width=True):
            _save_start_preferences(False, "single")
            st.session_state["stage"] = 1
            st.rerun()
    with mode_col2:
        if st.button("Multi-Philosopher Mode", type="primary", use_container_width=True):
            _save_start_preferences(True, "team")
            st.session_state["stage"] = 1
            st.rerun()
    with mode_col3:
        if st.button("Free Topic Mode", type="primary", use_container_width=True):
            _save_start_preferences(True, "free")
            st.session_state["stage"] = 1
            st.rerun()

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    with st.expander("Settings", expanded=False):
        setting_col1, setting_col2 = st.columns([1, 2])
        with setting_col1:
            model_keys = list(MODEL_OPTIONS.keys())
            _local_model_key = next(
                (k for k, (p, _) in MODEL_OPTIONS.items() if p == "local"),
                model_keys[0],
            )
            def_model = st.session_state.get(
                "summary_model_label",
                st.session_state.get("_pref_summary_model", _local_model_key),
            )
            if def_model not in model_keys:
                def_model = _local_model_key
            st.selectbox(
                "Summary Model",
                options=model_keys,
                index=model_keys.index(def_model),
                key="summary_model_label",
                help="Model to use for the written summary.",
            )
            audio_transcription = st.checkbox(
                "Audio transcription",
                key="audio_transcription",
                help="Turn spoken text-to-speech output for debate and judge responses on or off.",
            )
            st.slider(
                "Speaker Speed",
                min_value=0.5,
                max_value=2.0,
                value=float(st.session_state.get("speaker_speed", 1.0) or 1.0),
                step=0.1,
                key="speaker_speed",
                help="Playback speed for the text-to-speech audio output (0.5× = slow, 1.0× = normal, 2.0× = fast).",
                format="%.1f×",
                disabled=not audio_transcription,
            )
            st.checkbox(
                "Developer mode",
                key="developer_mode",
                help="Show development diagnostics such as token usage after a run.",
            )
        with setting_col2:
            if st.session_state.get("developer_mode", False):
                st.slider(
                    "Max philosopher words",
                    min_value=MIN_AGENT_MAX_WORDS,
                    max_value=MAX_AGENT_MAX_WORDS,
                    value=int(st.session_state.get("agent_max_words", DEFAULT_AGENT_MAX_WORDS)),
                    step=10,
                    key="agent_max_words",
                )
                st.slider(
                    "Argument rounds",
                    min_value=MIN_ARGUMENT_ROUNDS,
                    max_value=MAX_ARGUMENT_ROUNDS,
                    value=int(st.session_state.get("rounds", DEFAULT_ARGUMENT_ROUNDS)),
                    step=1,
                    key="rounds",
                    help="Each round gives both philosophers one argument.",
                )
                st.slider(
                    "Max review attempts (Team Mode)",
                    min_value=1,
                    max_value=5,
                    value=int(st.session_state.get("team_max_review_attempts", 3)),
                    step=1,
                    key="team_max_review_attempts",
                    help="Max attempts the critic reviews and requests revision of the philosopher's draft.",
                )


def render_topic_stage() -> None:
    if _current_start_mode() == "free":
        st.markdown(
            """
            <div class="topic-panel">
                <div class="topic-label">Free Topic</div>
                <div class="topic-text">Enter the topic your philosopher team should explore.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("free_topic_form"):
            topic_text = st.text_area(
                "Topic",
                key="free_topic_input",
                height=120,
                placeholder="Example: What does authenticity mean in an age of artificial intelligence?",
            )
            submitted = st.form_submit_button("Continue", type="primary", use_container_width=True)

        if submitted:
            clean_topic = topic_text.strip()
            if clean_topic:
                st.session_state["selected_topic"] = clean_topic
                st.session_state["_character_stage_seeded"] = False
                st.session_state["stage"] = 2
                st.rerun()
            st.warning("Please enter a topic before continuing.")
        return

    render_topic_panel(st.session_state["selected_topic"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Redraw Topic", use_container_width=True):
            st.session_state["selected_topic"] = random_topic()
            st.rerun()
    with col2:
        if st.button("Continue", type="primary", use_container_width=True):
            st.session_state["_character_stage_seeded"] = False
            st.session_state["stage"] = 2
            st.rerun()


def render_character_stage() -> None:
    render_topic_panel(st.session_state["selected_topic"])

    model_keys   = list(MODEL_OPTIONS.keys())
    
    judge_keys   = ["judge1_model_label", "judge2_model_label", "judge3_model_label"]
    judge_labels = ["⚖️ Judge 1 Model", "⚖️ Judge 2 Model", "⚖️ Judge 3 Model"]
    start_mode = _current_start_mode()
    free_topic_mode = start_mode == "free"

    # ── Philosophers ──────────────────────────────────────────────────────────
    philosopher_teams = bool(
        st.session_state.get("_pref_philosopher_teams")
        or st.session_state.get("philosopher_teams", False)
    )
    if free_topic_mode:
        agent1_name = render_free_team_character_selection()
        agent2_name = ""
    elif philosopher_teams:
        agent1_name, agent2_name = render_team_character_selection()
    else:
        philosopher_names = [v["name"] for v in PHILOSOPHER_LIBRARY.values()]
        _seed_character_stage_widgets()

        col_a, col_b = st.columns(2)
        with col_a:
            agent1_name = st.selectbox(
                "Player 1 Philosopher",
                options=philosopher_names,
                key="ui_agent1_philosopher_name",
            )
        with col_b:
            agent2_name = st.selectbox(
                "Player 2 Philosopher",
                options=philosopher_names,
                key="ui_agent2_philosopher_name",
            )

        st.session_state["agent1_philosopher_name"] = agent1_name
        st.session_state["agent2_philosopher_name"] = agent2_name

        strategy_col1, strategy_col2 = st.columns(2)
        with strategy_col1:
            st.selectbox("Player 1 Strategy", options=STRATEGY_OPTIONS, key="player1_strategy")
        with strategy_col2:
            st.selectbox("Player 2 Strategy", options=STRATEGY_OPTIONS, key="player2_strategy")

        philosopher_options = {v["name"]: k for k, v in PHILOSOPHER_LIBRARY.items()}
        agent1_key = philosopher_options[_valid_philosopher_name(agent1_name, "socrates")]
        agent2_key = philosopher_options[_valid_philosopher_name(agent2_name, "nietzsche")]

        card_col1, card_col2 = st.columns(2)
        with card_col1:
            render_fighter_card(agent1_key, "Player 1 • For")
        with card_col2:
            render_fighter_card(agent2_key, "Player 2 • Against")

    # ── Debater models ────────────────────────────────────────────────────────
    st.markdown("**Debater Models**")
    selected_judge_model_labels: list[str] = []
    current_agent1_model = _valid_model_label(st.session_state.get("agent1_model_label"))
    current_agent2_model = _valid_model_label(st.session_state.get("agent2_model_label"))
    if free_topic_mode:
        agent1_model_label = st.selectbox(
            "🤖 Free Topic Team Model",
            options=model_keys,
            index=model_keys.index(current_agent1_model),
            key="agent1_model_label",
            help="Model used by the free topic philosopher team.",
        )
        agent2_model_label = _valid_model_label(
            st.session_state.get("agent2_model_label"), agent1_model_label
        )
    else:
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            agent1_model_label = st.selectbox(
                "🤖 Player 1 Model",
                options=model_keys,
                index=model_keys.index(current_agent1_model),
                key="agent1_model_label",
                help="Model used by the FOR debater.",
            )
        with dcol2:
            agent2_model_label = st.selectbox(
                "🤖 Player 2 Model",
                options=model_keys,
                index=model_keys.index(current_agent2_model),
                key="agent2_model_label",
                help="Model used by the AGAINST debater.",
            )

    # ── Judge panel ───────────────────────────────────────────────────────────
    if free_topic_mode:
        num_judges = 0
        st.session_state["num_judges"] = 0
        st.caption("Free Topic Mode uses one philosopher team, so no pro/contra judge is shown.")
    else:
        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
        st.markdown("**Judge Panel**")
        num_judges = st.radio(
            "Number of Judges",
            options=[1, 2, 3],
            format_func=lambda n: f"{n} Judge{'s' if n > 1 else ''}",
            horizontal=True,
            key="num_judges",
        )
        focus_hints = [
            "General evaluation across all metrics.",
            "2 judges: Logic & Reasoning focus.",
            "3 judges: Clarity & Communication focus.",
        ]
        jcols = st.columns(num_judges)
        for i in range(num_judges):
            current_judge_model = _valid_model_label(st.session_state.get(judge_keys[i]))
            with jcols[i]:
                selected_judge_model_labels.append(st.selectbox(
                    judge_labels[i],
                    options=model_keys,
                    index=model_keys.index(current_judge_model),
                    key=judge_keys[i],
                    help=JUDGE_FOCUS[num_judges][i] or focus_hints[i],
                ))

    # ── Live model lineup (uses the same values returned by the widgets above) ─
    if free_topic_mode:
        rows = (
            f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
            f'<span style="color:#e8ecff;">{agent1_name} '
            f'<span style="color:#aaa;font-size:0.8rem;">(Free Topic Team)</span></span>'
            f'<span style="color:#ffd54a;font-size:0.85rem;">{agent1_model_label}</span>'
            f'</div>'
        )
        team_configs = _team_configs_from_state()
        rows += (
            f'<div style="margin-top:8px;color:#8ecbff;font-size:0.72rem;'
            f'text-transform:uppercase;letter-spacing:0.06em;">{TEAM_SIDE_LABELS["free"]}</div>'
        )
        for role_key, role_label, _fallback_key in TEAM_ROLE_DEFS:
            role = team_configs["free"][role_key]
            rows += (
                f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
                f'<span style="color:#e8ecff;">{role_label}: {role["philosopher_name"]}</span>'
                f'<span style="color:#ffd54a;font-size:0.85rem;">{role["strategy"]}</span>'
                f'</div>'
            )
    else:
        judge_roles = JUDGE_ROLE_LABELS[num_judges]
        rows = (
            f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
            f'<span style="color:#e8ecff;">{agent1_name} <span style="color:#aaa;font-size:0.8rem;">(Pro)</span></span>'
            f'<span style="color:#ffd54a;font-size:0.85rem;">{agent1_model_label}</span>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
            f'<span style="color:#e8ecff;">{agent2_name} <span style="color:#aaa;font-size:0.8rem;">(Contra)</span></span>'
            f'<span style="color:#ffd54a;font-size:0.85rem;">{agent2_model_label}</span>'
            f'</div>'
        )
    if philosopher_teams and not free_topic_mode:
        team_configs = _team_configs_from_state()
        for side in TEAM_SIDE_KEYS:
            rows += (
                f'<div style="margin-top:8px;color:#8ecbff;font-size:0.72rem;'
                f'text-transform:uppercase;letter-spacing:0.06em;">{TEAM_SIDE_LABELS[side]}</div>'
            )
            for role_key, role_label, _fallback_key in TEAM_ROLE_DEFS:
                role = team_configs[side][role_key]
                rows += (
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
                    f'<span style="color:#e8ecff;">{role_label}: {role["philosopher_name"]}</span>'
                    f'<span style="color:#ffd54a;font-size:0.85rem;">{role["strategy"]}</span>'
                    f'</div>'
                )
    if not free_topic_mode:
        for i in range(num_judges):
            rows += (
                f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
                f'<span style="color:#e8ecff;">{judge_roles[i]} Judge</span>'
                f'<span style="color:#ffd54a;font-size:0.85rem;">{selected_judge_model_labels[i]}</span>'
                f'</div>'
            )
    st.markdown(
        f"""
        <div class="score-card" style="margin-top:14px;margin-bottom:4px;padding:10px 16px;">
            <div style="font-size:0.7rem;color:#aaa;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">
                Confirmed Lineup
            </div>
            {rows}
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back", use_container_width=True):
            st.session_state["stage"] = 1
            st.rerun()
    with col2:
        if st.button("Continue", type="primary", use_container_width=True):
            # Snapshot all widget values to stable keys NOW, while they are live.
            # Going to stage 3 will cause Streamlit to disown these widget-bound keys,
            # but the _dp_ snapshot keys are unaffected by setdefault.
            _save_debate_params(
                agent1_name,
                agent2_name,
                agent1_model_label,
                agent2_model_label,
                selected_judge_model_labels,
            )
            st.session_state["stage"] = 3
            st.rerun()


def render_winner_announcement() -> None:
    judge_results = st.session_state.get("judge_results", [])
    if not judge_results:
        return
        
    num_judges = st.session_state.get("num_judges", 1)
    if len(judge_results) < num_judges:
        return
    for jr in judge_results:
        if jr is None or (isinstance(jr, dict) and jr.get("judgment") is None):
            return
            
    judgment = st.session_state.get("judgment")
    if not judgment or not isinstance(judgment, dict):
        return
        
    winner = judgment.get("winner", "N/A")
    if not winner or winner in ("N/A", "Error", "", "Draw"):
        winner = "Tie / Split Decision"
        
    icons_html = '<div style="display: flex; align-items: center; margin-right: 15px;">'
    for i in range(num_judges):
        jr = judge_results[i] if i < len(judge_results) else {}
        role = jr.get('role', 'General Evaluation') if isinstance(jr, dict) else 'General Evaluation'
        img_path = get_judge_image_path(role)
        try:
            img_src = image_to_data_uri(img_path)
        except Exception:
            img_src = ""
        margin_left = "0px" if i == 0 else "-12px"
        if img_src:
            icons_html += (
                f'<img src="{img_src}" style="'
                f'height: 38px; width: 38px; border-radius: 50%; '
                f'border: 2px solid #ffd54a; background: #1a1f35; '
                f'margin-left: {margin_left}; z-index: {10 - i};'
                f'" />'
            )
    icons_html += '</div>'
    
    banner_html = (
        f'<div class="winner-banner" style="display: flex; align-items: center; '
        f'padding: 16px 20px; border-radius: 18px; margin-top: 15px; margin-bottom: 15px;">'
        f'{icons_html}'
        f'<div style="flex-grow: 1;">'
        f'<div style="font-size: 0.72rem; text-transform: uppercase; color: #ffd54a; '
        f'letter-spacing: 1.5px; font-weight: 700; margin-bottom: 2px;">Debate Winner</div>'
        f'<div style="font-size: 1.35rem; font-weight: 900; color: white;'
        f'text-transform: uppercase; letter-spacing: 0.5px;">{html.escape(winner)}</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(banner_html, unsafe_allow_html=True)


def render_generation_snapshot(include_transcript: bool = True) -> None:
    transcript = st.session_state.get("transcript", []) or []
    judge_results = st.session_state.get("judge_results", []) or []
    agent_configs = st.session_state.get("agent_configs", []) or []

    avatar_by_speaker = {}
    for cfg in agent_configs:
        philosopher = PHILOSOPHER_LIBRARY.get(cfg.get("philosopher_key", ""))
        if not philosopher:
            continue
        display_name = cfg.get("display_name") or philosopher["name"]
        avatar_by_speaker[display_name] = philosopher.get("image")

    if include_transcript:
        for turn in transcript:
            speaker = turn.get("speaker", "Agent")
            avatar = turn.get("avatar") or avatar_by_speaker.get(speaker)
            if avatar:
                message = st.chat_message(speaker, avatar=avatar)
            else:
                message = st.chat_message(speaker)
            with message:
                st.markdown(turn.get("text", ""))

    for index, jr in enumerate(judge_results, start=1):
        if not isinstance(jr, dict):
            continue
        judge_role = jr.get('role', 'General Evaluation')
        judge_label = f"Judge {index} · {judge_role} · {jr.get('model_label', '')}"
        judgment = jr.get("judgment", {}) if isinstance(jr.get("judgment"), dict) else {}
        with st.chat_message(judge_label, avatar=get_judge_image_path(judge_role)):
            display_judgment = dict(judgment)
            if not display_judgment.get("scores"):
                display_judgment["scores"] = parse_judge_scores(
                    display_judgment.get("reasoning", ""),
                    transcript,
                )
            display_judgment["reasoning"] = clean_judge_reasoning(
                display_judgment.get("reasoning", ""),
                transcript,
            )
            render_judge_result_body(display_judgment)

    render_winner_announcement()


def render_generation_hold_actions() -> None:
    render_development_panel()
    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    if st.button("Show Summary", type="primary", use_container_width=True):
        st.session_state["stage"] = 5
        st.rerun()


def render_versus_stage() -> None:
    configs = current_agent_configs(use_saved_params=True)
    st.session_state["agent1_philosopher_name"] = configs[0]["philosopher_name"]
    st.session_state["agent2_philosopher_name"] = (
        configs[1]["philosopher_name"] if len(configs) > 1 else ""
    )
    render_topic_panel(st.session_state["selected_topic"], small=True)


    if "transcript" not in st.session_state:
        _clear_stale_stage_tail()
        handle_run_debate()
        st.session_state.pop("_loading_screen_primed", None)
        if st.session_state.get("team_mode_active", False):
            render_generation_snapshot(include_transcript=False)
    else:
        render_generation_snapshot()

    render_generation_hold_actions()


def extract_winner(text: str, transcript: list) -> str:
    speakers = []
    for turn in transcript:
        s = turn.get("speaker")
        if s and s not in speakers:
            speakers.append(s)

    match = re.search(r"(?:winner|gewinner):\s*\*?\*?([^\n\*\#]+)", text, re.IGNORECASE)
    if match:
        winner_candidate = match.group(1).strip()
        # Strip only markdown formatting — keep parentheses so role suffixes like
        # "(For)" / "(Against)" survive and distinguish same-philosopher speakers.
        winner_candidate = re.sub(r"[\*\#]+", "", winner_candidate).strip()
        # Pass through an explicit Draw verdict immediately.
        if winner_candidate.lower() in ("draw", "unentschieden", "tie"):
            return "Draw"
        # Pick the speaker whose longest alias matches the candidate; longest alias
        # wins so "Wittgenstein (Against)" beats bare "Wittgenstein".
        best_speaker = None
        best_len = -1
        for speaker in speakers:
            for alias in _speaker_aliases(speaker):
                a, c = alias.lower(), winner_candidate.lower()
                if a in c or c in a:
                    if len(alias) > best_len:
                        best_len = len(alias)
                        best_speaker = speaker
        if best_speaker:
            return best_speaker

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

def handle_run_debate() -> None:
    default_m = list(MODEL_OPTIONS.keys())[0]

    # Prefer the stable _dp_ snapshot keys; fall back to display/widget keys if
    # called directly from render_arena_stage's safety net.
    start_mode         = _current_start_mode()
    free_topic_mode    = start_mode == "free"
    saved_configs      = current_agent_configs(use_saved_params=True)
    agent1_name        = saved_configs[0]["philosopher_name"]
    agent2_name        = saved_configs[1]["philosopher_name"] if len(saved_configs) > 1 else ""
    agent1_model_label = st.session_state.get("_dp_model1") or st.session_state.get("agent1_model_label", default_m)
    agent2_model_label = st.session_state.get("_dp_model2") or st.session_state.get("agent2_model_label", default_m)
    p1_strategy        = st.session_state.get("_dp_strat1") or st.session_state.get("player1_strategy", STRATEGY_OPTIONS[0])
    p2_strategy        = st.session_state.get("_dp_strat2") or st.session_state.get("player2_strategy", STRATEGY_OPTIONS[0])
    num_judges         = st.session_state.get("_dp_num_judges") or st.session_state.get("num_judges", 1)
    if free_topic_mode:
        num_judges = 0
    argument_rounds    = _valid_argument_rounds(
        st.session_state.get("_dp_rounds")
        or st.session_state.get("_pref_rounds")
        or st.session_state.get("rounds")
    )
    if free_topic_mode:
        argument_rounds = 1
    agent_max_words    = _valid_agent_max_words(
        st.session_state.get("_dp_agent_max_words")
        or st.session_state.get("_pref_agent_max_words")
        or st.session_state.get("agent_max_words")
    )
    developer_mode     = bool(
        st.session_state.get("_dp_developer_mode")
        or st.session_state.get("_pref_developer_mode")
        or st.session_state.get("developer_mode", False)
    )
    audio_transcription = _session_bool(
        "_dp_audio_transcription",
        "_pref_audio_transcription",
        "audio_transcription",
        default=False,
    )
    audio_output       = audio_transcription
    audio_autoplay     = True
    speaker_speed      = float(
        st.session_state.get("_dp_speaker_speed")
        or st.session_state.get("_pref_speaker_speed")
        or st.session_state.get("speaker_speed", 1.0)
        or 1.0
    )
    philosopher_teams  = bool(
        free_topic_mode
        or
        st.session_state.get("_dp_philosopher_teams")
        or st.session_state.get("_pref_philosopher_teams")
        or st.session_state.get("philosopher_teams", False)
    )
    team_configs       = (
        st.session_state.get("_dp_team_configs")
        or st.session_state.get("_pref_team_configs")
        or _team_configs_from_state()
    )
    max_review_attempts = int(
        st.session_state.get("_dp_max_review_attempts")
        or st.session_state.get("_pref_team_max_review_attempts")
        or st.session_state.get("team_max_review_attempts", 3)
    )

    # Write back resolved display values early so any subsequent UI renders (e.g. Winner Announcement icons) see the correct counts
    st.session_state["agent1_philosopher_name"] = agent1_name
    st.session_state["agent2_philosopher_name"] = agent2_name
    st.session_state["agent1_model_label"]      = agent1_model_label
    st.session_state["agent2_model_label"]      = agent2_model_label
    st.session_state["num_judges"]              = num_judges
    st.session_state["rounds"]                  = argument_rounds
    st.session_state["agent_max_words"]         = agent_max_words
    st.session_state["developer_mode"]          = developer_mode
    st.session_state["audio_transcription"]     = audio_transcription
    st.session_state["speaker_speed"]           = speaker_speed
    st.session_state["philosopher_teams"]       = philosopher_teams

    agent1_provider, agent1_model = MODEL_OPTIONS[agent1_model_label]
    if free_topic_mode:
        free_team_config = dict(team_configs.get("free", {}))
        free_team_config["max_review_attempts"] = max_review_attempts
        agent_configs = [
            {
                "philosopher_key": saved_configs[0]["philosopher_key"],
                "philosopher_name": agent1_name,
                "provider":    agent1_provider,
                "model":       agent1_model,
                "model_label": agent1_model_label,
                "display_name": f"{agent1_name} (Free Topic)",
                "side":        "Free Topic",
                "goal":        "Give an opinion on the input, using pro and contra angles only when they fit.",
                "team_config": free_team_config,
            },
        ]
        player_strategies = [p1_strategy]
    else:
        agent2_provider, agent2_model = MODEL_OPTIONS[agent2_model_label]
        for_team_config = dict(team_configs.get("for", {}))
        for_team_config["max_review_attempts"] = max_review_attempts
        against_team_config = dict(team_configs.get("against", {}))
        against_team_config["max_review_attempts"] = max_review_attempts
        agent_configs = [
            {
                "philosopher_key": saved_configs[0]["philosopher_key"],
                "philosopher_name": agent1_name,
                "provider":    agent1_provider,
                "model":       agent1_model,
                "model_label": agent1_model_label,
                "display_name": f"{agent1_name} (Pro)",
                "side":        "Pro",
                "team_config": for_team_config,
            },
            {
                "philosopher_key": saved_configs[1]["philosopher_key"],
                "philosopher_name": agent2_name,
                "provider":    agent2_provider,
                "model":       agent2_model,
                "model_label": agent2_model_label,
                "display_name": f"{agent2_name} (Contra)",
                "side":        "Contra",
                "team_config": against_team_config,
            },
        ]
        player_strategies = [p1_strategy, p2_strategy]

    reset_token_usage()

    transcript = []
    
    if philosopher_teams:
        team_status = st.empty()
        team_message_slots: dict[tuple[int, str, str], object] = {}
        team_live_text: dict[tuple[int, str, str], str] = {}

        def clean_team_live_text(text: str) -> str:
            return re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL).strip()

        def ensure_team_message(agent, round_idx: int):
            side = getattr(agent, "side", "")
            key = (round_idx, side, agent.name)
            if key not in team_message_slots:
                with st.chat_message(agent.name, avatar=agent.image):
                    team_message_slots[key] = st.empty()
                team_live_text[key] = ""
            return key, team_message_slots[key]

        def render_team_step(message: str) -> None:
            team_status.caption(message)

        def reset_team_draft(agent, round_idx: int, attempt: int) -> None:
            key, slot = ensure_team_message(agent, round_idx)
            team_live_text[key] = ""
            slot.markdown(f"_Draft {attempt} is being generated..._")

        def stream_team_token(token: str, agent, round_idx: int) -> None:
            key, slot = ensure_team_message(agent, round_idx)
            team_live_text[key] += token
            display_text = clean_team_live_text(team_live_text[key])
            slot.markdown(display_text or " ")

        for item in run_debate(
            st.session_state["selected_topic"],
            argument_rounds,
            player_strategies,
            agent_configs,
            max_words=agent_max_words,
            team_mode=philosopher_teams,
            on_step=render_team_step,
            on_token=stream_team_token,
            on_draft_start=reset_team_draft,
        ):
            if item.get("type") == "team":
                turn = item["turn"]
                agent = item.get("agent")
                if agent:
                    key, slot = ensure_team_message(agent, turn.get("round", 0))
                    team_live_text[key] = turn.get("text", "")
                    slot.markdown(
                        clean_team_live_text(team_live_text[key])
                        or "_No argument generated._"
                    )
                transcript = item["transcript"]

        team_status.empty()
    else:
        # Real-time streaming for single mode!

        for item in run_debate(
            st.session_state["selected_topic"],
            argument_rounds,
            player_strategies,
            agent_configs,
            max_words=agent_max_words,
            team_mode=philosopher_teams,
            on_step=None
        ):
            if item.get("type") == "single":
                agent = item["agent"]
                with st.chat_message(agent.name, avatar=agent.image):
                    st.write_stream(item["generator"])
                transcript = item["transcript"]

    team_traces = [
        {
            "round": turn.get("round", 0),
            "speaker": turn.get("speaker", "Agent"),
            "trace": turn.get("team_trace", {}),
        }
        for turn in transcript
        if turn.get("team_trace")
    ]

    judge_results: list = []
    judgments: list = []
    if free_topic_mode:
        judgment = None
    else:
        focuses     = JUDGE_FOCUS[num_judges]
        role_labels = JUDGE_ROLE_LABELS[num_judges]

        judgments    = [None] * num_judges
        judge_results = [None] * num_judges

        selected_topic = st.session_state["selected_topic"]

        if philosopher_teams:
            # In team mode judges run silently; render_generation_snapshot() displays
            # transcript → judges → winner in the correct order after this function returns.
            with st.spinner("Judges are evaluating the debate..."):
                for i in range(num_judges):
                    j_model_label = (
                        st.session_state.get(f"_dp_judge{i+1}_model")
                        or st.session_state.get(f"judge{i+1}_model_label", default_m)
                    )
                    j_provider, j_model = MODEL_OPTIONS[j_model_label]
                    focus = focuses[i]
                    role  = role_labels[i]
                    try:
                        j_text = "".join(list(judge_debate(
                            selected_topic, transcript,
                            judge_provider=j_provider,
                            judge_model=j_model,
                            focus=focus,
                        )))
                        winner = extract_winner(j_text, transcript)
                        scores = parse_judge_scores(j_text, transcript)
                        reasoning = clean_judge_reasoning(j_text, transcript)
                        j = {"winner": winner, "reasoning": reasoning, "scores": scores}
                    except Exception as exc:
                        j = {"winner": "Error", "reasoning": f"Error running judge: {exc}", "scores": {}}
                    judgments[i] = j
                    judge_results[i] = {"judgment": j, "model_label": j_model_label, "role": role}
        else:
            # Single mode: stream judges live like the philosopher messages.
            for i in range(num_judges):
                j_model_label = (
                    st.session_state.get(f"_dp_judge{i+1}_model")
                    or st.session_state.get(f"judge{i+1}_model_label", default_m)
                )
                j_provider, j_model = MODEL_OPTIONS[j_model_label]
                focus = focuses[i]
                role  = role_labels[i]

                judge_label = f"Judge {i + 1} · {role} · {j_model_label}"
                with st.chat_message(judge_label, avatar=get_judge_image_path(role)):
                    try:
                        from audio_engine import SentenceChunker, get_audio_manager

                        # Configure distinct voice and panning per judge
                        judge_voices = ["en-US-EmmaNeural", "en-GB-SoniaNeural", "en-US-AndrewMultilingualNeural"]
                        voice_id = judge_voices[i % len(judge_voices)]
                        panning = -0.4 if i == 0 else (0.4 if i == 1 else 0.0)

                        speaker_speed = float(
                            st.session_state.get("_dp_speaker_speed")
                            or st.session_state.get("_pref_speaker_speed")
                            or st.session_state.get("speaker_speed", 1.0)
                            or 1.0
                        )

                        def on_sentence(sentence):
                            if not _session_bool(
                                "_dp_audio_transcription",
                                "_pref_audio_transcription",
                                "audio_transcription",
                                default=False,
                            ):
                                return
                            if get_audio_manager().is_running:
                                get_audio_manager().enqueue(sentence, voice_id, panning, speaker_speed)

                        chunker = SentenceChunker(on_sentence)

                        token_gen = judge_debate(
                            selected_topic, transcript,
                            judge_provider=j_provider,
                            judge_model=j_model,
                            focus=focus,
                        )

                        j_text = ""

                        def stream_judge_tokens():
                            nonlocal j_text
                            for chunk in token_gen:
                                j_text += chunk
                                chunker.add_token(chunk)
                                yield chunk

                        judge_body = st.empty()
                        with judge_body.container():
                            st.write_stream(stream_judge_tokens())

                        chunker.flush()
                        winner = extract_winner(j_text, transcript)
                        scores = parse_judge_scores(j_text, transcript)
                        reasoning = clean_judge_reasoning(j_text, transcript)
                        j = {
                            "winner": winner,
                            "reasoning": reasoning,
                            "scores": scores
                        }
                        judge_body.empty()
                        with judge_body.container():
                            render_judge_result_body(j)
                        judgments[i] = j
                        judge_results[i] = {
                            "judgment":    j,
                            "model_label": j_model_label,
                            "role":        role,
                        }

                    except Exception as exc:
                        err_j = {"winner": "Error", "reasoning": f"Error running judge: {exc}", "scores": {}}
                        judgments[i] = err_j
                        judge_results[i] = {
                            "judgment":    err_j,
                            "model_label": j_model_label,
                            "role":        role,
                        }
                        st.error(f"Error executing judge: {exc}")

        judgment = aggregate_judgments(judgments)

    st.session_state["judgment"] = judgment
    st.session_state["judge_results"] = judge_results

    if not free_topic_mode and not philosopher_teams:
        render_winner_announcement()

    # Summary is always generated
    s_model_label = st.session_state.get("_dp_summary_model") or st.session_state.get("_pref_summary_model") or st.session_state.get("summary_model_label", default_m)
    if s_model_label not in MODEL_OPTIONS:
        s_model_label = default_m
    s_provider, s_model = MODEL_OPTIONS[s_model_label]
    
    with st.spinner("Finalizing debate summary..."):
        try:
            summary_gen = summarize_debate(
                st.session_state["selected_topic"], transcript, judgment,
                judge_provider=s_provider, judge_model=s_model,
                free_topic_mode=free_topic_mode,
            )
            summary = "".join(list(summary_gen))
        except Exception as exc:
            summary = f"Error generating summary: {exc}"

    st.session_state["transcript"]    = transcript
    st.session_state["summary"]       = summary
    st.session_state["topic"]         = st.session_state["selected_topic"]
    st.session_state["agent_configs"] = agent_configs
    st.session_state["team_configs"]  = team_configs
    st.session_state["token_usage"]   = get_token_usage()
    st.session_state["team_traces"]   = team_traces
    st.session_state["team_mode_active"] = philosopher_teams
    st.session_state["judge_step"]    = 0  # start at first judge



def get_judge_focus_description(role: str) -> str:
    role_lower = str(role).lower()
    if "logic" in role_lower:
        return "Focuses on logical validity, structural consistency, and argument strength."
    elif "debate" in role_lower:
        return "Focuses on counterargument handling, style, and debate dynamics."
    elif "clarity" in role_lower:
        return "Focuses on clarity of expression, language, and communication quality."
    else:
        return "Focuses on overall debate quality, balancing logic, style, and clarity."


def render_judge_evaluation_panel(jr: dict) -> None:
    j = jr["judgment"]
    j_r = j.get("reasoning", "")
    role = jr.get("role", "General Evaluation")
    model_label = jr.get("model_label", "")

    col1, col2 = st.columns([1, 2.2])

    with col1:
        scale_image_src = image_to_data_uri(get_judge_image_path(role))
        focus_desc = get_judge_focus_description(role)

        card_html = (
            f'<div class="fighter-card" style="min-height: auto; padding-bottom: 24px; margin-bottom: 18px;">'
            f'<img src="{scale_image_src}" class="fighter-image" style="height: 180px; object-fit: contain; margin-bottom: 16px;">'
            f'<div class="fighter-name" style="font-size: 1.25rem;">{html.escape(role)} Judge</div>'
            f'<div class="fighter-side" style="font-size: 0.8rem; color: #ffd54a; margin-top: 4px;">{html.escape(model_label)}</div>'
            f'<div class="fighter-stance" style="font-size: 0.85rem; margin-top: 10px; line-height: 1.4; color: #e8ecff;">{html.escape(focus_desc)}</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

    with col2:
        j_r_html = (
            f'<div style="color:#e8ecff; margin-bottom:12px; line-height:1.6; font-size:0.95rem;">'
            f'{_markdown_to_html(j_r)}'
            f'</div>'
            if j_r else ""
        )
        scores_html = _build_scores_html(j.get("scores", {}))

        details_html = (
            f'<div class="arcade-panel" style="padding: 20px; margin-bottom: 12px;">'
            f'<div class="topic-label" style="margin-bottom: 12px; font-size: 0.85rem;">Evaluation Details</div>'
            f'{j_r_html}'
            f'{scores_html}'
            f'</div>'
        )
        st.markdown(details_html, unsafe_allow_html=True)


def render_arena_stage() -> None:
    """Stage 4: shows the debate transcript, then continues to the summary."""
    if "transcript" not in st.session_state:
        handle_run_debate()

    transcript      = st.session_state.get("transcript")
    topic           = st.session_state.get("topic", st.session_state.get("selected_topic", ""))
    free_topic_mode = _current_start_mode() == "free"

    # ── Debater model attribution ─────────────────────────────────────────────
    phil1  = st.session_state.get("agent1_philosopher_name", "Player 1")
    phil2  = st.session_state.get("agent2_philosopher_name", "Player 2")
    label1 = st.session_state.get("agent1_model_label", "")
    label2 = st.session_state.get("agent2_model_label", "")
    philosopher_teams = st.session_state.get("philosopher_teams", False)
    team_configs = st.session_state.get("_dp_team_configs") or {}

    if label1 or label2:
        if free_topic_mode:
            team_label = f"{phil1} (Free Topic Team)"
            if philosopher_teams and isinstance(team_configs, dict):
                supporting = team_configs.get("free", {}).get("agent_a", {}).get("philosopher_name", "")
                if supporting:
                    team_label = f"{phil1} (Free Topic Team — Support: {supporting})"
            rows = (
                f'<div style="display:flex; justify-content:space-between; margin-bottom:2px;">'
                f'<span style="color:#e8ecff;">{team_label}</span>'
                f'<span style="color:#ffd54a; font-size:0.85rem;">{label1}</span>'
                f'</div>'
            )
        else:
            team_label1 = f"{phil1} (Pro)"
            team_label2 = f"{phil2} (Contra)"
            if philosopher_teams and isinstance(team_configs, dict):
                supporting1 = team_configs.get("for", {}).get("agent_a", {}).get("philosopher_name", "")
                supporting2 = team_configs.get("against", {}).get("agent_a", {}).get("philosopher_name", "")
                if supporting1:
                    team_label1 = f"{phil1} (Pro — Support: {supporting1})"
                if supporting2:
                    team_label2 = f"{phil2} (Contra — Support: {supporting2})"
            rows = (
                f'<div style="display:flex; justify-content:space-between; margin-bottom:2px;">'
                f'<span style="color:#e8ecff;">{team_label1}</span>'
                f'<span style="color:#ffd54a; font-size:0.85rem;">{label1}</span>'
                f'</div>'
                f'<div style="display:flex; justify-content:space-between; margin-bottom:2px;">'
                f'<span style="color:#e8ecff;">{team_label2}</span>'
                f'<span style="color:#ffd54a; font-size:0.85rem;">{label2}</span>'
                f'</div>'
            )
        st.markdown(
            f"""
            <div class="score-card" style="padding:8px 14px; margin-bottom:10px;">
                <div style="font-size:0.7rem; color:#aaa; text-transform:uppercase;
                            letter-spacing:0.06em; margin-bottom:6px;">Debater Models</div>
                {rows}
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_debate_text_panel(topic, transcript)
    render_development_panel()

    # ── Navigation ────────────────────────────────────────────────────────────
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("New Debate", use_container_width=True):
            reset_game()
            st.rerun()
    with btn_col2:
        if st.button("Show Summary", type="primary", use_container_width=True):
            st.session_state["stage"] = 5
            st.rerun()


SUMMARY_INTRO_RE = re.compile(
    r"^here(?:'s|\s+(?:are|is))\b.*summar",
    re.IGNORECASE,
)


def _summary_display_lines(summary: str) -> list[str]:
    return [
        line
        for line in (raw.strip() for raw in summary.strip().splitlines())
        if line and not SUMMARY_INTRO_RE.search(line)
    ]


def _clean_summary_speaker_label(label: str) -> str:
    cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", str(label)).strip()
    cleaned = re.sub(r"^(?:\*\*|__)(.*?)(?:\*\*|__)$", r"\1", cleaned).strip()
    return cleaned


def _summary_speaker_label_html(label: str) -> str:
    match = re.match(r"^(.*?)\s*\(([^)]*)\)\s*$", str(label).strip())
    if match:
        name, role = match.groups()
        return (
            f'<span>{html.escape(name.strip())}</span> '
            f'<span style="color:#cfd8ff;font-size:0.72rem;font-weight:700;'
            f'opacity:0.78;white-space:nowrap;">({html.escape(role.strip())})</span>'
        )
    return html.escape(str(label).strip())


def _summary_public_speaker_label(label: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(label).strip()).strip()


def _summary_display_rows(
    summary: str,
    transcript: list[dict] | None = None,
    use_transcript_speakers: bool = False,
) -> list[tuple[str, str]]:
    rows = []
    transcript = transcript or []
    for idx, line in enumerate(_summary_display_lines(summary)):
        speaker = ""
        text = line
        if ":" in line:
            speaker, _, text = line.partition(":")
            speaker = _clean_summary_speaker_label(speaker)
        if use_transcript_speakers and idx < len(transcript):
            transcript_speaker = str(transcript[idx].get("speaker", "")).strip()
            if transcript_speaker:
                speaker = transcript_speaker
        rows.append((speaker, text.strip()))
    return rows


def _free_topic_team_roles_for_display(trace: dict | None = None) -> dict:
    if isinstance(trace, dict):
        team_roles = trace.get("team_roles", {})
        if isinstance(team_roles, dict) and team_roles:
            return team_roles

    team_configs = (
        st.session_state.get("team_configs")
        or st.session_state.get("_dp_team_configs")
        or st.session_state.get("_pref_team_configs")
        or {}
    )
    free_config = team_configs.get("free", {}) if isinstance(team_configs, dict) else {}
    roles = {}
    for role_key, role_label, _fallback_key in TEAM_ROLE_DEFS:
        role = free_config.get(role_key, {}) if isinstance(free_config, dict) else {}
        if not isinstance(role, dict):
            continue
        roles[role_key] = {
            "label": role.get("label", role_label),
            "philosopher": role.get("philosopher_name") or role.get("philosopher", "-"),
            "strategy": role.get("strategy", "-"),
        }
    return roles


def render_summary_stage() -> None:
    """Stage 5: one-sentence-per-argument summary."""
    summary       = st.session_state.get("summary")
    topic         = st.session_state.get("topic", st.session_state.get("selected_topic", ""))
    transcript    = st.session_state.get("transcript", []) or []
    
    winner = None
    winner_display = ""

    # ── Topic reminder ────────────────────────────────────────────────────────
    render_topic_panel(topic, small=True)

    # ── Winner Announcement ───────────────────────────────────────────────────
    free_topic_mode = _current_start_mode() == "free"
    if not free_topic_mode:
        judge_results = st.session_state.get("judge_results", [])
        votes = []
        for jr in judge_results:
            if isinstance(jr, dict) and "judgment" in jr:
                w = jr["judgment"].get("winner")
                if w and w not in ("N/A", "Error"):
                    votes.append(w.strip())
        
        if votes:
            from collections import Counter
            counts = Counter(votes)
            most_common = counts.most_common()
            if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                winner = "Tie / Split Decision"
            else:
                winner = most_common[0][0]
        else:
            judgment = st.session_state.get("judgment")
            if isinstance(judgment, dict):
                w = judgment.get("winner")
                if w and w not in ("N/A", "Error"):
                    winner = w

        if winner:
            # Map winner to philosopher info
            winner_key = None
            for key, details in PHILOSOPHER_LIBRARY.items():
                if details["name"].lower() in winner.lower() or winner.lower() in details["name"].lower():
                    winner_key = key
                    break
            
            if winner_key:
                winner_img_path = PHILOSOPHER_LIBRARY[winner_key]["image"]
                winner_display = PHILOSOPHER_LIBRARY[winner_key]["name"]
                status_text = "🏆 Winner"
            else:
                winner_img_path = "images/Judge_General.png"
                winner_display = winner
                status_text = "⚖️ Result"
            
            try:
                winner_img_src = image_to_data_uri(winner_img_path)
            except Exception:
                winner_img_src = ""

            winner_html = (
                f'<div class="arcade-panel" style="margin-bottom:18px; display:flex; align-items:center; gap:20px;">'
            )
            if winner_img_src:
                winner_html += f'<img src="{winner_img_src}" style="width: 80px; height: 80px; border-radius: 50%; object-fit: cover; border: 2px solid #ffd54a; box-shadow: 0 0 10px rgba(255,213,74,0.3);">'
            winner_html += (
                f'<div>'
                f'<div style="font-size:0.75rem; color:#ffd54a; text-transform:uppercase; letter-spacing:2px; font-weight:700; margin-bottom:4px;">{status_text}</div>'
                f'<div style="font-size:1.6rem; font-weight:900; color:#fff3bf; text-transform:uppercase; text-shadow:0 0 8px rgba(255,243,191,0.2);">{html.escape(winner_display)}</div>'
                f'</div>'
                f'</div>'
            )
            st.markdown(winner_html, unsafe_allow_html=True)

    # ── Argument summary (one sentence per argument) ──────────────────────────
    if summary:
        # Render each line as its own styled row
        rows = _summary_display_rows(
            summary,
            transcript,
            use_transcript_speakers=True,
        )
        if free_topic_mode and rows:
            speaker_part, rest = rows[0]
            rows = [(_summary_public_speaker_label(speaker_part), rest)]
        rows_html = ""
        for speaker_part, rest in rows:
            if speaker_part:
                rows_html += (
                    f'<div style="display:grid; grid-template-columns:minmax(170px, 260px) minmax(0, 1fr); '
                    f'column-gap:18px; row-gap:4px; margin-bottom:10px; align-items:baseline;">'
                    f'<span style="color:#ffd54a; font-weight:800; font-size:0.88rem; '
                    f'text-align:right; overflow-wrap:anywhere;">'
                    f'{_summary_speaker_label_html(speaker_part)}</span>'
                    f'<span style="color:#e8ecff; line-height:1.6; min-width:0;">'
                    f'{html.escape(rest.strip())}</span>'
                    f'</div>'
                )
            else:
                rows_html += (
                    f'<div style="color:#e8ecff; margin-bottom:8px; line-height:1.6;">'
                    f'{html.escape(rest.strip())}</div>'
                )
        
        # Append Winner at the bottom of the summary box
        if not free_topic_mode and winner:
            rows_html += (
                f'<div style="margin-top:20px; border-top:1px solid rgba(255,255,255,0.1); padding-top:12px; '
                f'display:flex; gap:10px; align-items:baseline;">'
                f'<span style="color:#ffd54a; font-weight:800; white-space:nowrap; '
                f'font-size:0.95rem; min-width:130px; text-align:right;'
                f'">Winner</span>'
                f'<span style="color:#fff; font-weight:800; font-size:1.0rem; text-transform:uppercase; letter-spacing:0.5px;">'
                f'{html.escape(winner_display)}</span>'
                f'</div>'
            )

        st.markdown(
            f"""
            <div class="arcade-panel" style="margin-bottom:18px;">
                <div class="topic-label">Debate Summary</div>
                <div style="margin-top:0.8rem;">{rows_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Free Topic: decision process panel ───────────────────────────────────
    if free_topic_mode:
        team_traces = st.session_state.get("team_traces", [])
        for entry in team_traces:
            trace = entry.get("trace", {})
            if not isinstance(trace, dict):
                continue
            candidates = trace.get("candidates", [])
            selection = trace.get("selection", {}) if isinstance(trace.get("selection"), dict) else {}
            reviews = trace.get("reviews", [])
            team_roles = _free_topic_team_roles_for_display(trace)
            selected_id = selection.get("selected_candidate", "")
            final_score = int(trace.get("final_score", 0) or 0)
            approved = trace.get("approved", False)

            with st.expander("Decision Process — How this argument was crafted", expanded=True):
                if team_roles:
                    role_summary = " | ".join(
                        f"{role.get('label', key)}: {role.get('philosopher', '-')} / {role.get('strategy', '-')}"
                        for key, role in team_roles.items()
                        if isinstance(role, dict)
                    )
                    st.caption(role_summary)
                st.caption(
                    "The philosopher team internally generated two opinion angles. "
                    "When the input allows it, Support explores a supportive/pro angle and Counter explores a skeptical/contra angle. "
                    "The Critic evaluated both angles, selected the stronger perspective, and refined it into the final opinion."
                )

                if candidates:
                    st.markdown("**Both approaches considered:**")
                    for c in candidates:
                        if not isinstance(c, dict):
                            continue
                        cid = c.get("id", "?")
                        angle = c.get("angle", "")
                        plan = c.get("plan_summary", "No summary available.")
                        role_key = "agent_a" if cid.startswith("A") else "agent_b" if cid.startswith("B") else ""
                        role = team_roles.get(role_key, {}) if isinstance(team_roles, dict) else {}
                        role_name = c.get("role_philosopher") or role.get("philosopher", "")
                        role_strategy = c.get("role_strategy") or role.get("strategy", "")
                        is_selected = (cid == f"{selected_id} (Attempt {len(reviews)})") if reviews else (cid == selected_id)
                        border_color = "#ffd54a" if is_selected else "#444"
                        label_color = "#ffd54a" if is_selected else "#888"
                        badge = "Selected ✓" if is_selected else "Not selected ✗"
                        role_bits = f" · {role_name}" if role_name else ""
                        strategy_bits = f" · {role_strategy}" if role_strategy else ""
                        angle_bits = f" · {angle}" if angle else ""
                        st.markdown(
                            f'<div style="border-left:3px solid {border_color};padding:8px 12px;'
                            f'margin-bottom:10px;background:rgba(255,255,255,0.03);border-radius:0 6px 6px 0;">'
                            f'<div style="color:{label_color};font-weight:700;font-size:0.88rem;margin-bottom:4px;">'
                            f'Approach {html.escape(str(cid))}{html.escape(angle_bits)}'
                            f'{html.escape(role_bits)}{html.escape(strategy_bits)} · {badge}'
                            f'</div>'
                            f'<div style="color:#e8ecff;font-size:0.9rem;line-height:1.5;">{html.escape(plan)}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                sel_reasoning = selection.get("selection_reasoning", "")
                if sel_reasoning:
                    st.markdown(f"**Why Approach {selected_id} was chosen:**")
                    st.write(sel_reasoning)

                if reviews:
                    total = len(reviews)
                    status_str = "approved ✓" if approved else "best effort (max attempts reached)"
                    st.markdown(
                        f"**Quality review — {total} draft{'s' if total > 1 else ''} · "
                        f"final score {final_score}/10 · {status_str}:**"
                    )
                    for review in reviews:
                        if not isinstance(review, dict):
                            continue
                        attempt = int(review.get("attempt", 0) or 0)
                        score = int(review.get("score", 0) or 0)
                        rev_approved = review.get("approved", False)
                        critique = review.get("critique", "")
                        rev_status = "Approved ✓" if rev_approved else "Revision requested ↺"
                        st.markdown(f"Draft {attempt}: **{score}/10** · {rev_status}")
                        if critique and not rev_approved:
                            st.caption(f"Critique: {critique}")

    render_development_panel()

    # ── Navigation ────────────────────────────────────────────────────────────
    if st.button("New Debate", type="primary", use_container_width=True):
        reset_game()
        st.rerun()
