import json
import time
from typing import Dict
import base64
from pathlib import Path
import streamlit as st

from config import PHILOSOPHER_LIBRARY, STRATEGY_OPTIONS, random_topic
from debate_engine_cloud import (
    aggregate_judgments,
    judge_debate,
    run_debate,
    summarize_debate,
)

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
    "Llama3.2:1b (Local)":            ("local", "llama3.2:1b"),
    "Phi-4 mini  (Platform)":         ("custom", "Phi-4-mini-reasoning"),
    "GPT-5-chat  (Platform)":         ("custom", "gpt-5-chat"),
    "GPT-4.1 mini  (Platform)":       ("custom", "gpt-4.1-mini"),
    "DeepSeek-V3.2  (Platform)":      ("custom", "DeepSeek-V3.2"),
    "Mistral Large 3  (Platform)":    ("custom", "mistral-Large-3"),
    "Mistral Small  (Platform)":      ("custom", "mistral-small-2503"),
    "Llama 4 Maverick  (Platform)":   ("custom", "Llama-4-Maverick-17B-128E-Instruct-FP8"),
    "Llama 3.3 70B  (Platform)":      ("custom", "Llama-3.3-70B-Instruct"),
}


def image_to_data_uri(image_path: str) -> str:
    image_bytes = Path(image_path).read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

_DEBATE_PARAM_KEYS = [
    "_dp_phil1", "_dp_phil2", "_dp_model1", "_dp_model2",
    "_dp_strat1", "_dp_strat2", "_dp_num_judges",
    "_dp_judge1_model", "_dp_judge2_model", "_dp_judge3_model",
]


def reset_game() -> None:
    st.session_state["stage"] = 0
    st.session_state["selected_topic"] = random_topic()
    st.session_state["rounds"] = 2
    st.session_state["player1_strategy"] = STRATEGY_OPTIONS[0]
    st.session_state["player2_strategy"] = STRATEGY_OPTIONS[0]
    st.session_state["agent1_philosopher_name"] = PHILOSOPHER_LIBRARY["socrates"]["name"]
    st.session_state["agent2_philosopher_name"] = PHILOSOPHER_LIBRARY["nietzsche"]["name"]
    _default_model = list(MODEL_OPTIONS.keys())[0]
    st.session_state.setdefault("agent1_model_label",  _default_model)
    st.session_state.setdefault("agent2_model_label",  _default_model)
    st.session_state.setdefault("judge1_model_label",  _default_model)
    st.session_state.setdefault("judge2_model_label",  _default_model)
    st.session_state.setdefault("judge3_model_label",  _default_model)
    st.session_state.setdefault("num_judges", 1)

    st.session_state["_pref_summary"] = False  # reset to default on new game

    for key in [
        "transcript", "judgment", "judge_results", "summary", "topic", "agent_configs",
        *_DEBATE_PARAM_KEYS,
    ]:
        st.session_state.pop(key, None)


def _save_debate_params() -> None:
    """Copy live widget values to stable keys before a stage transition.

    Widget-bound keys are disowned by Streamlit when their widget isn't rendered,
    so we snapshot them here while they are guaranteed to be current.
    """
    default_m = list(MODEL_OPTIONS.keys())[0]
    st.session_state["_dp_phil1"]       = st.session_state.get("agent1_philosopher_name")
    st.session_state["_dp_phil2"]       = st.session_state.get("agent2_philosopher_name")
    st.session_state["_dp_model1"]      = st.session_state.get("agent1_model_label", default_m)
    st.session_state["_dp_model2"]      = st.session_state.get("agent2_model_label", default_m)
    st.session_state["_dp_strat1"]      = st.session_state.get("player1_strategy", STRATEGY_OPTIONS[0])
    st.session_state["_dp_strat2"]      = st.session_state.get("player2_strategy", STRATEGY_OPTIONS[0])
    num_j = st.session_state.get("num_judges", 1)
    st.session_state["_dp_num_judges"]  = num_j
    for i in range(3):
        st.session_state[f"_dp_judge{i+1}_model"] = st.session_state.get(
            f"judge{i+1}_model_label", default_m
        )


def ensure_session_state() -> None:
    if "stage" not in st.session_state:
        reset_game()
        return

    st.session_state.setdefault("selected_topic", random_topic())
    st.session_state.setdefault("rounds", 2)
    st.session_state.setdefault("player1_strategy", STRATEGY_OPTIONS[0])
    st.session_state.setdefault("player2_strategy", STRATEGY_OPTIONS[0])
    st.session_state.setdefault(
        "agent1_philosopher_name", PHILOSOPHER_LIBRARY["socrates"]["name"]
    )
    st.session_state.setdefault(
        "agent2_philosopher_name", PHILOSOPHER_LIBRARY["nietzsche"]["name"]
    )
    _default_model = list(MODEL_OPTIONS.keys())[0]
    st.session_state.setdefault("agent1_model_label",  _default_model)
    st.session_state.setdefault("agent2_model_label",  _default_model)
    st.session_state.setdefault("judge1_model_label",  _default_model)
    st.session_state.setdefault("judge2_model_label",  _default_model)
    st.session_state.setdefault("judge3_model_label",  _default_model)
    st.session_state.setdefault("num_judges", 1)
    st.session_state.setdefault("include_summary", False)
    st.session_state.setdefault("_pref_summary", False)


def current_agent_configs():
    philosopher_options = {v["name"]: k for k, v in PHILOSOPHER_LIBRARY.items()}
    return [
        {
            "philosopher_key": philosopher_options[
                st.session_state["agent1_philosopher_name"]
            ]
        },
        {
            "philosopher_key": philosopher_options[
                st.session_state["agent2_philosopher_name"]
            ]
        },
    ]


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
    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        st.empty()
    with top_col2:
        if st.button("New Game", use_container_width=True):
            reset_game()
            st.rerun()


def render_topic_panel(topic: str) -> None:
    st.markdown(
        f"""
        <div class="topic-panel">
            <div class="topic-label">Debate Topic</div>
            <div class="topic-text">{topic}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_fighter_card(philosopher_key: str, side_label: str) -> None:
    philosopher = PHILOSOPHER_LIBRARY[philosopher_key]
    image_src = image_to_data_uri(philosopher["image"])

    st.markdown(
        f"""
        <div class="fighter-card">
            <img src="{image_src}" class="fighter-image">
            <div class="fighter-name">{philosopher['name']}</div>
            <div class="fighter-side">{side_label}</div>
            <div class="fighter-stance">{philosopher['stance']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_character_cards() -> None:
    configs = current_agent_configs()
    col1, col2 = st.columns(2)
    with col1:
        render_fighter_card(configs[0]["philosopher_key"], "Player 1 • For")
    with col2:
        render_fighter_card(configs[1]["philosopher_key"], "Player 2 • Against")


def _build_scores_html(scores: dict) -> str:
    """Return score cards as an HTML string (safe to inline in a single st.markdown call).

    Handles both the new metric keys (logical_validity, …) and any legacy or
    unexpected format the model might return — nothing is silently dropped.
    """
    _fallback_labels = {
        "logic": "Logic", "relevance": "Relevance",
        "rebuttal": "Rebuttal", "fairness": "Fairness",
    }
    parts = []
    for name, vals in scores.items():
        has_new_keys = any(k in METRIC_LABELS for k in vals)
        if has_new_keys:
            rows = "".join(
                f'<div>{label}: {vals.get(key, "-")}</div>'
                for key, label in METRIC_LABELS.items()
            )
        else:
            # Fall back to whatever the model returned (old or unexpected format)
            rows = "".join(
                f'<div>{_fallback_labels.get(k, k.replace("_", " ").title())}: {v}</div>'
                for k, v in vals.items()
            )
        parts.append(
            f'<div class="score-card">'
            f'<div class="score-name">{name}</div>'
            f'<div class="score-grid">{rows}</div>'
            f'</div>'
        )
    return "".join(parts)


def render_scores(scores: Dict[str, Dict[str, int]]) -> None:
    st.markdown(_build_scores_html(scores), unsafe_allow_html=True)


def render_start_screen() -> None:
    st.markdown(
        """
        <div class="arcade-panel" style="text-align:center;">
            <div style="font-size:2rem; font-weight:800; margin-bottom:1rem;">Philosopher Arena</div>
            <div style="font-size:1.1rem; color:#cfd8ff; margin-bottom:1rem;">Loading Debate Engine</div>
            <div class="blink" style="font-size:1.2rem; font-weight:800; color:#ffd54a;">Press Start</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    st.checkbox(
        "Include written summary",
        key="include_summary",
        help="Generate a written debate summary after the arena stage (costs extra tokens). Off by default.",
    )

    if st.button("Start Game", type="primary", use_container_width=True):
        # Save to a stable key before transitioning — the widget key will be
        # disowned once stage 0 is no longer rendered.
        st.session_state["_pref_summary"] = st.session_state.get("include_summary", False)
        st.session_state["stage"] = 1
        st.rerun()


def render_topic_stage() -> None:
    render_topic_panel(st.session_state["selected_topic"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Redraw Topic", use_container_width=True):
            st.session_state["selected_topic"] = random_topic()
            st.rerun()
    with col2:
        if st.button("Continue", type="primary", use_container_width=True):
            st.session_state["stage"] = 2
            st.rerun()


def render_character_stage() -> None:
    render_topic_panel(st.session_state["selected_topic"])

    model_keys   = list(MODEL_OPTIONS.keys())
    judge_keys   = ["judge1_model_label", "judge2_model_label", "judge3_model_label"]
    judge_labels = ["⚖️ Judge 1 Model", "⚖️ Judge 2 Model", "⚖️ Judge 3 Model"]

    # ── Philosophers ──────────────────────────────────────────────────────────
    philosopher_names = [v["name"] for v in PHILOSOPHER_LIBRARY.values()]

    col_a, col_b = st.columns(2)
    with col_a:
        st.selectbox(
            "Player 1 Philosopher",
            options=philosopher_names,
            key="agent1_philosopher_name",
        )
    with col_b:
        st.selectbox(
            "Player 2 Philosopher",
            options=philosopher_names,
            key="agent2_philosopher_name",
        )

    # ── Strategies ────────────────────────────────────────────────────────────
    strategy_col1, strategy_col2 = st.columns(2)
    with strategy_col1:
        st.selectbox("Player 1 Strategy", options=STRATEGY_OPTIONS, key="player1_strategy")
    with strategy_col2:
        st.selectbox("Player 2 Strategy", options=STRATEGY_OPTIONS, key="player2_strategy")

    render_character_cards()

    # ── Debater models ────────────────────────────────────────────────────────
    st.markdown("**Debater Models**")
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.selectbox(
            "🤖 Player 1 Model",
            options=model_keys,
            key="agent1_model_label",
            help="Model used by the FOR debater.",
        )
    with dcol2:
        st.selectbox(
            "🤖 Player 2 Model",
            options=model_keys,
            key="agent2_model_label",
            help="Model used by the AGAINST debater.",
        )

    # ── Judge panel ───────────────────────────────────────────────────────────
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
        with jcols[i]:
            st.selectbox(
                judge_labels[i],
                options=model_keys,
                key=judge_keys[i],
                help=JUDGE_FOCUS[num_judges][i] or focus_hints[i],
            )

    # ── Live model lineup (reads from session state updated by widgets above) ─
    judge_roles = JUDGE_ROLE_LABELS[num_judges]
    rows = (
        f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
        f'<span style="color:#e8ecff;">{st.session_state["agent1_philosopher_name"]} <span style="color:#aaa;font-size:0.8rem;">(For)</span></span>'
        f'<span style="color:#ffd54a;font-size:0.85rem;">{st.session_state["agent1_model_label"]}</span>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
        f'<span style="color:#e8ecff;">{st.session_state["agent2_philosopher_name"]} <span style="color:#aaa;font-size:0.8rem;">(Against)</span></span>'
        f'<span style="color:#ffd54a;font-size:0.85rem;">{st.session_state["agent2_model_label"]}</span>'
        f'</div>'
    )
    for i in range(num_judges):
        rows += (
            f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
            f'<span style="color:#e8ecff;">{judge_roles[i]} Judge</span>'
            f'<span style="color:#ffd54a;font-size:0.85rem;">{st.session_state[judge_keys[i]]}</span>'
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
            _save_debate_params()
            st.session_state["stage"] = 3
            st.rerun()


def render_versus_stage() -> None:
    configs = current_agent_configs()
    render_topic_panel(st.session_state["selected_topic"])

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        render_fighter_card(configs[0]["philosopher_key"], "Player 1 • For")
    with col2:
        st.markdown(
            """
            <div style="display:flex; align-items:center; justify-content:center; height:100%;">
                <div class="arcade-vs">VS</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        render_fighter_card(configs[1]["philosopher_key"], "Player 2 • Against")

    st.markdown(
        """
        <div class="arcade-panel" style="text-align:center; margin-top:18px;">
            <div style="font-size:1.8rem; font-weight:900; margin-bottom:10px;">Match Loading</div>
            <div style="font-size:1.05rem; color:#cfd8ff; margin-bottom:16px;">
                Preparing arena, syncing philosophers, loading debate engine...
            </div>
            <div class="blink" style="font-size:1.1rem; font-weight:800; color:#ffd54a;">
                Entering Arena...
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "transcript" not in st.session_state:
        with st.spinner("Running debate — this may take a minute…"):
            handle_run_debate()
    st.session_state["stage"] = 4
    st.rerun()


def handle_run_debate() -> None:
    philosopher_options = {v["name"]: k for k, v in PHILOSOPHER_LIBRARY.items()}
    default_m = list(MODEL_OPTIONS.keys())[0]

    # Prefer the stable _dp_ snapshot keys; fall back to widget keys if not set
    # (e.g. when called directly from render_arena_stage's safety net).
    agent1_name        = st.session_state.get("_dp_phil1")  or st.session_state.get("agent1_philosopher_name", PHILOSOPHER_LIBRARY["socrates"]["name"])
    agent2_name        = st.session_state.get("_dp_phil2")  or st.session_state.get("agent2_philosopher_name", PHILOSOPHER_LIBRARY["nietzsche"]["name"])
    agent1_model_label = st.session_state.get("_dp_model1") or st.session_state.get("agent1_model_label", default_m)
    agent2_model_label = st.session_state.get("_dp_model2") or st.session_state.get("agent2_model_label", default_m)
    p1_strategy        = st.session_state.get("_dp_strat1") or st.session_state.get("player1_strategy", STRATEGY_OPTIONS[0])
    p2_strategy        = st.session_state.get("_dp_strat2") or st.session_state.get("player2_strategy", STRATEGY_OPTIONS[0])
    num_judges         = st.session_state.get("_dp_num_judges") or st.session_state.get("num_judges", 1)

    agent1_provider, agent1_model = MODEL_OPTIONS[agent1_model_label]
    agent2_provider, agent2_model = MODEL_OPTIONS[agent2_model_label]

    agent_configs = [
        {
            "philosopher_key": philosopher_options[agent1_name],
            "provider":    agent1_provider,
            "model":       agent1_model,
            "model_label": agent1_model_label,
            "display_name": f"{agent1_name} (For)",
        },
        {
            "philosopher_key": philosopher_options[agent2_name],
            "provider":    agent2_provider,
            "model":       agent2_model,
            "model_label": agent2_model_label,
            "display_name": f"{agent2_name} (Against)",
        },
    ]

    transcript = run_debate(
        st.session_state["selected_topic"],
        st.session_state["rounds"],
        [p1_strategy, p2_strategy],
        agent_configs,
    )

    focuses     = JUDGE_FOCUS[num_judges]
    role_labels = JUDGE_ROLE_LABELS[num_judges]
    judge_results: list = []
    judgments: list = []
    for i in range(num_judges):
        j_model_label = (
            st.session_state.get(f"_dp_judge{i+1}_model")
            or st.session_state.get(f"judge{i+1}_model_label", default_m)
        )
        j_provider, j_model = MODEL_OPTIONS[j_model_label]
        j = judge_debate(
            st.session_state["selected_topic"], transcript,
            judge_provider=j_provider,
            judge_model=j_model,
            focus=focuses[i],
        )
        judgments.append(j)
        judge_results.append({
            "judgment":    j,
            "model_label": j_model_label,
            "role":        role_labels[i],
        })

    judgment = aggregate_judgments(judgments)

    if st.session_state.get("_pref_summary", False):
        s_model_label = st.session_state.get("_dp_judge1_model") or st.session_state.get("judge1_model_label", default_m)
        s_provider, s_model = MODEL_OPTIONS[s_model_label]
        summary = summarize_debate(
            st.session_state["selected_topic"], transcript, judgment,
            judge_provider=s_provider, judge_model=s_model,
        )
    else:
        summary = None

    st.session_state["transcript"]    = transcript
    st.session_state["judgment"]      = judgment
    st.session_state["judge_results"] = judge_results
    st.session_state["summary"]       = summary
    st.session_state["topic"]         = st.session_state["selected_topic"]
    st.session_state["agent_configs"] = agent_configs

    # Write back resolved display values so the arena/summary stages show the right names
    st.session_state["agent1_philosopher_name"] = agent1_name
    st.session_state["agent2_philosopher_name"] = agent2_name
    st.session_state["agent1_model_label"]      = agent1_model_label
    st.session_state["agent2_model_label"]      = agent2_model_label
    st.session_state["num_judges"]              = num_judges


def render_arena_stage() -> None:
    if "transcript" not in st.session_state:
        handle_run_debate()

    transcript = st.session_state.get("transcript")
    judgment = st.session_state.get("judgment")

    topic = st.session_state.get("topic", st.session_state.get("selected_topic", ""))

    # Participant model attribution — read directly from session state to avoid stale agent_configs
    phil1 = st.session_state.get("agent1_philosopher_name", "Player 1")
    phil2 = st.session_state.get("agent2_philosopher_name", "Player 2")
    label1 = st.session_state.get("agent1_model_label", "")
    label2 = st.session_state.get("agent2_model_label", "")
    if label1 or label2:
        rows = (
            f'<div style="display:flex; justify-content:space-between; margin-bottom:2px;">'
            f'<span style="color:#e8ecff;">{phil1} (For)</span>'
            f'<span style="color:#ffd54a; font-size:0.85rem;">{label1}</span>'
            f'</div>'
            f'<div style="display:flex; justify-content:space-between; margin-bottom:2px;">'
            f'<span style="color:#e8ecff;">{phil2} (Against)</span>'
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

    st.markdown(
        f"""
        <div class="arcade-panel" style="margin-bottom:18px;">
            <div class="topic-label">Debate Chat</div>
            <div class="topic-text">{topic}</div>
        """,
        unsafe_allow_html=True,
    )

    if transcript:
        for turn in transcript:
            st.markdown(
                f"""
                <div class="score-card">
                    <div class="score-name">{turn['speaker']}</div>
                    <div style="color:#e8ecff; line-height:1.6;">{turn['text']}</div>
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

    if judgment:
        num_judges    = st.session_state.get("num_judges", 1)
        judge_results = st.session_state.get("judge_results", [])

        # ── Panel header label ────────────────────────────────────────────────
        if num_judges == 1:
            jr0         = judge_results[0] if judge_results else {}
            panel_label = f"Judge Result · {jr0.get('role', 'General Evaluation')} · {jr0.get('model_label', '')}"
        else:
            panel_label = f"Panel Result · {num_judges} Judges · Averaged"

        # ── Reasoning block (all built as HTML strings before the markdown call)
        if num_judges == 1:
            raw_r = judgment.get("reasoning", "")
            reasoning_html = (
                f'<p style="color:#cfd8ff; margin:0.8rem 0 0.4rem;">{raw_r}</p>'
                if raw_r else ""
            )
        else:
            parts = [
                f'<p style="color:#aaa; font-size:0.85rem; margin:0.3rem 0;">'
                f'<b>{jr["role"]} Judge ({jr["model_label"]}):</b> '
                f'{jr["judgment"].get("reasoning", "")}</p>'
                for jr in judge_results if jr["judgment"].get("reasoning")
            ]
            reasoning_html = "".join(parts)

        # ── Single markdown call — everything truly inside the arcade-panel div
        st.markdown(
            f"""
            <div class="arcade-panel" style="margin-top:8px; margin-bottom:18px;">
                <div class="topic-label">{panel_label}</div>
                <div class="winner-banner">Winner: {judgment.get("winner", "N/A")}</div>
                {reasoning_html}
                {_build_scores_html(judgment.get("scores", {}))}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Individual judge breakdowns (multi-judge only) ────────────────────
        if len(judge_results) > 1:
            cards_html = ""
            for jr in judge_results:
                j   = jr["judgment"]
                j_r = j.get("reasoning", "")
                j_r_html = (
                    f'<p style="color:#aaa; font-size:0.82rem; margin:0.4rem 0;">{j_r}</p>'
                    if j_r else ""
                )
                cards_html += (
                    f'<div class="score-card" style="border-left:3px solid #7c8fff; margin-bottom:10px;">'
                    f'<div class="score-name" style="font-size:0.95rem;">{jr["role"]} Judge</div>'
                    f'<div style="color:#aaa; font-size:0.8rem; margin-bottom:0.5rem;">{jr["model_label"]}</div>'
                    f'<div class="winner-banner" style="font-size:0.9rem; padding:10px 14px; margin-bottom:0.5rem;">'
                    f'Winner: {j.get("winner", "N/A")}</div>'
                    f'{j_r_html}'
                    f'{_build_scores_html(j.get("scores", {}))}'
                    f'</div>'
                )

            st.markdown(
                f"""
                <div class="arcade-panel" style="margin-top:8px; margin-bottom:18px;">
                    <div class="topic-label">Individual Judge Breakdowns</div>
                    {cards_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

    bottom_left, bottom_right = st.columns(2)

    with bottom_left:
        if st.button("New Debate", use_container_width=True):
            reset_game()
            st.rerun()

    with bottom_right:
        if st.button("Continue to Summary", type="primary", use_container_width=True):
            st.session_state["stage"] = 5
            st.rerun()


def render_summary_stage() -> None:
    summary       = st.session_state.get("summary")
    judgment      = st.session_state.get("judgment")
    judge_results = st.session_state.get("judge_results", [])
    num_judges    = st.session_state.get("num_judges", 1)

    # ── Summary text panel ────────────────────────────────────────────────────
    if summary:
        summary_safe = summary.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        summary_html = summary_safe.replace("\n", "<br>")
        st.markdown(
            f"""
            <div class="arcade-panel" style="margin-bottom:18px;">
                <div class="topic-label">Summary</div>
                <div style="color:#e8ecff; line-height:1.75; margin-top:0.6rem;">{summary_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="score-card" style="margin-bottom:18px; color:#aaa; font-size:0.9rem;">
                Written summary disabled — enable <em>Include written summary</em> on the start screen.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Individual judge results (always shown, one panel per judge) ──────────
    if judgment and judge_results:
        for jr in judge_results:
            j     = jr["judgment"]
            j_r   = j.get("reasoning", "")
            j_r_html = (
                f'<p style="color:#cfd8ff; margin:0.8rem 0 0.4rem;">{j_r}</p>'
                if j_r else ""
            )
            st.markdown(
                f"""
                <div class="arcade-panel" style="margin-top:8px; margin-bottom:18px;">
                    <div class="topic-label">{jr["role"]} Judge</div>
                    <div style="color:#aaa; font-size:0.8rem; margin-bottom:0.6rem;">
                        Model: <span style="color:#ffd54a;">{jr["model_label"]}</span>
                    </div>
                    <div class="winner-banner">Winner: {j.get("winner", "N/A")}</div>
                    {j_r_html}
                    {_build_scores_html(j.get("scores", {}))}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Averaged panel (only when multiple judges) ────────────────────────
        if num_judges > 1:
            st.markdown(
                f"""
                <div class="arcade-panel" style="margin-top:8px; margin-bottom:18px;">
                    <div class="topic-label">Panel Result · {num_judges} Judges · Averaged</div>
                    <div class="winner-banner">Winner: {judgment.get("winner", "N/A")}</div>
                    {_build_scores_html(judgment.get("scores", {}))}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Navigation ────────────────────────────────────────────────────────────
    bottom_left, bottom_right = st.columns(2)

    with bottom_left:
        if st.button("Back to Arena", use_container_width=True):
            st.session_state["stage"] = 4
            st.rerun()

    with bottom_right:
        if st.button("New Debate", type="primary", use_container_width=True):
            reset_game()
            st.rerun()