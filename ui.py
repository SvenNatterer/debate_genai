import json
import time
from typing import Dict
import base64
from pathlib import Path
import streamlit as st

from config import PHILOSOPHER_LIBRARY, STRATEGY_OPTIONS, random_topic
from debate_engine_cloud import (
    judge_debate,
    run_debate,
    summarize_debate,
    set_active_model,
    get_active_model_label,
)


# Available models shown in the UI selector
MODEL_OPTIONS = {
    "GPT-5-chat  (Platform)":         ("custom", "gpt-5-chat"),
    "GPT-4.1 mini  (Platform)":       ("custom", "gpt-4.1-mini"),
    "DeepSeek-V3.2  (Platform)":      ("custom", "DeepSeek-V3.2"),
    "Mistral Large 3  (Platform)":    ("custom", "mistral-Large-3"),
    "Mistral Small  (Platform)":      ("custom", "mistral-small-2503"),
    "Llama 4 Maverick  (Platform)":   ("custom", "Llama-4-Maverick-17B-128E-Instruct-FP8"),
    "Llama 3.3 70B  (Platform)":      ("custom", "Llama-3.3-70B-Instruct"),
    "Phi-4 mini  (Platform)":         ("custom", "Phi-4-mini-reasoning"),

}


def image_to_data_uri(image_path: str) -> str:
    image_bytes = Path(image_path).read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def reset_game() -> None:
    st.session_state["stage"] = 0
    st.session_state["selected_topic"] = random_topic()
    st.session_state["rounds"] = 2
    st.session_state["player1_strategy"] = STRATEGY_OPTIONS[0]
    st.session_state["player2_strategy"] = STRATEGY_OPTIONS[0]
    st.session_state["agent1_philosopher_name"] = PHILOSOPHER_LIBRARY["socrates"]["name"]
    st.session_state["agent2_philosopher_name"] = PHILOSOPHER_LIBRARY["nietzsche"]["name"]
    st.session_state.setdefault("selected_model_label", list(MODEL_OPTIONS.keys())[0])
    
    for key in [
        "transcript",
        "judgment",
        "summary",
        "topic",
        "agent_configs",
    ]:
        st.session_state.pop(key, None)


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
    st.session_state.setdefault("selected_model_label", list(MODEL_OPTIONS.keys())[0])


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
        label = get_active_model_label()
        st.markdown(f'<div class="small-status">Azure Active · {_html.escape(label)}</div>', unsafe_allow_html=True)
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


def render_scores(scores: Dict[str, Dict[str, int]]) -> None:
    for name, vals in scores.items():
        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-name">{name}</div>
                <div class="score-grid">
                    <div>Logic: {vals['logic']}</div>
                    <div>Relevance: {vals['relevance']}</div>
                    <div>Rebuttal: {vals['rebuttal']}</div>
                    <div>Fairness: {vals['fairness']}</div>
                    <div>Total: {vals['total']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


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

    model_label = st.selectbox(
        "🤖 LLM Model",
        options=list(MODEL_OPTIONS.keys()),
        index=list(MODEL_OPTIONS.keys()).index(st.session_state["selected_model_label"]),
        key="selected_model_label",
        help="Choose which cloud model powers the debate. Requires the matching API key in your .env file.",
    )

    provider, model = MODEL_OPTIONS[model_label]
    set_active_model(provider, model)

    st.markdown(
        f"<div class='small-status' style='margin-bottom:1rem;'>Active: {provider.upper()} / {model}</div>",
        unsafe_allow_html=True,
    )

    if st.button("Start Game", type="primary", use_container_width=True):
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

    philosopher_names = [v["name"] for v in PHILOSOPHER_LIBRARY.values()]

    col_a, col_b = st.columns(2)
    with col_a:
        st.selectbox(
            "Player 1 Philosopher",
            options=philosopher_names,
            index=philosopher_names.index(st.session_state["agent1_philosopher_name"]),
            key="agent1_philosopher_name",
        )
    with col_b:
        st.selectbox(
            "Player 2 Philosopher",
            options=philosopher_names,
            index=philosopher_names.index(st.session_state["agent2_philosopher_name"]),
            key="agent2_philosopher_name",
        )

    strategy_col1, strategy_col2 = st.columns(2)
    with strategy_col1:
        st.selectbox(
            "Player 1 Strategy",
            options=STRATEGY_OPTIONS,
            key="player1_strategy",
        )
    with strategy_col2:
        st.selectbox(
            "Player 2 Strategy",
            options=STRATEGY_OPTIONS,
            key="player2_strategy",
        )

    render_character_cards()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back", use_container_width=True):
            st.session_state["stage"] = 1
            st.rerun()
    with col2:
        if st.button("Continue", type="primary", use_container_width=True):
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

    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    handle_run_debate()
    st.session_state["stage"] = 4
    st.rerun()


def handle_run_debate() -> None:
    agent_configs = current_agent_configs()
    transcript = run_debate(
        st.session_state["selected_topic"],
        st.session_state["rounds"],
        [
            st.session_state["player1_strategy"],
            st.session_state["player2_strategy"],
        ],
        agent_configs,
    )
    judgment = judge_debate(st.session_state["selected_topic"], transcript)
    summary = summarize_debate(st.session_state["selected_topic"], transcript, judgment)

    st.session_state["transcript"] = transcript
    st.session_state["judgment"] = judgment
    st.session_state["summary"] = summary
    st.session_state["topic"] = st.session_state["selected_topic"]
    st.session_state["agent_configs"] = agent_configs


def render_arena_stage() -> None:
    if "transcript" not in st.session_state:
        handle_run_debate()

    transcript = st.session_state.get("transcript")
    judgment = st.session_state.get("judgment")

    topic = st.session_state.get("topic", st.session_state.get("selected_topic", ""))

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
        st.markdown(
            """
            <div class="arcade-panel" style="margin-top:8px; margin-bottom:18px;">
                <div class="topic-label">Judge Result</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="winner-banner">
                Winner: {judgment.get("winner", "N/A")}
            </div>
            """,
            unsafe_allow_html=True,
        )

        reasoning = judgment.get("reasoning", "")
        if reasoning:
            st.write(reasoning)

        scores = judgment.get("scores", {})
        if scores:
            render_scores(scores)

        st.markdown("</div>", unsafe_allow_html=True)

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
    summary = st.session_state.get("summary")

    st.markdown(
        """
        <div class="arcade-panel" style="margin-bottom:18px;">
            <div class="topic-label">Summary</div>
        """,
        unsafe_allow_html=True,
    )

    if summary:
        st.write(summary)
    else:
        st.markdown(
            """
            <div class="score-card">
                <div style="color:#e8ecff; line-height:1.6;">Summary is not available yet.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


    bottom_left, bottom_right = st.columns(2)

    with bottom_left:
        if st.button("Back to Arena", use_container_width=True):
            st.session_state["stage"] = 4
            st.rerun()

    with bottom_right:
        if st.button("New Debate", type="primary", use_container_width=True):
            reset_game()
            st.rerun()