import streamlit as st

from styles import inject_arcade_css
from debate_engine_cloud import get_client_and_model

from ui import (
    ensure_session_state,
    render_arena_stage,
    render_character_stage,
    render_header,
    render_mode_status,
    render_start_screen,
    render_summary_stage,
    render_top_bar,
    render_topic_stage,
    render_versus_stage,
)


st.set_page_config(page_title="Philosopher Arena", page_icon="🧠", layout="wide")


def main() -> None:
    ensure_session_state()
    inject_arcade_css()

    render_header()

    base_url, model, status_message = get_client_and_model()
    cloud_active = base_url is not None and model is not None
    render_mode_status(cloud_active, status_message)
    render_top_bar()

    stage = st.session_state["stage"]

    if stage == 0:
        render_start_screen()
    elif stage == 1:
        render_topic_stage()
    elif stage == 2:
        render_character_stage()
    elif stage == 3:
        render_versus_stage()
    elif stage == 4:
        render_arena_stage()
    elif stage == 5:
        render_summary_stage()


if __name__ == "__main__":
    main()
