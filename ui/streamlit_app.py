import streamlit as st

from app.main import run_once
from app.session_manager import SessionManager


st.set_page_config(page_title="AI Blog Generator", page_icon="📝", layout="wide")
st.title("AI Blog Generator for Backoffice")
st.caption("Test truc tiep flow: Parse -> Retrieve -> Generate")


if "session_manager" not in st.session_state:
    st.session_state.session_manager = SessionManager()

if "messages" not in st.session_state:
    st.session_state.messages = []


def _push_turn(user_text: str, assistant_text: str) -> None:
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.messages.append({"role": "assistant", "content": assistant_text})


def _render_messages() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


with st.sidebar:
    st.subheader("Quick Actions")
    if st.button("Clear chat session", use_container_width=True):
        st.session_state.session_manager = SessionManager()
        st.session_state.messages = []
        st.rerun()

    st.subheader("Sample Prompts")
    sample_prompts = [
        "Write a blog about what a police check is for first-time job applicants, in a clear and professional tone.",
        "Make it shorter",
        "Rewrite this in a friendly tone",
    ]

    for idx, item in enumerate(sample_prompts, start=1):
        st.caption(f"{idx}. {item}")


st.markdown("### Chat Console")
_render_messages()

prompt = st.chat_input("Nhap prompt de test prototype...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Dang xu ly..."):
            output = run_once(prompt, st.session_state.session_manager)
        st.markdown(output)

    _push_turn(prompt, output)

if not st.session_state.messages:
    st.info("Nhap prompt o o chat ben duoi de bat dau test. Goi y: thu prompt create blog, sau do nhap 'Make it shorter'.")
