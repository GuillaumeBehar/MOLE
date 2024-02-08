import os
import tempfile
import streamlit as st
from streamlit_chat import message
from rag import *
from local_llm import *

st.set_page_config(page_title="ChatBot", layout="wide")


def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if (
        st.session_state["user_input"]
        and len(st.session_state["user_input"].strip()) > 0
    ):
        user_text = st.session_state["user_input"].strip()
        st.session_state["messages"].append((user_text, True))
        if st.session_state["ChatBot"]:
            with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
                agent_text = st.session_state["ChatBot"].ask(user_text)
            st.session_state["messages"].append((agent_text, False))
        else:
            st.session_state["messages"].append(("Please load a model first.", False))


def read_and_save_file():
    if st.session_state["ChatBot"]:
        st.session_state["ChatBot"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(
            f"Ingesting {file.name}"
        ):
            st.session_state["ChatBot"].ingest(file_path)
        os.remove(file_path)


def load_model():
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""
    st.session_state["LocalLLM"] = None
    st.session_state["ChatBot"] = None

    gguf_path = st.session_state["selectbox_model"]
    with st.session_state["load_model_spinner"], st.spinner(
        f"Loading {get_filename(gguf_path)}"
    ):
        st.session_state["LocalLLM"] = LocalLLM(gguf_path)
        st.session_state["ChatBot"] = ChatBot(st.session_state["LocalLLM"])


def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["LocalLLM"] = None
        st.session_state["ChatBot"] = None

    st.header("Local Chat Bot")

    st.subheader("Select a model")

    col1, col2, col3 = st.columns((2, 1, 2))
    gguf_paths = get_gguf_paths()

    col1.selectbox(
        "Select a model",
        gguf_paths,
        format_func=get_filename,
        key="selectbox_model",
        label_visibility="collapsed",
    )

    col2.button(label="Load Model", on_click=load_model)

    st.session_state["load_model_spinner"] = col3.empty()

    st.subheader("Upload a document")
    st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)


if __name__ == "__main__":
    page()
