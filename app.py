import streamlit as st
from PIL import Image


def get_main_dir(depth: int = 0):  # nopep8
    """Get the main directory of the project."""
    import os
    import sys
    from os.path import dirname as up
    main_dir = os.path.dirname(os.path.abspath(__file__))
    for _ in range(depth):
        sys.path.append(up(main_dir))
        main_dir = up(main_dir)
    return main_dir


MAIN_DIR_PATH = get_main_dir(0)  # nopep8

from models.hugchat_llm import HugChatLLM
from models.local_gguf_llm import LocalGgufLLM
from models.rag import RAG
from models.simple_rag import SimpleRAG
from models.metadata_rag import MetadataRAG
from utils.custom_utils import get_filename, load_yaml, get_extensions_paths


st.set_page_config(page_title="MOLE", layout="wide")


def display_messages():
    st.subheader("Chat")
    for msg, user, stream in st.session_state["messages"]:
        if user == "user":
            st.session_state["container_message"].chat_message(user).write(msg)
        elif user == "assistant" and msg != "":
            if not stream:
                st.session_state["container_message"].chat_message(
                    user).write(msg)
            elif stream:
                st.session_state["container_message"].chat_message(
                    user).write_stream(msg)
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if (st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0):
        user_text = st.session_state["user_input"].strip()
        st.session_state["messages"].append((user_text, "user", False))
        if st.session_state["llm"] and st.session_state["llm"].loaded:
            with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
                if st.session_state["toggle_rag"]:
                    agent_text, stream = st.session_state["rag"].ask_stream(
                        user_text, web_search=st.session_state["toggle_web_search"]), True
                else:
                    agent_text, stream = st.session_state["llm"].ask_stream(
                        user_text, web_search=st.session_state["toggle_web_search"]), True
            st.session_state["messages"].append(
                (agent_text, "assistant", stream))
        else:
            st.session_state["messages"].append(
                ("Please load a model first.", "assistant", False))


def llm_loader():
    st.subheader("LLM Loader")
    with st.spinner(f"Loading"):
        st.toggle("Use Local LLM", key="toggle_local_llm")
        if st.session_state["llm"]:
            print("Current LLM is : ", st.session_state["llm"].name)
        else:
            print("No model loaded")
        if st.session_state["toggle_local_llm"]:
            if type(st.session_state["llm"]) != LocalGgufLLM:
                st.session_state["messages"] = []
                st.session_state["llm"] = LocalGgufLLM(
                    st.session_state["config"])
            gguf_paths = get_extensions_paths(
                st.session_state["config"]["model_directory"], "gguf")
            st.selectbox("Select a LLM model",
                         gguf_paths,
                         format_func=get_filename,
                         key="selectbox_local_llm",
                         label_visibility="collapsed")
            load_col, unload_col = st.columns((1, 3))
            load_col.button(label="Load", on_click=load_local_llm)
            unload_col.button(label="Unload", on_click=unload_local_llm)
            if st.session_state["llm"].loaded and st.session_state["llm"].name == get_filename(st.session_state["selectbox_local_llm"]):
                st.write("LLM loaded \u2705")
            else:
                st.session_state["load_model_spinner"] = st.empty()

        else:
            if type(st.session_state["llm"]) == LocalGgufLLM:
                st.session_state["llm"].unload_model()
            if type(st.session_state["llm"]) != HugChatLLM:
                st.session_state["messages"] = []
                st.session_state["llm"] = HugChatLLM(
                    st.session_state["config"])
                st.session_state["llm"].load_model(0)
            st.selectbox("Select a LLM model",
                         st.session_state["llm"].available_models,
                         key="selectbox_hugchat_llm",
                         format_func=lambda k: k.displayName,
                         on_change=change_hugchat_llm,
                         label_visibility="collapsed")
            if st.session_state["llm"].loaded:
                st.write("LLM loaded \u2705")
            else:
                st.session_state["load_model_spinner"] = st.empty()
        st.toggle("Web search", key="toggle_web_search")


def rag_loader():
    st.subheader("RAG Loader")
    with st.spinner(f"Loading"):
        st.toggle("Use RAG", key="toggle_rag")
        if st.session_state["toggle_rag"] and st.session_state["rag"] is None:
            st.session_state["messages"] = []
            st.session_state["rag"] = MetadataRAG(
                st.session_state["llm"], st.session_state["config"]
            )
            st.session_state["rag"].load_collection("test_collection")
        elif not st.session_state["toggle_rag"] and st.session_state["rag"] is not None:
            st.session_state["messages"] = []
            del st.session_state["rag"]
            st.session_state["rag"] = None
        if st.session_state["toggle_rag"]:
            st.write("Number of documents:",
                     st.session_state["rag"].collection.count())


def load_local_llm():
    st.session_state["messages"] = []

    gguf_path = st.session_state["selectbox_local_llm"]
    gguf_paths = get_extensions_paths(
        st.session_state["config"]["model_directory"], "gguf")
    with st.session_state["load_model_spinner"], st.spinner(f"Loading"):
        st.session_state["llm"].load_model(gguf_paths.index(gguf_path))
        print("Changed LLM to : ", st.session_state["llm"].name)


def unload_local_llm():
    st.session_state["messages"] = []
    st.session_state["llm"].unload_model()


def change_hugchat_llm():
    st.session_state["messages"] = []

    available = st.session_state["llm"].available_models
    llm_number = available.index(st.session_state["selectbox_hugchat_llm"])
    st.session_state["llm"].load_model(llm_number)
    print("Changed LLM to : ", st.session_state["llm"].name)


def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["llm"] = None
        st.session_state["rag"] = None
        st.session_state["config"] = load_yaml(MAIN_DIR_PATH + "./config.yaml")

    st.header("MOLE")

    with st.sidebar:
        logo = Image.open(MAIN_DIR_PATH + r"\assets\logo.png")
        st.image(logo, width=300)
        llm_loader()
        rag_loader()

    st.session_state["container_message"] = st.container(height=600)
    display_messages()
    st.chat_input("Message", key="user_input", on_submit=process_input)


if __name__ == "__main__":
    page()
