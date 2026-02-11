"""Streamlit chat UI. Run with: streamlit run app.py"""

import os
import streamlit as st
from rag_pipeline import RAGPipeline

st.set_page_config(page_title="Business Data Chat", page_icon="📊", layout="wide")

# Sidebar
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input(
        "Groq API Key", type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Get one at https://console.groq.com",
    )
    model = st.selectbox("Model", [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ])
    show_debug = st.checkbox("Show generated code & raw data", value=False)
    st.markdown("---")
    st.caption("Tables: Clients (20), Invoices (40), LineItems (96)")


@st.cache_resource
def get_pipeline(_api_key: str, _model: str) -> RAGPipeline:
    return RAGPipeline(api_key=_api_key, model=_model)


st.title("📊 Business Data Chat")
st.caption("Ask questions about clients, invoices, and line items.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("code") and show_debug:
            with st.expander("Generated code"):
                st.code(msg["code"], language="python")
            with st.expander("Raw data"):
                st.text(msg["data"])

if prompt := st.chat_input("Ask a question about the data..."):
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # build history from prior turns so follow-ups work
    chat_history = []
    msgs = st.session_state.messages
    i = 0
    while i < len(msgs) - 1:
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
            chat_history.append({
                "question": msgs[i]["content"],
                "answer": msgs[i + 1]["content"],
                "code": msgs[i + 1].get("code", ""),
                "data": msgs[i + 1].get("data", ""),
            })
            i += 2
        else:
            i += 1

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            pipeline = get_pipeline(api_key, model)
            result = pipeline.ask(prompt, history=chat_history)

        st.markdown(result["answer"])

        if result["error"]:
            st.error(f"Pipeline error: {result['error']}")
        if show_debug and result["code"]:
            with st.expander("Generated code"):
                st.code(result["code"], language="python")
            with st.expander("Raw data"):
                st.text(result["data"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "code": result.get("code", ""),
        "data": result.get("data", ""),
    })
