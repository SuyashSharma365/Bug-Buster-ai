"""Streamlit chat UI for Bug Buster AI."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import os
import queue
import threading
from collections.abc import Iterator
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage

from agent.graph import stream_agent
from ingestion.embed_and_store_repo import ingest_repository

load_dotenv()


def _produce_events(question: str, provider: str, events: queue.Queue[tuple[str, Any] | None]) -> None:
    """Run the async MCP/LangGraph stream on a worker thread for Streamlit."""
    async def consume() -> None:
        try:
            async for update in stream_agent(question, provider):
                for state in update.values():
                    for message in state.get("messages", []):
                        if isinstance(message, ToolMessage):
                            events.put(("tool", message.name or "search_docs"))
                        elif isinstance(message, AIMessage):
                            for call in message.tool_calls:
                                events.put(("tool", call.get("name", "tool")))
                            if isinstance(message.content, str) and message.content.strip():
                                events.put(("text", message.content))
        except Exception as error:
            events.put(("error", error))
        finally:
            events.put(None)

    asyncio.run(consume())


def stream_response(question: str, provider: str) -> Iterator[str]:
    """Adapt async graph events to Streamlit's synchronous write_stream API."""
    events: queue.Queue[tuple[str, Any] | None] = queue.Queue()
    worker = threading.Thread(target=_produce_events, args=(question, provider, events), daemon=True)
    worker.start()
    seen_tools: list[str] = []
    st.session_state["last_tools"] = seen_tools
    while True:
        event = events.get()
        if event is None:
            break
        kind, value = event
        if kind == "error":
            raise value
        if kind == "tool":
            if value not in seen_tools:
                seen_tools.append(value)
            continue
        if kind == "text":
            yield value


def main() -> None:
    st.set_page_config(page_title="Bug Buster AI", page_icon="B", layout="wide")
    st.title("Bug Buster AI")
    st.caption("An MCP-powered assistant for document Q&A and GitHub code review — finds bugs, security issues, and code smells using retrieval-augmented generation.")

    with st.sidebar:
        st.header("Configuration")
        provider = st.selectbox(
            "LLM provider",
            options=["groq", "hf", "anthropic"],
            index=["groq", "hf", "anthropic"].index(os.getenv("DOCMIND_LLM_PROVIDER", "groq")),
        )
        repo_url = st.text_input("GitHub repo URL", placeholder="https://github.com/user/repository")
        if st.button("Ingest Repository"):
            if not repo_url.strip():
                st.warning("Enter a GitHub repository URL first.")
            else:
                with st.spinner("Cloning and indexing repository..."):
                    try:
                        report = ingest_repository(repo_url.strip(), reset=True)
                        st.success(
                            f"Repository ingested: {len(report.documents)} files processed, "
                            f"{report.chunks_stored} chunks stored."
                        )
                        if report.skipped_files:
                            st.warning(f"Skipped {len(report.skipped_files)} files due to size or read errors.")
                    except Exception as error:
                        st.error(f"Repository ingestion failed: {error}")
        st.info("Add documents to data/ or ingest a GitHub repository before asking document or code questions.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask about your documents")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                answer = st.write_stream(stream_response(question, provider))
                tools = st.session_state.get("last_tools", [])
                if tools:
                    st.caption(f"Tools used: {', '.join(tools)}")
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as error:
                message = f"Unable to answer: {error}"
                st.error(message)
                st.session_state.messages.append({"role": "assistant", "content": message})


if __name__ == "__main__":
    main()
