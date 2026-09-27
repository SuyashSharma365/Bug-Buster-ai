"""LangGraph ReAct agent wired to the Bug Buster AI MCP server."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from agent.llm_config import get_llm

load_dotenv()

SYSTEM_PROMPT = """You are Bug Buster, an assistant that can either answer questions about indexed documents or review ingested source code for bugs and issues.

Use search_docs when the question is about the user's indexed text documents.
Use search_code when the question is about a GitHub repository's source code, bugs, security issues, code quality, or specific files/functions.

When using search_code, act as a senior code reviewer: identify concrete bugs, security vulnerabilities, error-handling gaps, and code smells. Always cite the exact file_path (and chunk_index if relevant) for every issue you report. Do not invent issues that aren't visible in the retrieved code — if the retrieved code looks fine, say so honestly rather than fabricating problems.

If a tool reports that no data has been ingested yet, tell the user to run the appropriate ingestion script first, and tell them which one (document ingestion vs repo ingestion).

Answer directly only for greetings or questions unrelated to both document and code collections."""


@dataclass
class AgentResult:
    """Final response plus tool names observed during one graph run."""

    answer: str
    tool_calls: list[str] = field(default_factory=list)



def mcp_server_config() -> dict[str, dict[str, Any]]:
    """Describe the stdio MCP process consumed by langchain-mcp-adapters."""
    module = os.getenv("DOCMIND_MCP_MODULE", "mcp_server.docs_server")
    return {
        "docs": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", module],
        }
    }


async def build_agent(provider: str | None = None) -> tuple[Any, MultiServerMCPClient]:
    """Connect to the MCP server and build a LangGraph ReAct agent."""
    selected_provider = provider or os.getenv("DOCMIND_LLM_PROVIDER", "groq")
    client = MultiServerMCPClient(mcp_server_config())
    tools = await client.get_tools()
    if not tools:
        raise RuntimeError("The MCP server exposed no tools. Check the server and ingestion state.")
    llm = get_llm(selected_provider)
    try:
        llm_with_tools = llm.bind_tools(tools)
    except NotImplementedError as error:
        raise RuntimeError(
            f"The {selected_provider} model does not support tool calling. Use an API provider or a tool-capable HF model."
        ) from error
    return create_react_agent(llm_with_tools, tools, prompt=SYSTEM_PROMPT), client


async def run_agent(question: str, provider: str | None = None) -> AgentResult:
    """Run one question through the async LangGraph/MCP pipeline."""
    if not question.strip():
        raise ValueError("Question must not be empty")
    graph, _client = await build_agent(provider)
    final_answer = ""
    tool_calls: list[str] = []
    async for update in graph.astream(
        {"messages": [HumanMessage(content=question)]},
        stream_mode="updates",
    ):
        for state in update.values():
            for message in state.get("messages", []):
                if isinstance(message, ToolMessage):
                    continue
                if isinstance(message, AIMessage):
                    final_answer = message.content if isinstance(message.content, str) else str(message.content)
                    for call in message.tool_calls:
                        name = call.get("name")
                        if name and name not in tool_calls:
                            tool_calls.append(name)
    if not final_answer:
        final_answer = "I could not produce an answer."
    return AgentResult(answer=final_answer, tool_calls=tool_calls)


async def stream_agent(question: str, provider: str | None = None) -> AsyncIterator[dict[str, Any]]:
    """Yield graph updates for a UI that wants incremental output and tool visibility."""
    if not question.strip():
        raise ValueError("Question must not be empty")
    graph, _client = await build_agent(provider)
    async for update in graph.astream(
        {"messages": [HumanMessage(content=question)]},
        stream_mode="updates",
    ):
        yield update
