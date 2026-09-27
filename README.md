# DocMind

DocMind is a local-first GenAI RAG assistant. It indexes PDFs, Markdown, and text files into ChromaDB, exposes semantic search through a custom MCP server, and lets a LangGraph ReAct agent decide when to retrieve context before answering in Streamlit.

## Architecture

```text
data/ -> ingestion/loader.py -> chunker.py -> embed_and_store.py -> ChromaDB
                                                                    ^
Streamlit -> LangGraph agent -> langchain-mcp-adapters -> MCP stdio server
                                                            -> search_docs
```

- `ingestion/loader.py` loads `.pdf`, `.md`, and `.txt` files.
- `ingestion/chunker.py` uses LangChain's `RecursiveCharacterTextSplitter`.
- `ingestion/embed_and_store.py` uses `sentence-transformers` with `BAAI/bge-small-en-v1.5` by default. Set `DOCMIND_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5` for the larger model.
- `mcp_server/docs_server.py` is an official Python MCP SDK server using stdio transport. Its `search_docs` tool embeds a query and searches persistent ChromaDB.
- `agent/graph.py` starts the MCP server through `langchain-mcp-adapters`, loads its tools asynchronously, and builds a LangGraph ReAct agent.
- `agent/llm_config.py` supports local Transformers generation (`hf`) and API-backed `openai` or `anthropic` chat models.
- `ui/app.py` provides a Streamlit chat UI and displays the MCP tools used for each answer.

## Setup

Requires Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and choose a provider:

```dotenv
DOCMIND_LLM_PROVIDER=openai
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-4o-mini
```

For Anthropic, use `DOCMIND_LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`. For a local Hugging Face model, use `DOCMIND_LLM_PROVIDER=hf`; the first run downloads the model configured by `DOCMIND_HF_MODEL`. Local tool calling depends on the selected model and chat template, so an API provider is the simplest starting point.

## Ingest documents

Place source files in `data/`, then run:

```powershell
python -m ingestion.embed_and_store --reset
```

The default embedding model is smaller for practical local startup. The larger `BAAI/bge-large-en-v1.5` model can improve retrieval quality at a higher memory cost. Chroma data is persisted under `storage/chroma/`.

## Run the MCP server

The agent starts this stdio server automatically. To start it directly for inspection:

```powershell
python -m mcp_server.docs_server
```

Do not print ordinary logs to stdout while using stdio MCP transport; stdout is reserved for MCP JSON-RPC messages.

## Run the assistant

```powershell
streamlit run ui/app.py
```

Or ask one question from the command line:

```powershell
python -m agent "What does the documentation say about authentication?" --provider openai
```

## Configuration

See `.env.example` for all supported settings. Important values include `DOCMIND_LLM_PROVIDER`, `DOCMIND_HF_MODEL`, `DOCMIND_EMBEDDING_MODEL`, `DOCMIND_CHROMA_DIR`, and `DOCMIND_COLLECTION`.

## Troubleshooting

- `ChromaDB is empty`: put supported files in `data/` and rerun ingestion.
- Missing API key: select `hf` for local inference or set the key required by the selected API provider.
- Local model memory errors: use a smaller instruction-tuned model in `DOCMIND_HF_MODEL`, or select `openai`/`anthropic`.
- MCP startup errors: run the command from the project root so Python can import `mcp_server.docs_server`.
