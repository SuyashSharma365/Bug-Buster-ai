# Bug Buster AI

Bug Buster AI is an MCP-powered retrieval-augmented generation assistant with two capabilities: it answers questions over ingested documents and reviews GitHub repositories for bugs, security issues, and code quality problems. It uses Hugging Face embeddings, ChromaDB, LangChain and LangGraph orchestration, and a Groq-hosted LLM to retrieve relevant evidence before answering.

## Architecture

- **Hugging Face embeddings**: Sentence Transformers embed documents and source-code chunks for semantic retrieval.
- **ChromaDB vector store**: Local persistent storage keeps document chunks in `docmind_documents` and repository code in `code_chunks`.
- **MCP server**: `mcp_server/docs_server.py` exposes `search_docs` and `search_code` over stdio.
- **LangGraph ReAct agent**: Chooses the appropriate MCP search tool and grounds responses in retrieved content.
- **Groq LLM**: Provides the default tool-calling model, configured through `GROQ_API_KEY` and `GROQ_MODEL`.
- **Streamlit UI**: Provides chat, provider selection, repository ingestion, and tool-use visibility.

## Setup

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and fill in your Groq credentials:

```dotenv
DOCMIND_LLM_PROVIDER=groq
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=openai/gpt-oss-120b
```

The default embedding model is `BAAI/bge-small-en-v1.5`. Set `DOCMIND_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5` when higher retrieval quality is worth the additional memory use.

## Usage

### Ingest documents

Place PDF, Markdown, or text files in `data/`, then run:

```powershell
python -m ingestion.embed_and_store
```

The document chunks are stored in the document collection used by `search_docs`.

### Ingest a GitHub repository

Clone and index a repository into the separate `code_chunks` collection:

```powershell
python -m ingestion.embed_and_store_repo --repo-url <url> --reset
```

The Streamlit sidebar also provides a **GitHub repo URL** field and **Ingest Repository** button that calls the same ingestion function directly. The ingestion process skips generated/vendor directories, unsupported extensions, and source files larger than 500 KB.

### Launch the app

```powershell
streamlit run ui/app.py
```

Ask questions about indexed documents or request a code review, for example:

- `What does the documentation say about authentication?`
- `Review the repository for security vulnerabilities in the login flow.`
- `Find error-handling gaps in the API client.`

## Example

Question:

```text
Review the repository for bugs in the authentication code.
```

A grounded response may look like:

```text
Issue: User input is interpolated directly into the SQL query, allowing SQL injection.
File: src/auth.py
Chunk: 2
Recommendation: Use a parameterized query and validate the input before execution.
```

Bug Buster AI reports only issues visible in retrieved code and cites the relevant `file_path` and `chunk_index`.

## Tech Stack

- Hugging Face Transformers and Sentence Transformers
- LangChain
- LangGraph
- Model Context Protocol (MCP)
- ChromaDB
- Groq
- Streamlit
