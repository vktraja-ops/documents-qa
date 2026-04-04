"""
Capstone Project — Documents Q&A
=================================
A Retrieval-Augmented Generation (RAG) application built with:
  * Streamlit      — web UI
  * Google Gemini  — LLM reasoning + text embeddings
  * ChromaDB       — in-memory vector store for semantic search
"""

import os
import re
import textwrap

import chromadb
import pandas as pd
import streamlit as st
from chromadb.utils.embedding_functions import EmbeddingFunction
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Load GOOGLE_API_KEY (and any other vars) from a local .env file. This keeps secrets out of source code.
load_dotenv()

GEMINI_MODEL    = "gemini-2.5-flash"    # LLM used for answer generation
EMBED_MODEL     = "gemini-embedding-001" # Model used to create vector embeddings
CHUNK_SIZE      = 512                    # Maximum characters per text chunk
CHUNK_OVERLAP   = 100                    # Characters shared between adjacent chunks
TOP_K           = 5                      # Number of chunks to retrieve per query
COLLECTION_NAME = "enterprise_docs"      # ChromaDB collection name

# Single shared Gemini client — reused by both the embedding function and the LLM calls.
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# ---------------------------------------------------------------------------
# Gemini embedding function (ChromaDB-compatible)
# ---------------------------------------------------------------------------

class GeminiEmbeddingFunction(EmbeddingFunction):
    # Wraps the Gemini embedding API in the interface ChromaDB expects. ChromaDB calls this whenever it needs to embed documents or queries.
    def __init__(self):
        # ChromaDB requires a no-argument constructor on embedding functions.
        pass

    def __call__(self, input: list) -> list:
        # Embed a batch of strings in a single API round-trip.
        # task_type="RETRIEVAL_DOCUMENT" tells Gemini these are passages being indexed (as opposed to a search query).
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=input,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        if response.embeddings is None:
            raise ValueError("Gemini returned no embeddings")

        # e.values is a protobuf repeated field; convert to plain Python list so ChromaDB receives the concrete type it expects.
        return [list(e.values) for e in response.embeddings if e.values is not None]


# ---------------------------------------------------------------------------
# Document ingestion helpers
# ---------------------------------------------------------------------------

def extract_text(uploaded_file) -> str:
    """
    Convert an uploaded file to a single plain-text string.
    Supported formats - .txt, .pdf, .csv, .xlsx - are handled with different methods:
    Returns an empty string for unrecognised file types so callers can safely check `if not raw_text.strip()` without raising an exception.
    """
    name = uploaded_file.name

    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")

    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        # Some pages may return None from extract_text(); `or ""` guards against that.
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return df.to_string(index=False)

    if name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file, engine="openpyxl")
        return df.to_string(index=False)

    return ""  # Unsupported type — ingest_document will skip it gracefully


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split a long string into overlapping fixed-size character chunks.
    Overlap prevents a sentence or key phrase that falls exactly on a chunk boundary from being split across two chunks with no shared context.
    """
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:                        # Skip chunks that are only whitespace
            chunks.append(chunk)
        start += size - overlap          # Step forward by (size - overlap)
    return chunks


def ingest_document(uploaded_file, collection) -> int:
    """
    Full ingestion pipeline for a single uploaded file:
      1. Extract raw text from the file.
      2. Split the text into overlapping chunks.
      3. Generate a stable, unique ID for each chunk based on the filename.
      4. Store chunks + metadata in ChromaDB (embeddings are created automatically by GeminiEmbeddingFunction when collection.add() is called).
      5. Returns the number of chunks added (0 if the file contained no usable text).
    """
    raw_text = extract_text(uploaded_file)
    if not raw_text.strip():
        return 0  # Nothing to ingest — skip silently

    chunks = chunk_text(raw_text)

    # Build a filesystem-safe base ID from the filename (replace non-word chars with _).
    base_id   = re.sub(r"\W+", "_", uploaded_file.name)
    ids       = [f"{base_id}_chunk_{i}" for i in range(len(chunks))]

    # Metadata is stored alongside each chunk so we can show the source filename and chunk index in the UI when displaying retrieved context.
    metadatas = [{"source": uploaded_file.name, "chunk": i} for i in range(len(chunks))]

    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


# ---------------------------------------------------------------------------
# AI Agent  (Plan -> Retrieve -> Reason -> Validate)
# ---------------------------------------------------------------------------

# System prompt that instructs Gemini to follow a structured reasoning process before producing an answer.
# This encourages the model to be methodical and to stay grounded in the provided context rather than hallucinating.
AGENT_SYSTEM_PROMPT = textwrap.dedent("""
    1. PLAN: Identify what information is needed to answer the question.
    2. REASON: Analyse the retrieved context carefully.
    3. ANSWER: Give a clear, concise, factual answer grounded only in the context.
""").strip()


def retrieve_context(query: str, collection, top_k: int = TOP_K) -> list[dict]:
    """
    Semantic retrieval: embed the query then find the top-k nearest chunks.
    Steps:
    -----
    1. Embed the query with task_type="RETRIEVAL_QUERY" (a different Gemini optimisation hint than "RETRIEVAL_DOCUMENT" used at index time).
    2. Run a nearest-neighbour search in ChromaDB using cosine distance.
    3. Convert raw distances to relevance scores (score = 1 - distance), where 1.0 is a perfect match and 0.0 is completely dissimilar.
    4. Returns a list of dicts with keys: text, source, score.
    """
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[query],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    if not response.embeddings:
        raise ValueError("Gemini returned no embeddings for the query.")

    # Guard against the SDK returning an embedding object with a None values field.
    raw_values = response.embeddings[0].values
    if raw_values is None:
        raise ValueError("Gemini returned an embedding with no values.")
    query_embedding: list[float] = list(raw_values)

    # n_results must not exceed the total number of stored chunks.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB returns nested lists (one list per query); [0] unwraps the single-query results into flat lists of documents / metadata / distances.
    chunks: list[dict] = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        chunks.append({
            "text":   doc,
            "source": meta.get("source", ""),
            "score":  1 - dist,              # Convert distance -> similarity score
        })
    return chunks


def build_prompt(query: str, context_chunks: list[dict]) -> str:
    """
    Assemble the final prompt sent to the LLM.
    Each chunk is prefixed with its source filename and relevance score so the model can weight evidence, so the answer can be traced back to a specific document.
    """
    context_text = "\n\n---\n\n".join(
        f"[Source: {c['source']} | Relevance: {c['score']:.2f}]\n{c['text']}"
        for c in context_chunks
    )
    return f"CONTEXT:\n{context_text}\n\nQUESTION:\n{query}"


def validate_response(response: str, context_chunks: list[dict]) -> tuple[str, str]:
    """
    Lightweight hallucination guardrail based on vocabulary overlap.
    Calculates what fraction of the meaningful words in the model's answer also appear somewhere in the retrieved context. 
    A low overlap ratio suggests the model may have introduced information not present in the documents. This is a heuristic, not a guarantee.
    Returns (answer, warning_message) where warning_message is "" if all good.
    """
    all_context    = " ".join(c["text"].lower() for c in context_chunks)
    response_words = set(re.findall(r"\b\w{5,}\b", response.lower()))
    context_words  = set(re.findall(r"\b\w{5,}\b", all_context))
    overlap_ratio  = len(response_words & context_words) / max(len(response_words), 1)

    warning = (
        "**Low grounding score** — the answer may not be fully supported by the documents."
        if overlap_ratio < 0.10 else ""
    )
    return response, warning


def run_agent(query: str, collection) -> dict:
    """
    Orchestrates the full RAG pipeline for a single user query:
      1. Retrieve — find the most relevant chunks from ChromaDB.
      2. Reason   — send chunks + query to Gemini for answer generation.
      3. Validate — run the hallucination guardrail on the raw answer.
      4. Returns a dict with keys: answer (str), chunks (list), warning (str).
    """
    if collection.count() == 0:
        return {
            "answer":  "No documents have been ingested yet. Please upload files first.",
            "chunks":  [],
            "warning": "",
        }

    # Step 1 — Retrieve relevant context from the vector store.
    chunks = retrieve_context(query, collection)

    # Step 2 — Ask Gemini to reason over the context and answer the question.
    prompt = build_prompt(query, chunks)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        config=types.GenerateContentConfig(system_instruction=AGENT_SYSTEM_PROMPT),
        contents=prompt,
    )
    if response.text is None:
        return {"answer": "LLM returned an empty response.", "chunks": chunks, "warning": ""}

    raw_answer = response.text.strip()

    # Step 3 — Validate grounding before returning to the UI.
    answer, warning = validate_response(raw_answer, chunks)
    return {"answer": answer, "chunks": chunks, "warning": warning}


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def init_session_state():
    # Initialise all Streamlit session state keys exactly once per browser session.
    if "chroma_client" not in st.session_state:
        # EphemeralClient stores everything in memory — data is lost on page refresh.
        st.session_state["chroma_client"] = chromadb.EphemeralClient()
        st.session_state["chroma_client"].get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=GeminiEmbeddingFunction(),
        )

    if "chat_history" not in st.session_state:
        # Each entry is {"role": "user"|"assistant", "content": str}
        st.session_state["chat_history"] = []


def get_collection():
    """
    Retrieve the ChromaDB collection from session state.
    Always passes a fresh GeminiEmbeddingFunction instance so ChromaDB can embed any new documents or queries even after a hot-reload.
    """
    return st.session_state["chroma_client"].get_collection(
        name=COLLECTION_NAME,
        embedding_function=GeminiEmbeddingFunction(),
    )


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def render_sidebar():
    """
    Sidebar panel with two responsibilities:
      * File uploader + "Ingest Documents" button — processes uploads into ChromaDB.
      * "Clear Knowledge Base" button — wipes session state and reruns the app, effectively resetting everything to a clean slate.
    """
    with st.sidebar:
        st.header("Documents Ingestion")
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["txt", "pdf", "csv", "xlsx"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("Ingest Documents"):
            total = 0
            collection = get_collection()
            for file in uploaded_files:
                with st.spinner(f"Processing {file.name}..."):
                    num = ingest_document(file, collection)
                    total += num
                    st.success(f"{file.name} — {num} chunks added")
            st.info(f"Total chunks in store: {collection.count()}")

        st.markdown("---")

        # Live chunk counter — shows how much content is currently indexed.
        try:
            chunk_count = get_collection().count()
        except Exception:
            chunk_count = 0      # Collection not yet created
        st.metric("Chunks in vector store", chunk_count)

        if st.button("Clear Knowledge Base"):
            # Remove both the client and chat history from session state, then rerun — init_session_state() will recreate them fresh.
            st.session_state.pop("chroma_client", None)
            st.session_state.pop("chat_history", None)
            st.rerun()


def render_message(role: str, content: str):
    # Display a single chat message in a coloured box.
    if role == "user":
        col_gap, col_msg = st.columns([1, 5])
        with col_msg:
            st.info(f"**USER:**\n\n{content}")
    else:
        col_gap, col_msg = st.columns([1, 5])
        with col_msg:
            st.success(f"**BOT:**\n\n{content}")


def render_chat():
    """
    Main chat panel:
      1. Replays the full conversation history on every rerun so the UI stays consistent after Streamlit's script re-execution model.
      2. Presents a form (text input + submit button) for the next question.
      3. On submission, runs the agent and appends both the user message and the assistant reply to session state.
      4. Shows an expandable panel of the raw retrieved chunks so the user can inspect exactly what context the answer was based on.
    """
    st.subheader("Ask a Question")

    # Re-render all previous messages (Streamlit reruns wipe the screen).
    for msg in st.session_state["chat_history"]:
        render_message(msg["role"], msg["content"])

    # clear_on_submit=True resets the text box after the user clicks Submit.
    with st.form(key="chat_form", clear_on_submit=True):
        query = st.text_input("Your question:", placeholder="Ask anything about your documents...")
        submitted = st.form_submit_button("Submit")

    # Do nothing if the form wasn't submitted or the query is blank.
    if not submitted or not query.strip():
        return

    # Persist and render the user's message immediately.
    st.session_state["chat_history"].append({"role": "user", "content": query})
    render_message("user", query)

    # Run the full RAG pipeline (retrieve -> reason -> validate).
    with st.spinner("Agent reasoning..."):
        result = run_agent(query, get_collection())

    # Surface the hallucination warning above the answer if grounding is low.
    if result["warning"]:
        st.warning(result["warning"])

    # Persist and render the assistant's answer.
    st.session_state["chat_history"].append({"role": "assistant", "content": result["answer"]})
    render_message("assistant", result["answer"])

    # Collapsible panel showing the raw chunks that were fed to the LLM. Truncated to 256 chars to keep the UI readable.
    if result["chunks"]:
        with st.expander("Retrieved context chunks"):
            for i, chunk in enumerate(result["chunks"], 1):
                st.markdown(
                    f"**Chunk {i}** | Source: `{chunk['source']}` | Relevance: `{chunk['score']:.2f}`"
                )
                st.text(chunk["text"][:256] + ("..." if len(chunk["text"]) > 256 else ""))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Capstone Project - Documents Q&A",
        layout="wide",
    )

    # Initialise session state before any widgets are rendered.
    init_session_state()

    st.title("Documents Q&A")
    st.caption("Upload a document, then ask questions. Powered by Gemini LLM + ChromaDB vector store.")

    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()

