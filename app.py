import hashlib
import json
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

APP_DIR = Path(__file__).parent
DB_DIR = APP_DIR / "chroma_db"
BOOKS_FILE = APP_DIR / "books.json"
COLLECTION_NAME = "bookmind_books"

# Hybrid retrieval settings
DENSE_K = 8
SPARSE_K = 8
FINAL_K = 4
RRF_K = 60
DENSE_WEIGHT = 0.5
SPARSE_WEIGHT = 0.5

st.set_page_config(
    page_title="BookMind | AI Book Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1180px;}
    .hero {padding: 1.5rem 1.7rem; border-radius: 18px;
           background: linear-gradient(120deg, rgba(91,76,230,.16), rgba(20,184,166,.10));
           border: 1px solid rgba(128,128,128,.22); margin-bottom: 1.2rem;}
    .hero h1 {margin: 0 0 .35rem 0; font-size: 2.1rem;}
    .muted {color: var(--text-color); opacity: .72;}
    [data-testid="stChatMessage"] {border-radius: 14px;}
    div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.2);
       padding: 12px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)


def read_book_registry():
    if not BOOKS_FILE.exists():
        return {}
    try:
        return json.loads(BOOKS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def write_book_registry(books):
    BOOKS_FILE.write_text(
        json.dumps(books, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def get_vectorstore():
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_DIR),
        embedding_function=get_embeddings(),
    )


@st.cache_resource
def get_bm25_retriever(book_id):
    """
    Build an in-memory BM25 index for only the selected book.
    The cache is cleared whenever a book is added or removed.
    """
    data = get_vectorstore().get(
        where={"book_id": book_id},
        include=["documents", "metadatas"],
    )

    contents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    documents = [
        Document(page_content=content, metadata=metadata or {})
        for content, metadata in zip(contents, metadatas)
        if content and content.strip()
    ]

    if not documents:
        return None

    retriever = BM25Retriever.from_documents(documents)
    retriever.k = SPARSE_K
    return retriever


def get_groq_api_key():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except (FileNotFoundError, KeyError):
            api_key = None

    return api_key


def get_llm():
    api_key = get_groq_api_key()

    if not api_key:
        return None

    return init_chat_model(
        "openai/gpt-oss-120b",
        model_provider="groq",
        temperature=0,
        api_key=api_key
    )


def remove_book(book_id):
    """Delete a book's vectors and its registry entry."""
    books = read_book_registry()
    if book_id not in books:
        return False

    # Remove only chunks associated with this book.
    get_vectorstore().delete(where={"book_id": book_id})

    # Ensure a future BM25 index is built from the updated collection.
    get_bm25_retriever.clear()

    books.pop(book_id, None)
    write_book_registry(books)
    return True


def index_uploaded_book(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    book_id = hashlib.sha256(file_bytes).hexdigest()[:20]
    books = read_book_registry()

    if book_id in books:
        return book_id, books[book_id], False

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        documents = PyPDFLoader(temp_path).load()
    finally:
        Path(temp_path).unlink(missing_ok=True)

    if not documents or not any(doc.page_content.strip() for doc in documents):
        raise ValueError(
            "No readable text was found. This may be a scanned PDF."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["book_id"] = book_id
        chunk.metadata["book_name"] = uploaded_file.name
        # PyPDFLoader page numbers are zero-based.
        # They are converted to one-based page labels when displaying sources.

    get_vectorstore().add_documents(chunks)

    # Clear cached BM25 indexes so they reflect the current Chroma collection.
    get_bm25_retriever.clear()

    info = {
        "name": uploaded_file.name,
        "book_id": book_id,
        "pages": len(documents),
        "chunks": len(chunks),
    }
    books[book_id] = info
    write_book_registry(books)

    return book_id, info, True


def reciprocal_rank_fusion(result_lists, weights=None, rrf_k=RRF_K):
    """
    Merge ranked document lists using Reciprocal Rank Fusion.

    RRF score for a document:
        sum(weight / (rrf_k + rank))

    Ranks start at 1. Duplicate chunks from both retrievers are counted once
    as a document, but receive a score contribution from each retriever.
    """
    if weights is None:
        weights = [1.0] * len(result_lists)

    if len(weights) != len(result_lists):
        raise ValueError("There must be one weight for each result list.")

    scores = {}
    documents_map = {}

    for docs, weight in zip(result_lists, weights):
        for rank, doc in enumerate(docs, start=1):
            # Build a stable key for the same chunk returned by both retrievers.
            content_hash = hashlib.sha256(
                doc.page_content.encode("utf-8")
            ).hexdigest()

            doc_id = (
                f"{doc.metadata.get('book_id', '')}:"
                f"{doc.metadata.get('page', '')}:"
                f"{doc.metadata.get('start_index', '')}:"
                f"{content_hash}"
            )

            documents_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + (
                weight / (rrf_k + rank)
            )

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    return [documents_map[doc_id] for doc_id in ranked_ids]


def answer_question(question, book_id):
    llm = get_llm()
    if llm is None:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your .env file or Streamlit Cloud Secrets."
        )

    # 1. Dense retrieval: semantic similarity search in ChromaDB.
    dense_retriever = get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": DENSE_K,
            "filter": {"book_id": book_id},
        },
    )
    dense_docs = dense_retriever.invoke(question)

    # 2. Sparse retrieval: BM25 keyword search over the selected book's chunks.
    sparse_retriever = get_bm25_retriever(book_id)
    sparse_docs = (
        sparse_retriever.invoke(question)
        if sparse_retriever is not None
        else []
    )

    # 3. Hybrid retrieval: combine both ranked lists using RRF.
    retrieved_docs = reciprocal_rank_fusion(
        [dense_docs, sparse_docs],
        weights=[DENSE_WEIGHT, SPARSE_WEIGHT],
        rrf_k=RRF_K,
    )

    # Keep only the best-ranked unique chunks for the LLM context.
    retrieved_docs = retrieved_docs[:FINAL_K]

    if not retrieved_docs:
        return "I couldn't find relevant passages in this book.", []

    # 4. Prepare context and source information.
    context_parts = []
    sources = []

    for doc in retrieved_docs:
        page = doc.metadata.get("page")
        page_label = (
            f"Page {int(page) + 1}"
            if page is not None
            else "Page unknown"
        )

        context_parts.append(
            f"[{page_label}]\n{doc.page_content}"
        )
        sources.append({
            "page": page_label,
            "text": doc.page_content,
        })

    # 5. Generate an answer using the retrieved context.
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are BookMind, a helpful assistant that answers questions about a
            user's uploaded book. Use only the supplied context. If the context does
            not contain the answer, clearly say you don't know based on this book.
            Do not invent facts. Refer to page numbers when useful.""",
        ),
        (
            "human",
            "Book context:\n{context}\n\nQuestion: {question}",
        ),
    ])

    response = llm.invoke(
        prompt.invoke({
            "context": "\n\n---\n\n".join(context_parts),
            "question": question,
        })
    )

    return response.content, sources


# Header
st.markdown("""
<div class="hero">
  <h1>📚 BookMind</h1>
  <div class="muted">Your personal AI reading companion — upload a book and ask it anything.</div>
</div>
""", unsafe_allow_html=True)

books = read_book_registry()

with st.sidebar:
    st.title("📖 Your Library")
    st.caption("Upload a PDF to add it to your searchable library.")

    uploaded_file = st.file_uploader(
        "Choose a book (PDF)",
        type=["pdf"],
        help="Text-based PDFs work best. Scanned image-only PDFs may need OCR first.",
    )

    if st.button(
        "Process book",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    ):
        try:
            with st.spinner(
                "Reading pages, creating chunks, and indexing embeddings..."
            ):
                book_id, info, was_added = index_uploaded_book(uploaded_file)

            if was_added:
                st.success(f"Added **{info['name']}** to your library.")
            else:
                st.info("This exact PDF is already in your library.")

            st.session_state["selected_book_id"] = book_id
            st.session_state["messages"] = []
            st.rerun()

        except Exception as exc:
            st.error(f"Could not process this PDF: {exc}")

    st.divider()

    if books:
        book_ids = list(books.keys())
        selected_default = st.session_state.get("selected_book_id")

        if selected_default not in books:
            selected_default = book_ids[0]

        selected_id = st.selectbox(
            "Choose a book to chat with",
            options=book_ids,
            index=book_ids.index(selected_default),
            format_func=lambda book_id: books[book_id]["name"],
        )

        if selected_id != st.session_state.get("selected_book_id"):
            st.session_state["selected_book_id"] = selected_id
            st.session_state["messages"] = []

        active_book = books[selected_id]

        st.caption(f"📄 {active_book.get('pages', '?')} pages")
        st.caption(f"🧩 {active_book.get('chunks', '?')} text chunks")

        st.button(
            "🗑️ Remove selected book",
            key="remove_book_button",
            use_container_width=True,
            help="Remove this book and its indexed chunks from the local library.",
            on_click=lambda: st.session_state.__setitem__(
                "confirm_delete_book_id", selected_id
            ),
        )

        if st.session_state.get("confirm_delete_book_id") == selected_id:
            st.warning(
                f"Remove **{active_book['name']}**? Its indexed vectors and library "
                "entry will be deleted. This cannot be undone."
            )

            confirm_col, cancel_col = st.columns(2)

            with confirm_col:
                if st.button(
                    "Yes, remove",
                    type="primary",
                    key="confirm_remove_book",
                    use_container_width=True,
                ):
                    try:
                        remove_book(selected_id)
                        st.session_state.pop("confirm_delete_book_id", None)

                        if st.session_state.get("selected_book_id") == selected_id:
                            st.session_state.pop("selected_book_id", None)

                        st.session_state["messages"] = []
                        st.session_state.pop("pending_question", None)
                        st.rerun()

                    except Exception as exc:
                        st.error(f"Could not remove this book: {exc}")

            with cancel_col:
                if st.button(
                    "Cancel",
                    key="cancel_remove_book",
                    use_container_width=True,
                ):
                    st.session_state.pop("confirm_delete_book_id", None)
                    st.rerun()

    else:
        selected_id = None
        st.info("Your library is empty. Upload a PDF book to get started.")

    st.divider()
    st.caption("**How Hybrid RAG works**")
    st.markdown(
        "1. Upload a PDF\n"
        "2. Split text into chunks\n"
        "3. Create embeddings and store in ChromaDB\n"
        "4. Dense retrieval using semantic similarity\n"
        "5. Sparse retrieval using BM25\n"
        "6. Combine rankings using RRF\n"
        "7. Generate an answer with Groq"
    )


# Main panel
if not get_groq_api_key():
    st.warning(
        "Add GROQ_API_KEY to your .env file or Streamlit Cloud Secrets to enable chat."
    )

if selected_id:
    active_book = books[selected_id]

    st.subheader(f"Chat with: {active_book['name']}")
    st.caption(
        "Hybrid retrieval combines semantic search and BM25 keyword search. "
        "Open Sources under an answer to inspect retrieved passages."
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if not st.session_state["messages"]:
        st.markdown("#### Try asking")

        suggestions = [
            "Summarize the main ideas in this book",
            "Explain an important concept in simple words",
            "What are the key takeaways?",
        ]

        cols = st.columns(len(suggestions))
        for col, suggestion in zip(cols, suggestions):
            if col.button(suggestion, use_container_width=True):
                st.session_state["pending_question"] = suggestion
                st.rerun()

    # Display previous conversation messages.
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("Sources from the book"):
                    for i, source in enumerate(message["sources"], start=1):
                        st.markdown(f"**{i}. {source['page']}**")
                        st.write(source["text"])

    pending_question = st.session_state.pop("pending_question", None)
    question = st.chat_input("Ask a question about this book...")

    if pending_question:
        question = pending_question

    if question:
        st.session_state["messages"].append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner(
                "Running dense search + BM25, combining results, and preparing an answer..."
            ):
                try:
                    answer, sources = answer_question(question, selected_id)
                    st.markdown(answer)

                    if sources:
                        with st.expander("Sources from the book"):
                            for i, source in enumerate(sources, start=1):
                                st.markdown(f"**{i}. {source['page']}**")
                                st.write(source["text"])

                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception as exc:
                    st.error(str(exc))

else:
    st.markdown("### Get started")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📤 Upload")
        st.write("Add a PDF book using the panel on the left.")

    with col2:
        st.markdown("#### 🧠 Index")
        st.write(
            "The app extracts text, creates overlapping chunks, and stores "
            "embeddings in ChromaDB. BM25 builds a keyword index for each book."
        )

    with col3:
        st.markdown("#### 💬 Ask")
        st.write(
            "Ask questions and inspect the retrieved source passages behind "
            "each answer. Dense and sparse rankings are combined using RRF."
        )
