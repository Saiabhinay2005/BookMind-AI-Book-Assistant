
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# 1. Initialize Embedding Model
embedding_model = HuggingFaceEmbeddings()

# 2. Load Existing ChromaDB
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# 3. Dense Retriever (Semantic Search)
dense_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 4. Get documents for Sparse Retriever
# BM25 needs the original documents in memory.
all_docs = vectorstore.get()

from langchain_core.documents import Document

documents = [
    Document(
        page_content=content,
        metadata=metadata
    )
    for content, metadata in zip(
        all_docs["documents"],
        all_docs["metadatas"]
    )
]

# 5. Sparse Retriever (Keyword Search)
sparse_retriever = BM25Retriever.from_documents(
    documents
)

sparse_retriever.k = 4

# 6. Hybrid Retriever (RRF)
ensemble_retriever = EnsembleRetriever(
    retrievers=[
        dense_retriever,
        sparse_retriever
    ],
    weights=[0.5, 0.5]
)

# 7. Initialize LLM
llm = init_chat_model(
    "openai/gpt-oss-120b",
    model_provider="groq"
)

# 8. Prompt Template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.
        Use only the context provided to answer the question.
        If you don't know the answer, say you don't know.
        Do not make up information."""
    ),
    (
        "human",
        """Context:
        {context}

        Question:
        {question}"""
    )
])

# 9. RAG Pipeline
print("Hybrid RAG Project is running...")
print("Enter 0 to exit the program.")

while True:

    query = input("Enter your question: ")

    if query == "0":
        break

    # Retrieve documents using Hybrid Search
    docs = ensemble_retriever.invoke(query)

    # Combine retrieved documents
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Generate prompt
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    # Generate answer
    response = llm.invoke(final_prompt)

    print("Answer:", response.content)
