# 📚 BookMind – AI Book Assistant

BookMind is an AI-powered PDF question-answering application built using **Retrieval-Augmented Generation (RAG)**. It allows users to upload PDF books and ask questions based on their contents.

Instead of relying only on the LLM's general knowledge, BookMind retrieves relevant information from uploaded documents and generates context-aware answers.

## 🚀 Features

- Upload and process PDF books.
- Extract text from PDF documents.
- Split documents into smaller chunks.
- Generate text embeddings using HuggingFace.
- Store embeddings in ChromaDB.
- Dense retrieval using semantic similarity search.
- Sparse retrieval using BM25 keyword search.
- Hybrid retrieval using Reciprocal Rank Fusion (RRF).
- Generate answers using Groq LLM.
- Display source passages and page numbers.
- Manage uploaded books through the application.

## 🛠️ Technologies Used

- Python
- Streamlit
- LangChain
- HuggingFace Embeddings
- ChromaDB
- BM25 Retriever
- Groq LLM
- PyPDF

## 🧠 How It Works

1. Upload a PDF book.
2. Extract text from the PDF.
3. Split the text into overlapping chunks.
4. Generate embeddings for each chunk.
5. Store embeddings and metadata in ChromaDB.
6. Retrieve relevant chunks using dense and sparse retrieval.
7. Combine retrieval results using Reciprocal Rank Fusion.
8. Pass the retrieved context to the Groq LLM.
9. Generate an answer with relevant source references.

## 📂 Project Structure

```text
BookMind/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/BookMind-AI-Book-Assistant.git
cd BookMind-AI-Book-Assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the Groq API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

### 6. Run the application

```bash
streamlit run app.py
```

## 🔑 API Configuration

BookMind uses the Groq API to generate answers. Obtain an API key from:

https://console.groq.com/

Keep your API key private. Never commit your `.env` file to GitHub.

## ☁️ Deployment

The application can be deployed using Streamlit Community Cloud:

https://share.streamlit.io/

Configure `GROQ_API_KEY` in the application's Streamlit Secrets settings.

## 👨‍💻 Project

**BookMind – AI Book Assistant**

