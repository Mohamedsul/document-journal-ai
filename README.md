# 📓 AI Document Journal – Semantic PDF Search with Streamlit

An AI-powered document search assistant that allows you to upload PDF files and ask natural language questions to extract insights from your documents.

## 🚀 Features

- Upload and organize any kind of PDF (reports, manuals, meeting notes, etc.)
- Search using natural language queries
- Answers backed by source document snippets and metadata
- Persistent vector storage with ChromaDB
- Local embeddings using HuggingFace (no API key needed)
- Streamlit web interface



## 🧰 Tech Stack

- Python
- Streamlit
- LlamaIndex
- ChromaDB
- Hugging Face Embeddings

## 📂 Folder Structure

📦 ai-document-journal/
├── 📁 documents/                  # Folder for uploading PDF files
├── 📁 storage/                    # Persistent storage (ChromaDB)
│   └── chroma_db/
├── 📄 document_journal.py        # Project main indexer class
├── 📄 app.py                     # Project Streamlit UI
├── 📄 requirements.txt           # Required Python packages
├── 📄 README.md                  # Project description for GitHub

