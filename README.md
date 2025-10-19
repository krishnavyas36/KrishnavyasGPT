# KrishnavyasGPT

Interactive Streamlit application that answers questions about Krishnavyas Desugari's work, skills, and experience by running retrieval-augmented generation (RAG) over curated PDF documents.

## Features
- 📄 Ingests multiple PDFs (resume and extended profile) and indexes them with Chroma.
- 🔎 Uses semantic search to retrieve the most relevant document chunks for each question.
- 🤖 Generates grounded answers with the local Ollama `llama3` model.
- 🧠 Handles embedding generation with `sentence-transformers/all-MiniLM-L6-v2` via Hugging Face embeddings.
- 🖥️ Provides a simple Streamlit UI so visitors can ask questions about projects, skills, and tools.

## Requirements

- Python 3.11+ (the project currently runs on 3.12)
- [Ollama](https://ollama.ai/) installed locally with the `llama3` model pulled

Python dependencies are listed below:

```bash
streamlit
langchain
langchain-community
langchain-text-splitters    # fallback for RecursiveCharacterTextSplitter
chromadb
tiktoken
pypdf
sentence-transformers
```

You can install them with:

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file does not yet exist, create one with the packages above.

## Project Structure

```
.
├─ app.py                # Streamlit entry point with the RAG pipeline
├─ chroma_db/            # Vector store persisted by Chroma (created at runtime)
├─ data/
│  ├─ Krishnavyas_Desugari_Resume_Data_science.pdf
│  └─ Full Profile.pdf
└─ README.md
```

## Running the App

```bash
streamlit run app.py
```

Streamlit will open a browser window (or provide a local URL). Ask questions in the text box, and the app will display an answer along with the sources used.

## Updating the Knowledge Base

1. Add or replace PDFs in the `data/` directory.
2. Delete the `chroma_db/` folder to force a clean re-ingestion the next time the app starts.

## Troubleshooting

- **Import errors** – Make sure `langchain-community` is installed: `pip install langchain-community`.
- **Embedding model download issues** – Ensure you have internet access the first time the Hugging Face model is loaded.
- **Ollama not found** – Install Ollama and run `ollama pull llama3` before starting the app.

## License

This project is provided for personal portfolio use. Adapt as needed for your own deployments.
