import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings

try:
    from langchain_community.llms import Ollama
except ImportError:
    from langchain.llms import Ollama

# The minimal LangChain install used here does not ship the legacy RetrievalQA chain,
# so we build a lightweight retrieval-augmented prompt manually.
from langchain_core.output_parsers import StrOutputParser


def format_docs_for_prompt(docs):
    """Return a single context string the LLM can ground on."""
    return "\n\n".join(doc.page_content for doc in docs)

# --- Title and Intro ---
st.set_page_config(page_title="KrishnavyasGPT", page_icon="KV")
st.title("KrishnavyasGPT - Ask Me About My Work")
st.markdown("""
Hi! I'm **Krishnavyas Desugari**, a Data Scientist passionate about Machine Learning, Automation, and Analytics.  
You can ask me about my **projects, skills, tools, or experience**!
""")

# --- Load Data ---
@st.cache_resource
def load_vector_db():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    pdf_paths = [
        "data/Krishnavyas_Desugari_Resume_Data_science.pdf",
        "data/Full Profile.pdf"
    ]

    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pdf_docs = loader.load()
        docs.extend(text_splitter.split_documents(pdf_docs))

    vectorstore = Chroma.from_documents(
        docs,
        embedding=embedding_model,
        persist_directory="chroma_db"
    )
    return vectorstore


vectorstore = load_vector_db()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- Ollama LLM Setup ---
llm = Ollama(model="llama3", temperature=0)
output_parser = StrOutputParser()

# Prompt template used to keep the model grounded in retrieved content.
QA_PROMPT = """You are a helpful assistant that answers questions about Krishnavyas Desugari.
Only use the factual information found in the context to answer. If the context
does not contain the answer, say you do not have that information.

Context:
{context}

Question:
{question}

Helpful answer:"""

# --- Chat Interface ---
query = st.text_input("Ask a question about my profile or projects:")

if query:
    with st.spinner("Thinking..."):
        source_docs = retriever.invoke(query)
        context = format_docs_for_prompt(source_docs)
        prompt = QA_PROMPT.format(context=context, question=query)
        answer = output_parser.invoke(llm.invoke(prompt))

        st.write("### Answer:")
        st.write(answer)

        with st.expander("Sources used"):
            for doc in source_docs:
                st.write(doc.metadata.get("source", "Unknown file"))
