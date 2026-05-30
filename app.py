import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI

# ======================
# UI
# ======================
st.set_page_config(page_title="AI Pembangkit", layout="wide")
st.title("⚡ AI Pembangkit Assistant")

# ======================
# LOAD PDF
# ======================
def load_pdfs():
    docs = []

    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            path = os.path.join("data", file)
            loader = PyPDFLoader(path)
            pages = loader.load()

            for p in pages:
                p.metadata["source"] = file

            docs.extend(pages)

    return docs


# ======================
# BUILD VECTOR DB (FAISS)
# ======================
@st.cache_resource
def load_db():
    documents = load_pdfs()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    texts = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(texts, embeddings)
    return db


# ======================
# LLM (OpenRouter)
# ======================
def get_llm():
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=st.secrets["API_KEY"],
        model="meta-llama/llama-3.1-8b-instruct:free"
    )


# ======================
# CHAT UI
# ======================
query = st.text_input("Tanya manual pembangkit:")

if query:

    with st.spinner("Mencari jawaban di manual..."):
        db = load_db()
        retriever = db.as_retriever(search_kwargs={"k": 3})

        docs = retriever.invoke(query)

        context = "\n\n".join([d.page_content for d in docs])

        llm = get_llm()

        prompt = f"""
Kamu adalah engineer pembangkit listrik.

Gunakan konteks berikut untuk menjawab:

{context}

Pertanyaan:
{query}

Jawab secara teknis, jelas, dan praktis untuk operator.
"""

        response = llm.invoke(prompt)

        st.subheader("📌 Jawaban")
        st.write(response.content)

        st.subheader("📚 Sumber Manual")
        for d in docs:
            st.write("-", d.metadata.get("source", "unknown"))
