import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

st.title("AI Pembangkit 🔧")

def load_pdfs():
    docs = []
    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join("data", file))
            docs.extend(loader.load())
    return docs

@st.cache_resource
def load_db():
    documents = load_pdfs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300)
    texts = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma.from_documents(texts, embeddings)

def load_llm():
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=st.secrets["API_KEY"],
        model="mistralai/mistral-7b-instruct"
    )

query = st.text_input("Tanya:")

if query:
    db = load_db()
    docs = db.as_retriever().invoke(query)[:2]

    context = "\n\n".join([d.page_content for d in docs])

    llm = load_llm()

    response = llm.invoke(f"{context}\n\n{query}")

    st.write(response.content)