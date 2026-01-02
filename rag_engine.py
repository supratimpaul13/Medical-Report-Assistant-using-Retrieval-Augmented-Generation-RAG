# rag_engine.py
import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

DOC_DIR = "medical_docs"
DB_DIR = "medical_chroma_db"

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

vector_store = Chroma(
    collection_name="medical_documents",
    persist_directory=DB_DIR,
    embedding_function=embeddings
)

def load_documents():
    documents = []
    for file in os.listdir(DOC_DIR):
        path = os.path.join(DOC_DIR, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".txt"):
            loader = TextLoader(path)
        else:
            continue
        docs = loader.load()
        documents.extend(docs)
    return documents

def update_vectorstore():
    existing = set(vector_store.get()["ids"])
    new_docs = []
    new_ids = []
    all_docs = load_documents()
    for idx, doc in enumerate(all_docs):
        doc_id = f"{doc.metadata.get('source', '')}_{idx}"
        if doc_id not in existing:
            new_docs.append(doc)
            new_ids.append(doc_id)
    if new_docs:
        vector_store.add_documents(new_docs, ids=new_ids)

def get_retriever():
    update_vectorstore()
    return vector_store.as_retriever(search_kwargs={"k": 5})
