# medical_app.py
import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from rag_engine import get_retriever
import os

st.set_page_config(page_title="Medical RAG Assistant", layout="wide")
st.title("🩺 Medical Report Assistant")

UPLOAD_DIR = "medical_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Upload section
st.sidebar.header("📤 Upload Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload medical PDFs or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

for file in uploaded_files:
    with open(os.path.join(UPLOAD_DIR, file.name), "wb") as f:
        f.write(file.getbuffer())
    st.sidebar.success(f"{file.name} uploaded")

# Prompt + QA
question = st.text_input("💬 Ask a question:", placeholder="e.g., What are the symptoms of fever?")

# ... all imports + previous code

if question:
    st.write("🔍 Answering...")

    model = OllamaLLM(model="llama3.2")

    template = """
    You are a helpful and knowledgeable medical assistant.

    If the following context contains relevant information, use it to answer the user's question.
    Otherwise, answer using your general medical knowledge.

    Context:
    {reviews}

    Question:
    {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    retriever = get_retriever()
    docs = retriever.invoke(question)

    # If retriever finds nothing useful, we pass "No relevant context" to the prompt
    if not docs or all(len(doc.page_content.strip()) == 0 for doc in docs):
        context = "No relevant context available."
    else:
        context = "\n\n".join([doc.page_content for doc in docs])

    response = chain.invoke({"reviews": context, "question": question})

    st.markdown("### 📝 Answer:")
    st.success(response)
