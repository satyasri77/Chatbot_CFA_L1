import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------- CONFIG ----------
VECTORSTORE_PATH = "cfa_index"  # prebuilt FAISS folder
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------- Streamlit App ----------
st.set_page_config(page_title="CFA Tutor Bot", layout="wide")
st.title("📚 CFA Level 1 Tutor Bot")
st.markdown("This bot answers questions **only from CFA curriculum resources**.")

# ---------- Load Prebuilt FAISS Index ----------
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------- Chat History ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- Chat Input ----------
user_input = st.chat_input("Ask a CFA-related question...")
if user_input:
    # Get relevant context from CFA PDFs
    docs = retriever.get_relevant_documents(user_input)
    context = "\n\n".join([d.page_content for d in docs])

    # Simple GPT-like response with context
    # Offline: we can simulate a response (replace with OpenAI/Google if you want)
    # For demo, just echo question + context
    answer = f"I have found these excerpts relevant to your question:\n\n{context[:1000]}..."  # first 1000 chars

    st.session_state.chat_history.append((user_input, answer, [d.metadata.get("source", "unknown") for d in docs]))

# ---------- Display Chat ----------
for q, a, sources in st.session_state.chat_history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
    st.markdown(f"_Sources: {', '.join(set(sources))}_")
