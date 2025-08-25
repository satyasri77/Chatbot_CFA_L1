
import os
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pickle

# ---------- CONFIG ----------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # set this in Streamlit Cloud
genai.configure(api_key=GOOGLE_API_KEY)

# ---------- APP ----------
st.set_page_config(page_title="CFA Tutor Bot", layout="wide")
st.title("📚 CFA Level 1 Tutor Bot")
st.markdown("This bot answers questions **only from CFA curriculum resources**.")

# ---------- Load or Build Vector DB ----------
VECTORSTORE_PATH = "faiss_index"

if not os.path.exists(VECTORSTORE_PATH):
    st.info("Building knowledge base from CFA PDFs...")
    documents = []
    pdf_folder = "data"  # your folder with CFA PDFs
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = file
            documents.extend(docs)

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(documents)

    # Embeddings with Google Gemini
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # Save for reuse
    vectorstore.save_local(VECTORSTORE_PATH)

else:
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------- Chat History ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- Chat Input ----------
user_input = st.chat_input("Ask a CFA-related question...")
if user_input:
    docs = retriever.get_relevant_documents(user_input)
    context = "\n\n".join([d.page_content for d in docs])

    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    You are a CFA Level 1 tutor.
    Only use the following study material excerpts to answer.
    If the answer is not in the material, reply: "I don’t know based on CFA resources."

    Context:
    {context}

    Question:
    {user_input}
    """
    response = model.generate_content(prompt)
    answer = response.text

    st.session_state.chat_history.append((user_input, answer, [d.metadata["source"] for d in docs]))

# ---------- Display ----------
for q, a, sources in st.session_state.chat_history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
    st.markdown(f"_Sources: {', '.join(set(sources))}_")
