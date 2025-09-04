import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,pipeline

# ---------- CONFIG ----------
VECTORSTORE_PATH = "cfa_index"  # Prebuilt FAISS folder
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_LLM = "google/flan-t5-base"  # You can switch to a lighter one like "google/flan-t5-base"

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

# ---------- Load Local Language Model ----------
@st.cache_resource
def load_local_llm():
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM)
    model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_LLM)
    generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return generator

llm_pipeline = load_local_llm()

# ---------- Chat History ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- Chat Input ----------
user_input = st.chat_input("Ask a CFA-related question...")

if user_input:
    with st.spinner("Thinking..."):
        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(user_input)
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = [doc.metadata.get("source", "unknown") for doc in docs]

        # Prompt construction (no pre-existing 'answer' mistake)
        prompt = f"""
You are a CFA Level 1 tutor.
Use ONLY the provided context to answer the question clearly and accurately.
Explain your reasoning for incorrect answers and correct answer for the provided 3 options for CFA level 1 exam, and cite relevant concepts if available.
If the answer is not found in the context, say: "The answer is not available in the provided context."

Question: {user_input}

Context:
{context}

Answer:
"""
                # Generate answer using the local LLM
        try:
            output = llm_pipeline(prompt)[0]["generated_text"]
            # Clean up prompt echo from model output (optional)
            if "Answer:" in output:
                answer = output.split("Answer:")[-1].strip()
            else:
                answer = output.strip()
        except Exception as e:
            answer = f"⚠️ Error generating response: {str(e)}"
    
        # Store in session state
        st.session_state.chat_history.append((user_input, answer, sources))




# ---------- Display Chat ----------
for q, a, sources in st.session_state.chat_history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
    # st.markdown(f"_Sources: {', '.join(set(sources))}_")
