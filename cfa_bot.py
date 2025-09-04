import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------- CONFIG ----------
VECTORSTORE_PATH = "cfa_index"  # prebuilt FAISS folder
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_LLM = "mistralai/Mistral-7B-Instruct-v0.1"

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
    model = AutoModelForCausalLM.from_pretrained(LOCAL_LLM, device_map="auto", torch_dtype="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return pipe

llm_pipeline = load_local_llm()

# ---------- Chat History ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- User Input ----------
user_input = st.chat_input("Ask a CFA-related question...")
if user_input:
    docs = retriever.get_relevant_documents(user_input)
    context = "\n\n".join([d.page_content for d in docs])
    sources = [d.metadata.get("source", "unknown") for d in docs]

    # Construct prompt
    prompt = f"""
You are a CFA Level 1 tutor. 
Use ONLY the provided context to answer the question clearly and accurately. 
Explain your reasoning, and cite concepts if available. 
If the answer is not found in the context, say "The answer is not available in the provided context."

Question: {user_input}

Context:
{context}

Answer:
"""

    # Generate answer
    generated = llm_pipeline(prompt)[0]["generated_text"]
    
    # Post-process (remove prompt echo)
    answer = generated.split("Answer:")[-1].strip()

    # Save to session
    st.session_state.chat_history.append((user_input, answer, sources))

# ---------- Display Chat ----------
for q, a, sources in st.session_state.chat_history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
    st.markdown(f"_Sources: {', '.join(set(sources))}_")
