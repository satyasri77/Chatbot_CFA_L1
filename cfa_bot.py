import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,AutoModelForCausalLM, pipeline

# ---------- CONFIG ----------
VECTORSTORE_PATH = "cfa_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_LLM = "mistralai/Mistral-7B-Instruct-v0.1"  # Consider flan-t5-xl or a better model like mistral if you can

# ---------- Load Vectorstore ----------
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------- Load LLM ----------
@st.cache_resource
def load_local_llm():
    HF_TOKEN = st.secrets["HF_TOKEN"]
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM,use_auth_token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_LLM,
        use_auth_token=HF_TOKEN,
        device_map="auto",         # Automatically maps to GPU if available
        torch_dtype="auto"         # Uses appropriate precision
    )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.1,
        top_p=0.95
    )
    return generator

llm_pipeline = load_local_llm()

# ---------- Streamlit Setup ----------
st.set_page_config(page_title="CFA Tutor Bot", layout="wide")
st.title("📚 CFA Level 1 Tutor Bot")
st.markdown("Ask CFA exam-style multiple-choice questions. The bot will explain correct and incorrect options based on the official curriculum.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- User Input ----------
user_input = st.chat_input("Enter your CFA question with 3 options (A, B, C)...")

if user_input:
    with st.spinner("Analyzing..."):
        docs = retriever.get_relevant_documents(user_input)
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = [doc.metadata.get("source", "unknown") for doc in docs]

        # 👇 Better prompt for reasoning + structure
        prompt = f"""
You are a CFA Level 1 tutor bot.

You will receive a multiple-choice question with 3 options (A, B, C). Your task is to:

1. Identify the **correct option** based ONLY on the provided context.
2. Explain **why the correct option is right**.
3. Explain **why the other options are incorrect**.
4. If the answer cannot be determined from the context, reply with: 
   "The answer is not available in the provided context."

Format your answer like this:

Correct Answer: B  
Explanation:  
- Option A: [Why it's wrong]  
- Option B: [Why it's correct]  
- Option C: [Why it's wrong]

---

Question: {user_input}

Context:
{context}

Answer:
"""

        try:
            output = llm_pipeline(prompt)[0]["generated_text"]
            if "Answer:" in output:
                answer = output.split("Answer:")[-1].strip()
            else:
                answer = output.strip()
        except Exception as e:
            answer = f"⚠️ Error generating response: {e}"

        st.session_state.chat_history.append((user_input, answer, sources))

# ---------- Display Chat ----------
for q, a, sources in st.session_state.chat_history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
    st.markdown(f"<sub>Sources: {', '.join(set(sources))}</sub>", unsafe_allow_html=True)
