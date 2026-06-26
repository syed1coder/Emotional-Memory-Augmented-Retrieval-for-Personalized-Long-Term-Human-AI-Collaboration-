import streamlit as st
import json
from transformers import pipeline
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Emotional RAG Chatbot", page_icon="🤖")

st.title("🤖 Emotional Memory RAG Chatbot")
st.write("Talk to your AI that understands emotions + remembers your feelings")

# =========================
# LOAD MODELS (CACHE STYLE)
# =========================
@st.cache_resource
def load_models():
    loader = TextLoader("documents/sample.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    emotion_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1
    )

    llm = OllamaLLM(model="llama3.2")

    return vectorstore, emotion_model, llm

vectorstore, emotion_model, llm = load_models()

# =========================
# MEMORY LOAD
# =========================
def load_memory():
    try:
        with open("emotional_memory.json", "r") as f:
            return json.load(f)
    except:
        return []

def save_memory(memory):
    with open("emotional_memory.json", "w") as f:
        json.dump(memory, f, indent=4)

memory = load_memory()

# =========================
# CHAT UI MEMORY
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================
# INPUT
# =========================
user_input = st.text_input("Type your message here:")

if user_input:

    # ================= EMOTION =================
    result = emotion_model(user_input)
    raw_emotion = result[0][0]["label"]

    emotion_map = {
        "anger": "stress",
        "fear": "anxiety",
        "sadness": "sadness",
        "joy": "happy",
        "neutral": "neutral"
    }

    emotion = emotion_map.get(raw_emotion, raw_emotion)

    # ================= MEMORY =================
    related = [m["query"] for m in memory if m["emotion"] == emotion]

    # ================= RAG =================
    docs = vectorstore.similarity_search(user_input, k=2)
    context = "\n".join([d.page_content for d in docs])

    # ================= PROMPT =================
    prompt = f"""
You are an emotional AI assistant.

Emotion: {emotion}

Past memories:
{related}

Context:
{context}

User:
{user_input}

Respond in a friendly, empathetic way.
"""

    response = llm.invoke(prompt)

    # ================= SAVE MEMORY =================
    memory.append({
        "query": user_input,
        "emotion": emotion,
        "response": response
    })
    save_memory(memory)

    # ================= CHAT HISTORY =================
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("AI", response))

# =========================
# DISPLAY CHAT
# =========================
for role, msg in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 AI:** {msg}")