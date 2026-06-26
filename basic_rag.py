from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from transformers import pipeline
import json

# =========================
# Load document
# =========================
loader = TextLoader("documents/sample.txt")
documents = loader.load()

# =========================
# Split document
# =========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = text_splitter.split_documents(documents)

# =========================
# Embeddings
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# =========================
# Vector DB
# =========================
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

# =========================
# Emotion Model
# =========================
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

# =========================
# User Input
# =========================
query = input("\nAsk something: ")

# =========================
# Emotion Detection
# =========================
result = emotion_classifier(query)
raw_emotion = result[0][0]["label"]

# Emotion Mapping
emotion_map = {
    "anger": "stress",
    "fear": "anxiety",
    "sadness": "sadness",
    "joy": "happy",
    "surprise": "surprise",
    "neutral": "neutral"
}

emotion = emotion_map.get(raw_emotion, raw_emotion)

print("\nDetected Emotion (raw):", raw_emotion)
print("Mapped Emotion:", emotion)

# =========================
# Load Emotional Memory
# =========================
try:
    with open("emotional_memory.json", "r") as file:
        memory = json.load(file)
except:
    memory = []

# =========================
# CROSS-EMOTION MEMORY LOGIC (NEW)
# =========================
emotion_groups = {
    "stress": ["stress", "anger", "frustration"],
    "anxiety": ["fear", "anxiety", "stress"],
    "sadness": ["sadness", "lonely", "disappointment"],
    "happy": ["happy", "joy"],
    "neutral": ["neutral"]
}

allowed_emotions = emotion_groups.get(emotion, [emotion])

print("\nRelated Emotional Memories:")

related_memories = []

for m in memory:
    if m["emotion"] in allowed_emotions:
        print("-", m["query"])
        related_memories.append(m["query"])

if not related_memories:
    print("No related memories found.")

# =========================
# RAG Retrieval
# =========================
docs = vectorstore.similarity_search(query, k=2)

print("\nRetrieved Context:")
for doc in docs:
    print("-", doc.page_content)

context = "\n".join([doc.page_content for doc in docs])

# =========================
# Prompt Construction
# =========================
memory_text = "\n".join(related_memories)

prompt = f"""
You are an emotionally intelligent assistant.

Use context, emotional memory, and user emotion to respond naturally.

-----------------------
User Input:
{query}

Raw Emotion:
{raw_emotion}

Mapped Emotion:
{emotion}

Emotional Memories:
{memory_text}

Context:
{context}

Respond in a human, empathetic, and helpful way.
"""

# =========================
# LLM
# =========================
llm = OllamaLLM(model="llama3.2")

response = llm.invoke(prompt)

print("\nLLM Response:")
print(response)

# =========================
# Save Memory
# =========================
new_memory = {
    "query": query,
    "emotion": emotion,
    "response": response
}

memory.append(new_memory)

with open("emotional_memory.json", "w") as file:
    json.dump(memory, file, indent=4)

print("\nMemory saved successfully!")