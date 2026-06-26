# Emotional Memory-Augmented Retrieval for Personalized Long-Term Human-AI Collaboration (Emotional RAG)

## Project Overview

Emotional RAG is an AI-powered conversational system that extends traditional Retrieval-Augmented Generation (RAG) by integrating emotion detection and persistent memory. Unlike conventional chatbots that provide generic responses and forget previous interactions, this system remembers past conversations, understands users' emotional states, and generates personalized, context-aware responses.

The project aims to improve long-term human-AI collaboration by making interactions more intelligent, adaptive, and emotionally aware.

---

## Problem Statement

Traditional AI chatbots and Retrieval-Augmented Generation (RAG) systems retrieve relevant information but lack long-term memory and emotional understanding. They often forget previous interactions and fail to adapt their responses according to the user's emotional state, leading to impersonal and generic conversations. There is a need for an AI system that combines information retrieval, emotion recognition, and persistent memory to deliver personalized and emotionally aware interactions.

---

## Proposed Solution

The proposed Emotional RAG system enhances traditional RAG by combining emotion detection, semantic retrieval, and persistent memory. It analyzes the user's emotions, retrieves relevant information and past interactions, and generates personalized responses using a Large Language Model (LLM). By storing previous conversations and emotional contexts, the system enables long-term, emotionally aware human-AI interactions.

---

## Features

- Emotion detection from user queries
- Retrieval-Augmented Generation (RAG)
- Persistent emotional memory storage
- Semantic document retrieval
- Personalized and context-aware responses
- Interactive Streamlit user interface
- Long-term conversation history management

---

## System Architecture

text
User Query
     ↓
Emotion Detection
     ↓
Document Retrieval (ChromaDB)
     ↓
Emotional Memory Retrieval
     ↓
Context Combination
     ↓
Llama 3.2 Response Generation
     ↓
Memory Storage
     ↓
Personalized Response


---

## Technologies Used

- Python
- Streamlit
- LangChain
- ChromaDB
- Hugging Face Transformers
- Sentence Transformers
- Ollama
- Llama 3.2

---

## AI Components Used

### Large Language Model (LLM)
- Llama 3.2 (via Ollama)
- Responsible for generating natural language responses.

### Emotion Detection
- Hugging Face Model:
  j-hartmann/emotion-english-distilroberta-base
- Identifies emotions such as joy, sadness, anger, fear, and stress.

### Semantic Search
- Sentence Transformer:
  all-MiniLM-L6-v2
- Converts text into vector embeddings for similarity search.

### Vector Database
- ChromaDB
- Stores and retrieves document embeddings efficiently.

### Memory Module
- Maintains conversation history and emotional context for personalization.

---

## Project Structure

text
Emotional_RAG/
│
├── app.py
├── emotion_test.py
├── memory.json
├── documents/
├── chroma_db/
├── requirements.txt
├── model/
└── README.md


---

## Installation

### Clone the Repository

bash
git clone https://github.com/your-username/Emotional-RAG.git
cd Emotional-RAG


### Create Virtual Environment

bash
python -m venv venv


### Activate Environment

Windows:

bash
venv\Scripts\activate


Linux/Mac:

bash
source venv/bin/activate


### Install Dependencies

bash
pip install -r requirements.txt


---

## Running the Project

Start Ollama:

bash
ollama run llama3.2


Launch the Streamlit application:

bash
streamlit run app.py


---

## Sample Workflow

*User Input:*


I am stressed about my thesis submission.


*Detected Emotion:*


Stress / Anxiety


*Retrieved Context:*


Previous thesis discussions and relevant documents.


*Generated Response:*


Provides personalized guidance considering both retrieved information and emotional context.


---

## Target Users

- Students and learners
- Researchers and educators
- Users of personalized AI assistants
- Customer support systems
- Mental wellness applications
- Long-term human-AI collaboration systems

---

## SDG Alignment

- SDG 9: Industry, Innovation and Infrastructure
- SDG 4: Quality Education

---

## Expected Impact

The Emotional RAG system improves personalization, user engagement, and the quality of human-AI interactions by enabling AI systems to remember past conversations and understand users' emotional contexts. It delivers context-aware and emotionally intelligent responses, making AI assistants more effective for education, research support, customer service, and long-term collaborative applications.

---

## Future Scope

- Multilingual emotion detection
- Voice-based interaction support
- Advanced memory management
- Real-time emotion tracking
- Integration with educational and healthcare applications
- Emotion-based recommendation systems

---

## Author

Tejas L  
Syed Shadab
Master of Computer Applications (MCA)

---

## License

This project is developed for academic and research purposes
