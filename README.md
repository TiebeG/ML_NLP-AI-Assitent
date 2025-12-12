# 🤖 ML_NLP_AI-Assistant

A **multi-agent personal learning assistant** for a university **Machine Learning & NLP course**, built with **LangGraph**, **Groq LLMs**, **RAG**, and **Streamlit**.

The assistant answers course-specific questions, generates quizzes based on discussion topics, and maintains **persistent per-user chat history and long-term memory** — without requiring a full login system.


## 🎯 Project Goals (MVP)

This project was developed to meet the following core objectives:

- Chat-based assistant for a Machine Learning course  
- Multi-agent collaboration using **LangGraph**  
- Integration of at least one MCP tool (**RAG over course documents**)  
- Short-term conversational context  
- Useful educational outputs (explanations, summaries, quizzes)  

---

## ⭐ Extra Features / Extra Credit

Beyond the MVP, the assistant includes several advanced features:

- **Long-term memory** using a persistent vector database (ChromaDB)  
- **Per-user isolation** via a lightweight 4-digit PIN system  
- **Persistent chats** (survive refresh, reload, and deployment restarts)  
- **Quiz agent** focused on official course discussion topics  
- **Adaptive routing** between explanation, RAG, and quiz agents  


## 🧠 System Architecture Overview

The assistant is built as a **multi-agent system** using **LangGraph**.

### Agents

- **Router Agent**  
  Determines whether a user query requires:
  - RAG-based answering  
  - Conceptual explanation  
  - Quiz generation  

- **Teacher / Explanation Agent**  
  Provides structured explanations for ML concepts and synthesizes retrieved information.

- **RAG Agent**  
  Retrieves relevant content from course slides and documents.

- **Online Search Agent (Fallback)**  
  Performs an online search when the RAG agent does not return sufficient information, allowing the assistant to answer questions that go beyond the course material.

- **Quiz Agent**  
  Generates multiple-choice and reflection questions based on official discussion topics.

- **Memory Agents**
  - **Memory Retriever**: recalls relevant long-term memory  
  - **Memory Writer**: stores useful information automatically or on request  


## 🧠 Long-Term Memory

The assistant uses **ChromaDB** as a vector database to store long-term memory:

- Stored persistently (Streamlit Cloud–safe)  
- Automatically recalled when relevant  
- Hybrid strategy:
  - Explicit memory (e.g. *"remember this"*)  
  - Automatic memory (important explanations)  

This allows the assistant to adapt over time and remember user-specific learning context.


## 👤 User System (No Login Required)

Instead of a full authentication system, the app uses a **4-digit PIN code**:

- Users enter a PIN on first load  
- Each PIN represents a separate user space  
- Chats and memory are isolated per PIN  
- No passwords, no accounts, no setup  


## 🗂️ Project Structure
📦 ML_NLP_AI-Assistant/
```
│
├── app.py                       # Streamlit frontend (UI + persistence)
│
├── backend/
│   ├── graph_ml_assistant.py    # LangGraph multi-agent workflow
│   ├── router_agent.py          # Query classification & routing
│   ├── tools_rag.py             # RAG search over course documents
│   ├── quiz_agent.py            # Quiz generation logic
│   ├── memory.py                # Long-term memory (ChromaDB)
│   └── __init__.py
│
├── notebooks/
│   ├── preprocess_pptx.ipynb    # Slide preprocessing
│   ├── build_vector_db.ipynb    # RAG vector database creation
│   ├── test_rag.ipynb           # RAG testing
│   ├── test_router.ipynb        # Router testing
│   └── test_onlinesearch.ipynb  # Online Search
│
└── README.md
```   
