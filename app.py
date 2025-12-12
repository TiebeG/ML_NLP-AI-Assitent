import os
import sys
import uuid
import json
import sqlite3
import datetime
import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage

# ----------------------------------------------------
# Ensure backend import works
# ----------------------------------------------------
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from backend.graph_ml_assistant import graph_app


# ----------------------------------------------------
# DETECT ENVIRONMENT & SET DB PATH
# ----------------------------------------------------
if os.path.exists("/mount/data"):
    # Streamlit Cloud
    DATA_DIR = "/mount/data"
else:
    # Local development
    DATA_DIR = os.path.join(os.getcwd(), "local_data")

os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "chats.db")


# ----------------------------------------------------
# SQLITE SETUP
# ----------------------------------------------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS chats (
    chat_id TEXT PRIMARY KEY,
    name TEXT,
    messages TEXT,
    last_updated REAL
)
""")
conn.commit()


# ----------------------------------------------------
# DB HELPERS
# ----------------------------------------------------
def load_chats():
    cursor.execute("SELECT chat_id, name, messages, last_updated FROM chats")
    rows = cursor.fetchall()
    chats = {}
    for chat_id, name, messages, last_updated in rows:
        chats[chat_id] = {
            "name": name,
            "messages": json.loads(messages),
            "last_updated": last_updated
        }
    return chats


def save_chat(chat_id, chat):
    cursor.execute(
        """
        INSERT OR REPLACE INTO chats (chat_id, name, messages, last_updated)
        VALUES (?, ?, ?, ?)
        """,
        (
            chat_id,
            chat["name"],
            json.dumps(chat["messages"]),
            chat["last_updated"]
        )
    )
    conn.commit()


def delete_chat(chat_id):
    cursor.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
    conn.commit()


# ----------------------------------------------------
# SESSION INIT
# ----------------------------------------------------
if "chats" not in st.session_state:
    st.session_state.chats = load_chats()

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

if "rename_id" not in st.session_state:
    st.session_state.rename_id = None


# ----------------------------------------------------
# HELPERS
# ----------------------------------------------------
def auto_title_from_text(text):
    return " ".join(text.split()[:6]).title() or "New Chat"


def create_new_chat():
    chat_id = str(uuid.uuid4())
    chat = {
        "name": "New Chat",
        "messages": [],
        "last_updated": datetime.datetime.now().timestamp()
    }
    st.session_state.chats[chat_id] = chat
    st.session_state.current_chat = chat_id
    save_chat(chat_id, chat)


# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
with st.sidebar:
    st.markdown("## 💬 ML Assistant")

    if st.button("➕ New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()

    st.markdown("---")

    sorted_chats = sorted(
        st.session_state.chats.items(),
        key=lambda x: x[1]["last_updated"],
        reverse=True
    )

    for chat_id, chat in sorted_chats:
        col1, col2, col3 = st.columns([8, 1, 1])

        with col1:
            if st.button(chat["name"], key=f"open_{chat_id}", use_container_width=True):
                st.session_state.current_chat = chat_id
                st.rerun()

        with col2:
            if st.button("i", key=f"rename_{chat_id}"):
                st.session_state.rename_id = chat_id

        with col3:
            if st.button("x", key=f"delete_{chat_id}"):
                delete_chat(chat_id)
                del st.session_state.chats[chat_id]
                if st.session_state.current_chat == chat_id:
                    st.session_state.current_chat = None
                st.rerun()

    if st.session_state.rename_id:
        cid = st.session_state.rename_id
        new_name = st.text_input("Rename chat", st.session_state.chats[cid]["name"])
        if st.button("Save"):
            st.session_state.chats[cid]["name"] = new_name
            save_chat(cid, st.session_state.chats[cid])
            st.session_state.rename_id = None
            st.rerun()


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
if st.session_state.current_chat is None:
    st.markdown("## 👋 Welcome")
    st.markdown("Create a new chat from the sidebar.")
    st.stop()

chat_id = st.session_state.current_chat
chat = st.session_state.chats[chat_id]

st.markdown(f"## {chat['name']}")

for msg in chat["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])


user_input = st.chat_input("Ask anything from the ML course...")

if user_input:
    if chat["name"] == "New Chat" and not chat["messages"]:
        chat["name"] = auto_title_from_text(user_input)

    chat["messages"].append({"role": "user", "content": user_input})

    state = {
        "messages": [
            HumanMessage(content=m["content"]) if m["role"] == "user"
            else AIMessage(content=m["content"])
            for m in chat["messages"]
        ]
    }

    result = graph_app.invoke(state)
    reply = result["messages"][-1].content

    chat["messages"].append({"role": "assistant", "content": reply})
    chat["last_updated"] = datetime.datetime.now().timestamp()

    save_chat(chat_id, chat)
    st.rerun()
