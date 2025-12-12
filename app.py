import os
import sys
import uuid
import json
import sqlite3
import datetime
import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage

# ----------------------------------------------------
# Backend import
# ----------------------------------------------------
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from backend.graph_ml_assistant import graph_app


# ----------------------------------------------------
# ENV DETECTION (local vs Streamlit Cloud)
# ----------------------------------------------------
if os.path.exists("/mount/data"):
    DATA_DIR = "/mount/data"
else:
    DATA_DIR = os.path.join(os.getcwd(), "local_data")

os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "chats.db")


# ----------------------------------------------------
# DATABASE
# ----------------------------------------------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS chats (
    user_id TEXT,
    chat_id TEXT,
    name TEXT,
    messages TEXT,
    last_updated REAL,
    PRIMARY KEY (user_id, chat_id)
)
""")
conn.commit()


# ----------------------------------------------------
# DB HELPERS (USER-SCOPED)
# ----------------------------------------------------
def load_chats(user_id):
    cursor.execute(
        "SELECT chat_id, name, messages, last_updated FROM chats WHERE user_id = ?",
        (user_id,)
    )
    rows = cursor.fetchall()
    chats = {}
    for chat_id, name, messages, last_updated in rows:
        chats[chat_id] = {
            "name": name,
            "messages": json.loads(messages),
            "last_updated": last_updated
        }
    return chats


def save_chat(user_id, chat_id, chat):
    cursor.execute(
        """
        INSERT OR REPLACE INTO chats
        (user_id, chat_id, name, messages, last_updated)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            user_id,
            chat_id,
            chat["name"],
            json.dumps(chat["messages"]),
            chat["last_updated"]
        )
    )
    conn.commit()


def delete_chat(user_id, chat_id):
    cursor.execute(
        "DELETE FROM chats WHERE user_id = ? AND chat_id = ?",
        (user_id, chat_id)
    )
    conn.commit()


# ----------------------------------------------------
# SESSION INIT
# ----------------------------------------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

if "rename_id" not in st.session_state:
    st.session_state.rename_id = None


# ----------------------------------------------------
# USER LOGIN (4-DIGIT CODE)
# ----------------------------------------------------
if st.session_state.user_id is None:
    st.title("🔐 Enter your 4-digit code")
    pin = st.text_input("4-digit code", max_chars=4)

    if pin and pin.isdigit() and len(pin) == 4:
        st.session_state.user_id = pin
        st.session_state.chats = load_chats(pin)
        st.session_state.current_chat = None
        st.rerun()
    else:
        st.info("Enter any 4-digit number to continue.")
    st.stop()


# ----------------------------------------------------
# HELPERS
# ----------------------------------------------------
def auto_title(text):
    return " ".join(text.split()[:6]).title() or "New Chat"


def create_chat():
    chat_id = str(uuid.uuid4())
    chat = {
        "name": "New Chat",
        "messages": [],
        "last_updated": datetime.datetime.now().timestamp()
    }
    st.session_state.chats[chat_id] = chat
    st.session_state.current_chat = chat_id
    save_chat(st.session_state.user_id, chat_id, chat)


# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
with st.sidebar:
    st.markdown(f"## 💬 ML Assistant")
    st.caption(f"User code: **{st.session_state.user_id}**")

    if st.button("➕ New Chat", use_container_width=True):
        create_chat()
        st.rerun()

    if st.button("🔄 Change User", use_container_width=True):
        st.session_state.user_id = None
        st.session_state.chats = {}
        st.session_state.current_chat = None
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
                delete_chat(st.session_state.user_id, chat_id)
                del st.session_state.chats[chat_id]
                if st.session_state.current_chat == chat_id:
                    st.session_state.current_chat = None
                st.rerun()

    if st.session_state.rename_id:
        cid = st.session_state.rename_id
        new_name = st.text_input("Rename chat", st.session_state.chats[cid]["name"])
        if st.button("Save"):
            st.session_state.chats[cid]["name"] = new_name
            save_chat(st.session_state.user_id, cid, st.session_state.chats[cid])
            st.session_state.rename_id = None
            st.rerun()


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
if st.session_state.current_chat is None:
    st.markdown("## 👋 Welcome")
    st.markdown("Create or select a chat from the sidebar.")
    st.stop()

chat_id = st.session_state.current_chat
chat = st.session_state.chats[chat_id]

st.markdown(f"## {chat['name']}")

for msg in chat["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])


user_input = st.chat_input("Ask anything from the ML course...")

if user_input:
    if chat["name"] == "New Chat" and not chat["messages"]:
        chat["name"] = auto_title(user_input)

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

    save_chat(st.session_state.user_id, chat_id, chat)
    st.rerun()
