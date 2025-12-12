# backend/graph_ml_assistant.py

import os
import sys
from typing import List, Optional

from typing_extensions import TypedDict

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.tools_rag import course_docs_search
from backend.tools_web import web_search
from backend.router_agent import classify_query
from backend.quiz_agent import generate_quiz
from backend.memory import recall_memory, store_memory


class GraphState(TypedDict, total=False):
    messages: List[AnyMessage]
    route: Optional[str]
    chapter: Optional[str]
    memory: Optional[str]


llm_teacher = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
)


def router_node(state: GraphState) -> GraphState:
    last_input = state["messages"][-1].content
    route_info = classify_query(last_input)

    state["route"] = route_info.get("type")
    state["chapter"] = route_info.get("chapter")
    return state


def memory_retriever_node(state: GraphState) -> GraphState:
    if not state.get("messages"):
        state["memory"] = ""
        return state

    last_msg = state["messages"][-1]
    if not isinstance(last_msg, HumanMessage):
        state["memory"] = ""
        return state

    recalled = recall_memory(last_msg.content, k=3)
    if recalled:
        memory_text = "\n".join(f"- {m}" for m in recalled)
        state["memory"] = f"Relevant long-term memories:\n{memory_text}"
    else:
        state["memory"] = ""

    return state


def memory_writer_node(state: GraphState) -> GraphState:
    messages = state.get("messages", [])
    if len(messages) < 2:
        return state

    last_ai = messages[-1]
    last_user = messages[-2]

    if not isinstance(last_ai, AIMessage) or not isinstance(last_user, HumanMessage):
        return state

    user_text = last_user.content.lower()
    ai_text = last_ai.content

    explicit_triggers = ["remember this", "save this", "store this", "note this"]

    if any(t in user_text for t in explicit_triggers):
        cleaned = last_user.content
        for t in explicit_triggers:
            cleaned = cleaned.replace(t, "")
        cleaned = cleaned.strip()
        if cleaned:
            store_memory(cleaned)
        return state

    if len(ai_text) > 200:
        store_memory(ai_text)

    return state


def _rag_is_useful(rag_text: str) -> bool:
    """
    Heuristic to decide if RAG has enough information.
    Adjust as needed.
    """
    if not rag_text:
        return False
    txt = rag_text.strip().lower()

    # Common "no info" patterns (adjust if your RAG tool returns different text)
    bad_markers = [
        "no relevant",
        "no results",
        "nothing found",
        "i couldn't find",
        "not found",
        "empty",
    ]
    if any(m in txt for m in bad_markers):
        return False

    # Too short => probably useless
    if len(txt) < 200:
        return False

    return True


def teacher_rag_or_web_node(state: GraphState) -> GraphState:
    """
    Primary path for course-related questions:
    1) Try RAG from course docs
    2) If insufficient → fallback to online web search
    """
    user_msg = state["messages"][-1].content
    memory_context = state.get("memory", "") or ""

    rag_context = course_docs_search.invoke(user_msg)
    use_rag = _rag_is_useful(rag_context)

    if use_rag:
        system_content = (
            "You are a Machine Learning course assistant.\n"
            "Answer using the COURSE EXCERPTS provided.\n"
            "If the excerpts do not contain enough info, say so briefly.\n\n"
        )
        if memory_context:
            system_content += f"{memory_context}\n\n"
        system_content += f"COURSE EXCERPTS:\n{rag_context}\n"
        system_msg = SystemMessage(content=system_content)

        result = llm_teacher.invoke([system_msg] + state["messages"])
        state["messages"].append(result)
        return state

    # --- WEB FALLBACK ---
    web = web_search(user_msg, max_results=5)

    if not web["ok"]:
        system_content = (
            "You are a Machine Learning course assistant.\n"
            "The course docs did not contain enough information and web search is unavailable.\n"
            "Explain what you can confidently from general ML knowledge, and clearly state limits.\n\n"
        )
        if memory_context:
            system_content += f"{memory_context}\n\n"
        system_msg = SystemMessage(content=system_content)
        result = llm_teacher.invoke([system_msg] + state["messages"])
        state["messages"].append(result)
        return state

    sources_block = "\n".join(
        f"- {r['title']} ({r['url']})\n  {r['content']}"
        for r in web["results"]
        if r.get("url")
    )

    system_content = (
        "You are a Machine Learning course assistant.\n"
        "The course docs did not contain enough information; use the WEB SOURCES below.\n"
        "Write a clear answer.\n"
        "At the end, add a short 'Sources:' list with the URLs you used.\n\n"
    )
    if memory_context:
        system_content += f"{memory_context}\n\n"
    system_content += f"WEB SOURCES:\n{sources_block}\n"

    system_msg = SystemMessage(content=system_content)
    result = llm_teacher.invoke([system_msg] + state["messages"])
    state["messages"].append(result)
    return state


def teacher_general_node(state: GraphState) -> GraphState:
    memory_context = state.get("memory", "") or ""

    content = (
        "You are a Machine Learning teaching assistant. "
        "Explain clearly with examples.\n"
        "If you are unsure, say so and suggest what to look up.\n"
    )
    if memory_context:
        content += "\nHere is long-term memory about this student or past sessions:\n"
        content += memory_context

    system_msg = SystemMessage(content=content)
    result = llm_teacher.invoke([system_msg] + state["messages"])
    state["messages"].append(result)
    return state


def quiz_node(state: GraphState) -> GraphState:
    # QUIZ PATH UNCHANGED ✅ (no web search here)
    chapter = state.get("chapter")
    quiz = generate_quiz(chapter=chapter, n_questions=5)
    state["messages"].append(AIMessage(content=quiz))
    return state


def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("memory_retriever", memory_retriever_node)
    builder.add_node("router", router_node)

    # NOTE: this node now handles BOTH: RAG + web fallback
    builder.add_node("teacher_rag_or_web", teacher_rag_or_web_node)

    builder.add_node("teacher_general", teacher_general_node)
    builder.add_node("quiz", quiz_node)
    builder.add_node("memory_writer", memory_writer_node)

    builder.set_entry_point("memory_retriever")
    builder.add_edge("memory_retriever", "router")

    builder.add_conditional_edges(
        "router",
        lambda state: state.get("route"),
        {
            "rag_query": "teacher_rag_or_web",
            "general_explanation": "teacher_general",
            "quiz_request": "quiz",
        },
    )

    builder.add_edge("teacher_rag_or_web", "memory_writer")
    builder.add_edge("teacher_general", "memory_writer")
    builder.add_edge("quiz", "memory_writer")
    builder.add_edge("memory_writer", END)

    return builder.compile()


graph_app = build_graph()
