import streamlit as st
import uuid
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import TypedDict, Annotated
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from tools import add, multiply
from psycopg_pool import ConnectionPool

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LangGraph Chat with Tools",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

  .stApp { background: #0d0f14; color: #e2e8f0; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #111318 !important;
    border-right: 1px solid #1e2130;
  }
  [data-testid="stSidebar"] .stButton button {
    width: 100%;
    text-align: left;
    background: #1a1d27;
    color: #94a3b8;
    border: 1px solid #1e2130;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    padding: 0.4rem 0.7rem;
    margin-bottom: 4px;
    transition: all 0.15s;
  }
  [data-testid="stSidebar"] .stButton button:hover {
    background: #1e2233;
    border-color: #3b82f6;
    color: #e2e8f0;
  }

  /* Active thread button */
  .active-thread button {
    background: #1e2f4a !important;
    border-color: #3b82f6 !important;
    color: #93c5fd !important;
  }

  /* Chat messages */
  .chat-msg {
    display: flex;
    gap: 12px;
    padding: 14px 18px;
    border-radius: 10px;
    margin-bottom: 10px;
    font-size: 0.92rem;
    line-height: 1.6;
  }
  .chat-msg.user {
    background: #1a1d27;
    border: 1px solid #1e2130;
    flex-direction: row-reverse;
  }
  .chat-msg.assistant {
    background: #111827;
    border: 1px solid #1e2a40;
  }
  .chat-avatar {
    font-size: 1.1rem;
    flex-shrink: 0;
    line-height: 1.6;
  }
  .chat-content { flex: 1; white-space: pre-wrap; }

  /* Input */
  .stTextInput input, .stChatInput textarea {
    background: #1a1d27 !important;
    border: 1px solid #1e2130 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
  }

  /* Buttons */
  .stButton > button {
    background: #1e3a5f;
    color: #93c5fd;
    border: 1px solid #3b82f6;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
  }
  .stButton > button:hover {
    background: #1d4ed8;
    color: white;
  }

  h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
    color: #93c5fd;
  }

  .thread-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #475569;
    margin-bottom: 6px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
</style>
""", unsafe_allow_html=True)


# ── Graph setup (cached) ──────────────────────────────────────────────────────
DB_URI = "postgresql://postgres:2004@localhost:5432/langgraph-chatdb1"

@st.cache_resource
def build_graph():
    llm = ChatOllama(model="qwen2.5:7b")
    tools = [add, multiply]
    llm_with_tools = llm.bind_tools(tools=tools)
    tool_node = ToolNode(tools=tools)

    class ChatState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    def chat_node(state: ChatState):
        response = llm_with_tools.invoke(state["messages"])
        if response.tool_calls:
            tool_names = [tc["name"] for tc in response.tool_calls]
            print(f"[TOOL CALL] calling: {tool_names}", flush=True)
        else:
            print("[NO TOOL CALL] responding directly", flush=True)
        return {"messages": [response]}

    def should_continue(state):
        last = state["messages"][-1]
        return "tools" if last.tool_calls else END

    g = StateGraph(ChatState)
    g.add_node("agent", chat_node)
    g.add_node("tools", tool_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")

    pool = ConnectionPool(
        conninfo=DB_URI,
        max_size=10,
        kwargs={"autocommit": True},
    )
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()

    return g.compile(checkpointer=checkpointer), checkpointer


graph_builder, checkpointer = build_graph()


# ── Helpers ───────────────────────────────────────────────────────────────────
def list_threads():
    seen = set()
    for cp in checkpointer.list(None):
        seen.add(cp.config["configurable"]["thread_id"])
    return sorted(seen)


def load_messages(thread_id: str):
    state = graph_builder.get_state({"configurable": {"thread_id": thread_id}})
    msgs = state.values.get("messages", [])
    return [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant",
         "content": m.content}
        for m in msgs
        if isinstance(m, (HumanMessage, AIMessage)) and m.content
    ]


def send_message(thread_id: str, user_input: str):
    config = {"configurable": {"thread_id": thread_id}}
    graph_builder.invoke({"messages": [HumanMessage(content=user_input)]}, config)


# ── Session state ─────────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "threads_cache" not in st.session_state:
    st.session_state.threads_cache = list_threads()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 LangGraph")
    st.markdown("---")

    if st.button("＋  New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.threads_cache = list_threads()
        st.rerun()

    st.markdown('<div class="thread-label">Past threads</div>', unsafe_allow_html=True)

    threads = st.session_state.threads_cache
    if not threads:
        st.caption("No threads yet.")
    for tid in reversed(threads):
        short = tid[:8] + "…"
        is_active = tid == st.session_state.thread_id
        cls = "active-thread" if is_active else ""
        with st.container():
            st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
            if st.button(f"{'▶ ' if is_active else ''}{short}", key=f"t_{tid}"):
                st.session_state.thread_id = tid
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────
if st.session_state.thread_id is None:
    st.markdown("## Welcome")
    st.markdown("Start a **new conversation** from the sidebar, or pick an existing thread.")
else:
    tid = st.session_state.thread_id
    st.markdown(f"#### Thread `{tid[:8]}…`")

    messages = load_messages(tid)

    chat_area = st.container()
    with chat_area:
        if not messages:
            st.markdown('<p style="color:#475569;font-style:italic;">No messages yet. Say hello!</p>',
                        unsafe_allow_html=True)
        for msg in messages:
            avatar = "🧑" if msg["role"] == "user" else "🤖"
            css_cls = msg["role"]
            st.markdown(
                f'<div class="chat-msg {css_cls}">'
                f'<div class="chat-avatar">{avatar}</div>'
                f'<div class="chat-content">{msg["content"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    user_input = st.chat_input("Type a message…")
    if user_input:
        # Optimistically show the user message
        st.markdown(
            f'<div class="chat-msg user"><div class="chat-avatar">🧑</div>'
            f'<div class="chat-content">{user_input}</div></div>',
            unsafe_allow_html=True,
        )
        with st.spinner("Thinking…"):
            send_message(tid, user_input)
        # Refresh threads list in case this is a brand-new thread
        st.session_state.threads_cache = list_threads()
        st.rerun()