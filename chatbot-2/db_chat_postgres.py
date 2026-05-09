from langchain_ollama import ChatOllama
from langgraph.graph import START, StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage
from typing import TypedDict, Annotated
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(model="qwen2.5:7b")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# ─── Postgres connection ───────────────────────────────────────────────────────
DB_URI = "postgresql://postgres:2004@localhost:5432/langgraph-chatdb1"

check_pointer_cm = PostgresSaver.from_conn_string(DB_URI)
check_pointer = check_pointer_cm.__enter__()

check_pointer.setup()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=check_pointer)

def retrieve_all_threads():
    threads = set()
    for checkpoint in check_pointer.list(None):
        threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(threads)

def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )

    messages = state.values.get("messages", [])

    return [
        {
            "role": "user" if isinstance(msg, HumanMessage) else "assistant",
            "content": msg.content
        }
        for msg in messages
    ]