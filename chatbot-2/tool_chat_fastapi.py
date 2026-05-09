from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage
from typing import TypedDict, Annotated
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph.message import add_messages
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
from tools import add, multiply

llm = ChatOllama(model="qwen2.5:7b")
tools = [add, multiply]

llm_with_tools = llm.bind_tools(tools=tools)
tool_node = ToolNode(tools=tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

DB_URI = "postgresql://postgres:2004@localhost:5432/langgraph-chatdb1"

check_pointer_cm = PostgresSaver.from_conn_string(DB_URI)
check_pointer = check_pointer_cm.__enter__()

check_pointer.setup()

def should_continue(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

graph = StateGraph(ChatState)
graph.add_node("agent", chat_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent", 
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
graph.add_edge("tools", "agent")
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
    
# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(title="LangGraph Chatbot with Memory")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# Request Models
# -------------------------
class ChatRequest(BaseModel):
    thread_id: str
    message: str


class NewThreadResponse(BaseModel):
    thread_id: str


# -------------------------
# Create New Thread
# -------------------------
@app.post("/new-thread", response_model=NewThreadResponse)
def create_thread():
    thread_id = str(uuid.uuid4())
    return {"thread_id": thread_id}


# -------------------------
# Chat Endpoint
# -------------------------
@app.post("/chat")
def chat(request: ChatRequest):
    try:
        config = {
            "configurable": {
                "thread_id": request.thread_id
            }
        }

        response = chatbot.invoke(
            {
                "messages": [HumanMessage(content=request.message)]
            },
            config=config
        )

        last_message = response["messages"][-1]

        return {
            "thread_id": request.thread_id,
            "response": last_message.content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# Get Conversation History
# -------------------------
@app.get("/history/{thread_id}")
def get_history(thread_id: str):
    try:
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# List All Threads
# -------------------------
@app.get("/threads")
def get_threads():
    threads = set()

    for checkpoint in check_pointer.list(None):
        threads.add(checkpoint.config["configurable"]["thread_id"])

    return {"threads": list(threads)}

