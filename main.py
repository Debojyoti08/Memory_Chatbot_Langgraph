from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Store for managing multiple conversation sessions
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Get or create a chat history for a session"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Create a prompt with memory placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Initialize the model
llm = ChatOllama(model="qwen2.5:7b")

# Create a chain
chain = prompt | llm

# Wrap with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Have a conversation (note the session_id in config)
config = {"configurable": {"session_id": "alice_session"}}

response1 = chain_with_history.invoke(
    {"input": "Hi, my name is Alice"}, 
    config=config
)
print("AI:", response1.content)

# Continue conversation - history is automatically managed!
response2 = chain_with_history.invoke(
    {"input": "What's my name?"}, 
    config=config
)
print("AI:", response2.content)

# Check the conversation history
history = get_session_history("alice_session")
print("\nConversation history:")
for message in history.messages:
    print(f"{message.__class__.__name__}: {message.content}")