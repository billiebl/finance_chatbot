import streamlit as st
import os
import asyncio
from threading import Thread
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from langchain_huggingface import HuggingFacePipeline
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage

# --- App Configuration ---
st.set_page_config(page_title="Financial Analyst Chatbot", layout="wide")
st.title("💬 Financial Analyst Chatbot")
st.write("This chatbot uses Model Context Protocol (MCP) and DuckDB to analyze your structured financial data.")

# --- System Prompt ---
SYSTEM_PROMPT = """You are a powerful financial analyst assistant.
- Your primary goal is to help users by answering questions about financial data.
- First, use the 'get_db_schema' tool to understand the available tables and their columns.
- Next, use the 'query_db' tool to retrieve the necessary data with an SQL query.
- If the user's question requires calculations like CAGR or volatility, use the 'compute_cagr' or 'compute_volatility' tools.
- Always present the data you have retrieved before providing your final answer.
- If you don't know the answer or cannot find the data, simply state that."""


# --- Asyncio Event Loop Setup (Cached Globally) ---
@st.cache_resource
def get_event_loop():
    """Creates and starts a background event loop running in a separate thread.
    This runs globally and avoids any dependency on st.session_state.
    """
    loop = asyncio.new_event_loop()
    thread = Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return loop


def run_async(coro):
    """Runs an async coroutine in the background event loop safely."""
    loop = get_event_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


# --- Model and MCP Connection (Cached Globally) ---
@st.cache_resource
def get_llm():
    model_name = "AdaptLLM/finance-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0,
        do_sample=False,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=pipe)


@st.cache_resource
def connect_to_mcp():
    """Establishes and caches the MCP session and transport."""
    async def connect():
        # Split the command ("python") from the arguments (the script path)
        params = StdioServerParameters(
            command="python",
            args=[os.path.abspath("mcp_server.py")]
        )
        transport = stdio_client(params)
        reader, writer = await transport.__aenter__()
        session = ClientSession(reader, writer)
        await session.initialize()
        return transport, session
    return run_async(connect())



# Establish and retrieve the global MCP connection
transport, session = connect_to_mcp()


# --- LangChain Tools and Agent Setup (Cached Globally) ---
@st.cache_resource
def build_agent_and_tools(_session):
    """Builds the LangChain tools and the ReAct agent.
    The underscore prefix (_session) tells Streamlit not to hash this complex parameter.
    """
    mcp_tools_list = run_async(_session.list_tools())
    langchain_tools = []

    for t in mcp_tools_list.tools:
        def make_wrapper(tool_name):
            def wrapper(**kwargs):
                async def call():
                    result = await _session.call_tool(tool_name, arguments=kwargs)
                    return "\n".join(part.text for part in result.content if hasattr(part, 'text'))

                return run_async(call())

            return wrapper

        langchain_tools.append(
            Tool(name=t.name, description=t.description, func=make_wrapper(t.name))
        )

    # We supply state_modifier to bind our system prompt into LangGraph's ReAct agent
    agent = create_react_agent(model=get_llm(), tools=langchain_tools, state_modifier=SYSTEM_PROMPT)
    return agent, langchain_tools


# Build the agent and tools
agent, tools = build_agent_and_tools(session)

# --- Streamlit Chat Interface ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous chat messages
for message in st.session_state.chat_history:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.write(message.content)

# Input field for user query
if user_query := st.chat_input("Ask a question about your financial data..."):
    # Append and show user message
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.write(user_query)

    # Generate and show agent response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                # Format correct LangGraph input state
                agent_input = {
                    "messages": st.session_state.chat_history,
                }

                response = agent.invoke(agent_input)

                # Retrieve the last message from the updated LangGraph state
                last_message = response["messages"][-1]
                ai_message = last_message.content if hasattr(last_message, 'content') else str(last_message)

                # Append and show AI response
                st.session_state.chat_history.append(AIMessage(content=ai_message))
                st.write(ai_message)

            except Exception as e:
                st.error(f"An error occurred: {e}")
