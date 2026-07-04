import streamlit as st
import os
import asyncio
from threading import Thread
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage

# Use the official, modern LangGraph agent executor instead!
from langgraph.prebuilt import create_react_agent



# --- App Configuration ---
st.set_page_config(page_title="Financial Analyst Chatbot", layout="wide")
st.title("🤖 Financial Analyst Chatbot")
st.write("This chatbot uses Model Context Protocol (MCP) and DuckDB to analyze your structured financial data.")

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

# --- Thread-Safe Event Loop Runner ---
# Streamlit runs synchronously, so we run our async MCP client on a dedicated background thread.
if "async_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    t = Thread(target=loop.run_forever, daemon=True)
    t.start()
    st.session_state.async_loop = loop


def run_async(coro):
    """Utility to run an async coroutine on our background thread loop."""
    future = asyncio.run_coroutine_threadsafe(coro, st.session_state.async_loop)
    return future.result()


# --- MCP & LangChain Agent Setup ---
@st.cache_resource
def setup_agent():
    """Spawns the MCP server via stdio, extracts tools, and builds the LangGraph agent."""

    server_path = os.path.abspath("mcp_server.py")

    server_params = StdioServerParameters(
        command="python",
        args=[server_path],
        env=os.environ.copy()
    )

    async def connect_and_discover_tools():
        transport_context = stdio_client(server_params)
        read, write = await transport_context.__aenter__()
        st.session_state.transport_context = transport_context

        session = ClientSession(read, write)
        st.session_state.mcp_session = session
        await session.initialize()

        mcp_tools = await session.list_tools()
        langchain_tools = []

        for t in mcp_tools.tools:
            def make_wrapper(tool_name=t.name):
                def sync_wrapper(*args, **kwargs):
                    async def call():
                        res = await session.call_tool(tool_name, arguments=kwargs)
                        return res.content[0].text

                    return run_async(call())

                return sync_wrapper

            langchain_tools.append(
                Tool(
                    name=t.name,
                    func=make_wrapper(),
                    description=t.description
                )
            )
        return langchain_tools

    # Discover tools
    tools = run_async(connect_and_discover_tools())

    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Define our clean, straightforward system prompt
    system_prompt = """You are a powerful financial analyst. 
First, use 'get_db_schema' to understand the available database columns. 
Then, use 'query_db' to retrieve the financial data. 
Finally, use calculation tools like compute_cagr or compute_volatility if needed. 
Always show a clear markdown table of the data you retrieved before outputting your final answer."""

    # Create the modern LangGraph React Agent
    # It takes the model, the list of tools, and state instructions
    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )
    return agent


# Initialize the synchronized agent executor
agent_executor = setup_agent()

# --- Streamlit Chat Interface ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous conversation messages
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    else:
        with st.chat_message("assistant"):
            st.write(message.content)

# Accept user input
user_query = st.chat_input("Ask a question about your financial data...")
if user_query:
    # Add human message to chat history
    st.session_state.chat_history.append(HumanMessage(user_query))
    with st.chat_message("user"):
        st.write(user_query)

    # Generate assistant answer
    with st.chat_message("assistant"):
        with st.spinner("Analyzing financial logs..."):
            try:
                # Execute agent with previous chat history
                response = agent_executor.invoke({
                    "input": user_query,
                    "chat_history": st.session_state.chat_history
                })
                output_text = response["output"]
                st.session_state.chat_history.append(AIMessage(output_text))
                st.write(output_text)
            except Exception as e:
                st.error(f"An error occurred: {e}")
