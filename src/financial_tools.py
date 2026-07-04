import streamlit as st
import os
import asyncio
import subprocess
from mcp import ClientSession
from mcp.client.stdio import stdio_client_manager
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage

# --- App Configuration ---
st.set_page_config(page_title="Financial Analyst Chatbot", layout="wide")
st.title("🤖 Financial Analyst Chatbot")
st.write("This chatbot uses a series of tools to answer questions about your financial data.")

# Set your OpenAI API key (replace with your key or use secrets)
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

# --- MCP & LangChain Agent Setup ---
@st.cache_resource
def start_mcp_server():
    """Starts the mcp_server.py as a background process."""
    server_process = subprocess.Popen(
        ["python", "mcp_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return server_process

@st.cache_resource
def setup_agent():
    """Connects to MCP and builds the LangChain agent."""
    server_proc = start_mcp_server()

    async def get_tools_async():
        """Helper to run async MCP connection within sync Streamlit."""
        async with stdio_client_manager(server_proc) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                mcp_tools = await session.list_tools()

                langchain_tools = []
                for t in mcp_tools.tools:
                    def make_wrapper(tool_name=t.name):
                        # Wrapper to call async tool from sync agent
                        def sync_wrapper(*args, **kwargs):
                            result = asyncio.run(session.call_tool(tool_name, arguments=kwargs))
                            return result.content[0].text
                        return sync_wrapper

                    langchain_tools.append(Tool(name=t.name, func=make_wrapper(), description=t.description))
                return langchain_tools

    # Run the async setup
    tools = asyncio.run(get_tools_async())

    # Build LangChain Agent
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a powerful financial analyst. First, use 'get_db_schema' to understand the data. Then, use 'query_db' to get data. Finally, use calculation tools if needed. Always show the data you retrieved before your final answer."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# Initialize Agent
agent_executor = setup_agent()

# --- Streamlit Chat Interface ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
    else:
        with st.chat_message("AI"):
            st.write(message.content)

# Get user input
user_query = st.chat_input("Ask a question about your financial data...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(user_query))
    with st.chat_message("Human"):
        st.write(user_query)

    with st.chat_message("AI"):
        with st.spinner("Analyzing..."):
            try:
                # We need to pass history for the agent to have context
                response = agent_executor.invoke({
                    "input": user_query,
                    "chat_history": st.session_state.chat_history
                })
                st.session_state.chat_history.append(AIMessage(response["output"]))
                st.write(response["output"])
            except Exception as e:
                st.error(f"An error occurred: {e}")

