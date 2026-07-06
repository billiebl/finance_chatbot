import streamlit as st
import os
import asyncio
from threading import Thread
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Trading Data Analysis Chatbot", layout="wide")
st.title("Trading Data Analysis Chatbot")

SYSTEM_PROMPT = """You are a powerful financial analyst assistant.
- Your primary goal is to help users by answering questions about financial data.
- First, use the 'get_db_schema' tool to understand the available tables and their columns.
- Next, use the 'query_db' tool to retrieve the necessary data with an SQL query.
- If the user's question requires calculations like CAGR or volatility, use the 'compute_cagr' or 'compute_volatility' tools.
- Always present the data you have retrieved before providing your final answer.
- If you don't know the answer or cannot find the data, simply state that."""


@st.cache_resource
def get_event_loop():
    loop = asyncio.new_event_loop()
    thread = Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return loop


def run_async(coro):
    loop = get_event_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()

@st.cache_resource
def get_llm():
    model_name = "AdaptLLM/finance-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
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
    async def connect():
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

transport, session = connect_to_mcp()
@st.cache_resource
def build_agent_and_tools(_session):
    mcp_tools_list = run_async(_session.list_tools())
    langchain_tools = []
    for t in mcp_tools_list.tools:
        def make_wrapper(tool_name):
            def wrapper(**kwargs):
                async def call():
                    result = await _session.call_tool(tool_name, arguments=kwargs)
                    return "\n".join(part.text for part in result.content if hasattr(part, "text"))
                return run_async(call())
            return wrapper
        langchain_tools.append(
            Tool(name=t.name, description=t.description, func=make_wrapper(t.name))
        )
    agent = create_react_agent(model=get_llm(), tools=langchain_tools, state_modifier=SYSTEM_PROMPT)
    return agent, langchain_tools


agent, tools = build_agent_and_tools(session)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.write(message.content)

if user_query := st.chat_input("Ask a question about your financial data..."):
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                agent_input = {
                    "messages": st.session_state.chat_history,
                }
                response = agent.invoke(agent_input)
                last_message = response["messages"][-1]
                ai_message = last_message.content if hasattr(last_message, 'content') else str(last_message)
                st.session_state.chat_history.append(AIMessage(content=ai_message))
                st.write(ai_message)

            except Exception as e:
                st.error(f"An error occurred: {e}")
