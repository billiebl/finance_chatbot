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



# --- App Configuration ---
st.set_page_config(page_title="Financial Analyst Chatbot", layout="wide")
st.title("Chatbot")
st.write("This chatbot uses Model Context Protocol (MCP) and DuckDB to analyze your structured financial data.")


if "loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    Thread(target=loop.run_forever, daemon=True).start()
    st.session_state.loop = loop


def run_async(coro):
    fut = asyncio.run_coroutine_threadsafe(coro, st.session_state.loop)
    return fut.result()

@st.cache_resource
@st.cache_resource
def get_llm():
    model_name = "AdaptLLM/finance-chat"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
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


async def connect():
    params = StdioServerParameters(
        command="python",
        args=[os.path.abspath("mcp_server.py")],
    )
    transport = stdio_client(params)
    r, w = await transport.__aenter__()
    session = ClientSession(r, w)
    await session.initialize()
    return transport, session


if "mcp_session" not in st.session_state:
    transport, session = run_async(connect())
    st.session_state.transport = transport
    st.session_state.mcp_session = session


def build_tools():
    session = st.session_state.mcp_session
    mcp_tools = run_async(session.list_tools())
    tools = []

    from langchain_core.tools import Tool

    for t in mcp_tools.tools:
        def make_wrapper(name=t.name):
            def wrapper(**kwargs):
                async def call():
                    result = await session.call_tool(name, arguments=kwargs)
                    return "\n".join(
                        getattr(x, "text", str(x))
                        for x in result.content
                    )
                return run_async(call())
            return wrapper

        tools.append(
            Tool(
                name=t.name,
                description=t.description,
                func=make_wrapper(),
            )
        )
    return tools


if "tools" not in st.session_state:
    st.session_state.tools = build_tools()

if "agent" not in st.session_state:
    st.session_state.agent = create_react_agent(
        model=get_llm(),
        tools=st.session_state.tools,
        prompt=SYSTEM_PROMPT,
    )

agent = st.session_state.agent

