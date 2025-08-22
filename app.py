import os
import ast
import operator as op
from typing import Dict

import streamlit as st
from dotenv import load_dotenv

# LangChain / Google Generative AI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(
    page_title="AI Agent — Streamlit + Gemini + Tools + Memory",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AI Agent (Streamlit + Gemini + Tools + Memory)")
st.caption("Chat • Tools (Calculator, Web Search) • Conversation Memory")

# -----------------------------
# Env & Sidebar Controls
# -----------------------------
load_dotenv()

# ✅ Get API Keys from .env
google_api_key = os.getenv("GOOGLE_API_KEY")
serpapi_key = os.getenv("SERPAPI_API_KEY")

with st.sidebar:
    st.header("Settings")

    st.markdown("---")
    model = st.selectbox(
        "Model",
        ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    max_iters = st.slider("Max Agent Iterations", 1, 25, 10)
    show_steps = st.toggle("Show tool calls", value=True)

    st.markdown("---")
    if st.button("🧹 Clear chat"):
        st.session_state.pop("session_id", None)
        st.session_state.pop("store", None)
        st.session_state.pop("rendered_messages", None)
        st.rerun()

# -----------------------------
# Safe Calculator (no eval)
# -----------------------------
ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}

def _safe_eval_expr(node):
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
        return ALLOWED_OPS[type(node.op)](_safe_eval_expr(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPS:
        left = _safe_eval_expr(node.left)
        right = _safe_eval_expr(node.right)
        return ALLOWED_OPS[type(node.op)](left, right)
    raise ValueError("Unsupported expression")

@tool
def calculator(query: str) -> str:
    """Solve math expressions safely."""
    try:
        tree = ast.parse(query, mode="eval")
        result = _safe_eval_expr(tree.body)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# -----------------------------
# SerpAPI Search Tool
# -----------------------------
def get_search_wrapper(api_key: str):
    return SerpAPIWrapper(serpapi_api_key=api_key) if api_key else None

def make_search_tool(wrapper: SerpAPIWrapper):
    @tool
    def search_tool(query: str) -> str:
        """Search the web for current information, news, and facts."""
        try:
            return wrapper.run(query)
        except Exception as e:
            return f"Search error: {e}"

    return search_tool

# -----------------------------
# Build LLM + Tools + Agent
# -----------------------------
if not google_api_key:
    st.info("⚠️ Add your GOOGLE_API_KEY in the .env file to begin.")

llm = ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=google_api_key)

# Tools
tools = [calculator]
search_wrapper = get_search_wrapper(serpapi_key)
if search_wrapper is not None:
    tools.append(make_search_tool(search_wrapper))
else:
    st.warning("⚠️ SERPAPI_API_KEY missing — Web Search tool disabled.")

# Prompt
try:
    react_prompt = hub.pull("hwchase17/react")
except Exception:
    react_prompt = PromptTemplate.from_template(
        """
You are a helpful AI agent. You have access to tools. Use them when helpful.
Follow the ReAct pattern: Think -> Act -> Observe -> ... -> Final Answer.
{tools}

User: {input}
{agent_scratchpad}
        """.strip()
    )

agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=show_steps,
    handle_parsing_errors=True,
    max_iterations=max_iters,
    return_intermediate_steps=True,
)

# -----------------------------
# Memory
# -----------------------------
if "store" not in st.session_state:
    st.session_state.store = {}
if "session_id" not in st.session_state:
    st.session_state.session_id = os.urandom(8).hex()

store: Dict[str, ChatMessageHistory] = st.session_state.store
session_id = st.session_state.session_id

def get_session_history(sid: str) -> ChatMessageHistory:
    if sid not in store:
        store[sid] = ChatMessageHistory()
    return store[sid]

agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# -----------------------------
# Chat UI
# -----------------------------
history = get_session_history(session_id)
for m in history.messages:
    role = "user" if m.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(m.content)

user_msg = st.chat_input("Type your message…")

if user_msg and google_api_key:
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            result = agent_with_history.invoke(
                {"input": user_msg},
                config={"configurable": {"session_id": session_id}},
            )

            if show_steps and isinstance(result, dict) and "intermediate_steps" in result:
                steps = result.get("intermediate_steps", [])
                for action, obs in steps:
                    with st.expander(f"🔧 Tool: {action.tool}"):
                        st.markdown(f"**Input:** `{action.tool_input}`")
                        st.markdown("**Output:**")
                        st.code(str(obs))

            output_text = result.get("output") if isinstance(result, dict) else str(result)
            placeholder.markdown(output_text)
        except Exception as e:
            placeholder.error(f"Error: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit, LangChain, and Google Gemini.")
