"""
Smart Coding Copilot (Agent + Tools + Memory)
------------------------------------------------
A Streamlit app that combines:
- Agent: Google Generative AI via LangChain (ChatGoogleGenerativeAI)
- Tools: Run Python code safely (subprocess), Web search via SerpAPI
- Memory: Conversational memory + persistent JSON-based error & style memory

"""

import ast
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime
from typing import Dict, Any, List

import streamlit as st
from dotenv import load_dotenv

# LangChain core
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.schema import SystemMessage

# LLM: Google Generative AI
from langchain_google_genai import ChatGoogleGenerativeAI

# SerpAPI Tool
from langchain_community.utilities import SerpAPIWrapper

###############################################################################
# App Setup
###############################################################################
APP_TITLE = "Smart Coding Copilot "
MEMORY_FILE = "copilot_memory.json"  # persistent memory on disk
MAX_ERROR_MEMORY = 200  # max stored error records

load_dotenv(override=True)

st.set_page_config(page_title=APP_TITLE, page_icon="💻", layout="wide")
st.title(APP_TITLE)
st.caption("Hybrid agent with Google GenAI + SerpAPI + code runner + long-term memory")

###############################################################################
# Helpers: Persistent Memory (errors, style)
###############################################################################
def _load_persistent_memory(path: str = MEMORY_FILE) -> Dict[str, Any]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"errors": [], "style_notes": []}
    return {"errors": [], "style_notes": []}


def _save_persistent_memory(mem: Dict[str, Any], path: str = MEMORY_FILE) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mem, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Couldn't save memory: {e}")


def _summarize_errors(mem: Dict[str, Any], limit: int = 10) -> str:
    errors = mem.get("errors", [])[-limit:]
    if not errors:
        return "No prior errors recorded."
    blocks = []
    for e in errors:
        ts = e.get("timestamp", "")
        blocks.append(
            f"- [{ts}] {e.get('error_type','?')}: {e.get('message','')}\n  Snippet: {e.get('code_snippet','')[:160]}..."
        )
    return "\n".join(blocks)


def _update_style_notes(mem: Dict[str, Any], code: str) -> None:
    # very lightweight heuristics to capture coding style hints
    notes: List[str] = mem.get("style_notes", [])
    try:
        tree = ast.parse(code)
        func_defs = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
        class_defs = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
        if func_defs > 0 and "Uses functions" not in notes:
            notes.append("Uses functions")
        if class_defs > 0 and "Uses classes/OOP" not in notes:
            notes.append("Uses classes/OOP")
    except Exception:
        pass

    if len(code) > 800 and "Prefers longer scripts" not in notes:
        notes.append("Prefers longer scripts")
    if "import numpy" in code and "Uses numpy" not in notes:
        notes.append("Uses numpy")
    mem["style_notes"] = notes[:30]


def _style_summary(mem: Dict[str, Any]) -> str:
    notes = mem.get("style_notes", [])
    if not notes:
        return "No style patterns detected yet."
    return ", ".join(notes)

###############################################################################
# Tool: Safe-ish code execution (Python)
###############################################################################

def run_python_code(code: str, timeout: int = 8) -> Dict[str, Any]:
    """Execute Python code in a temp file using subprocess and capture outputs.
    NOTE: This is NOT fully safe. Use isolation for real deployments.
    """
    # Dedent to help with pasted code blocks
    code = textwrap.dedent(code)

    with tempfile.TemporaryDirectory() as tmp:
        script = os.path.join(tmp, "main.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(code)
        cmd = [sys.executable, script]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
            )
            return {
                "ok": proc.returncode == 0,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "stdout": "",
                "stderr": f"Execution timed out after {timeout}s",
                "returncode": -1,
            }


def analyze_traceback(tb: str) -> Dict[str, str]:
    """Extract a simple (type, message) from Python traceback text."""
    if not tb:
        return {"error_type": "", "message": ""}
    last_line = tb.strip().splitlines()[-1]
    if ": " in last_line:
        etype, msg = last_line.split(": ", 1)
        return {"error_type": etype.strip(), "message": msg.strip()}
    return {"error_type": "Error", "message": last_line.strip()}

###############################################################################
# Sidebar: Keys and settings
###############################################################################
st.sidebar.header("Settings")

# Keys will only come from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY", "")

if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
if SERPAPI_KEY:
    os.environ["SERPAPI_API_KEY"] = SERPAPI_KEY

model_name = st.sidebar.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0)
max_tokens = st.sidebar.slider("Max response tokens", 256, 4096, 1024, step=64)

###############################################################################
# Initialize LLM, Tools, Memory, and Agent
###############################################################################
if "persistent_memory" not in st.session_state:
    st.session_state.persistent_memory = _load_persistent_memory()

# LLM
llm = ChatGoogleGenerativeAI(
    model=model_name,
    temperature=0.2,
    max_output_tokens=max_tokens,
)

# Tools
serp = SerpAPIWrapper()
search_tool = Tool(
    name="web_search",
    func=serp.run,
    description=(
        "Useful for searching the web for programming questions, error messages, "
        "library usage, and documentation. Input: a natural language query."
    ),
)


def _run_code_tool(inp: str) -> str:
    """Agent-exposed tool to run code input as-is."""
    result = run_python_code(inp)
    output = []
    output.append(f"returncode={result['returncode']}")
    if result["stdout"]:
        output.append("STDOUT:\n" + result["stdout"])  # noqa: E501
    if result["stderr"]:
        output.append("STDERR:\n" + result["stderr"])  # noqa: E501
    return "\n".join(output)

run_code_tool = Tool(
    name="run_python_code",
    func=_run_code_tool,
    description=(
        "Execute raw Python code. Input must be a complete Python script. "
        "Returns return code, STDOUT, and STDERR. Use with caution."
    ),
)

# Conversational memory (short-term)
conv_memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=6,
    return_messages=True,
)

# System instruction to make the agent error-aware and style-aware
system_context = SystemMessage(
    content=(
        "You are a coding copilot. Give concise, actionable suggestions, diffs, and examples.\n"
        "Prefer step-by-step fixes and cite tool outputs when relevant.\n"
        "Leverage prior user error patterns and coding style notes to prevent repeat mistakes.\n"
        "When running code, warn about side effects and assume a clean environment.\n"
    )
)

agent = initialize_agent(
    tools=[search_tool, run_code_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    memory=conv_memory,
)

###############################################################################
# UI: Chat + Code Runner
###############################################################################
col_chat, col_code = st.columns([0.55, 0.45])

with col_code:
    st.subheader("Your Code")
    code_input = st.text_area(
        "Paste Python code here (the runner executes this).",
        height=360,
        placeholder="""# Example\nprint('Hello Smart Copilot')\n""",
        key="code_area",
    )
    run_clicked = st.button("▶️ Run Code", type="primary")

    if run_clicked and code_input.strip():
        res = run_python_code(code_input)
        st.write("**Return code:**", res["returncode"])
        if res["stdout"]:
            st.text_area("STDOUT", res["stdout"], height=160)
        if res["stderr"]:
            st.text_area("STDERR", res["stderr"], height=160)

        # Update memory from errors & style
        if res["stderr"]:
            err = analyze_traceback(res["stderr"])
            mem = st.session_state.persistent_memory
            mem.setdefault("errors", [])
            mem["errors"].append(
                {
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "error_type": err.get("error_type"),
                    "message": err.get("message"),
                    "code_snippet": code_input[:500],
                }
            )
            mem["errors"] = mem["errors"][-MAX_ERROR_MEMORY:]
            _update_style_notes(mem, code_input)
            _save_persistent_memory(mem)
            st.success("Error recorded to memory. The agent will try to prevent this next time.")
        else:
            # still collect style hints
            mem = st.session_state.persistent_memory
            _update_style_notes(mem, code_input)
            _save_persistent_memory(mem)

with col_chat:
    st.subheader("Chat with Copilot")

    # Show short summaries of memory for transparency
    with st.expander("🧠 Memory (read-only)"):
        st.markdown("**Recent Errors:**")
        st.code(_summarize_errors(st.session_state.persistent_memory), language="")
        st.markdown("**Style Notes:** ")
        st.code(_style_summary(st.session_state.persistent_memory), language="")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render chat history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_msg = st.chat_input("Ask about your code, errors, or request improvements…")

    if user_msg:
        st.session_state.messages.append({"role": "user", "content": user_msg})

        # Build a memory-aware prefix for the agent
        mem_summary = _summarize_errors(st.session_state.persistent_memory, limit=6)
        style_summary = _style_summary(st.session_state.persistent_memory)

        prefix = (
            "Context for you (do not repeat verbatim):\n"
            f"Prior frequent errors (most recent first):\n{mem_summary}\n\n"
            f"Observed coding style: {style_summary}\n\n"
            "If the user pasted code in the right pane, consider it as current context.\n"
            "When giving fixes, show minimal diffs or short edited snippets."
        )

        # Inject system context once per turn by prepending to the question
        full_query = prefix + "\n\nUser: " + user_msg

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    # Add a lightweight system nudge before calling the agent
                    agent.memory.chat_memory.add_message(system_context)
                    response = agent.run(full_query)
                except Exception as e:
                    response = f"Agent error: {e}"
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.caption(
    "Tip: Use the **Run Code** pane to reproduce errors. The agent reads your recent "
    "errors and style to tailor suggestions, and it can also use web_search when needed."
)

