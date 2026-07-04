from fastmcp import FastMCP
import torch
from transformers import pipeline
import re
import subprocess, sys, tempfile, ast
from pathlib import Path

MATH_PATTERNS = [
r"[\d]+\s*[\+\-\*\/\%\^]+\s*[\d]+",  # e.g. 1234 * 5678
    r"divisible",
    r"\bprime\b",
    r"\bfactorial\b",
    r"\bsqrt\b",
    r"\bcalculate\b",
    r"\bcompute\b",
    r"\bsolve\b",
    r"\brun\b.*\bcode\b",
    r"\bwrite\b.*\bpython\b",
    r"what is\s+[\d]"
]
FORBIDDEN = {"os","sys","subprocess","socket","shutil","requests","urllib"}
mcp = FastMCP("ChatBot")


@mcp.tool()
def execute_python_code(code: str) -> dict:
    """The single MCP tool. Runs any Python code safely."""
    # AST security check
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"success": False, "stdout": "", "stderr": str(e)}

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN:
                    return {"success": False, "stdout": "",
                            "stderr": f"Import '{alias.name}' is not allowed"}

    # Run in isolated subprocess with timeout
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "run.py"
        script.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                capture_output=True, text=True, timeout=5
            )
            return {
                "success": proc.returncode == 0,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip()
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "", "stderr": "Timeout"}

def needs_computation(query:str)->bool:
    q=query.lower()
    return(re.searchable(p,q) for p in MATH_PATTERNS)

def build_code_from_query(query:str)->str:
    q=query.lower().strip()

    # Divisibility
    m = re.search(r"is\s+(\d+)\s+divisible\s+by\s+(\d+)", q)
    if m:
        return f"print({m.group(1)} % {m.group(2)} == 0)"

    # Primality
    m = re.search(r"is\s+(\d+)\s+(a\s+)?prime", q)
    if m:
        n = int(m.group(1))
        return f"n={n}\nprint(n>1 and all(n%i!=0 for i in range(2,int(n**0.5)+1)))"

    # Raw arithmetic expression in the query
    m = re.search(r"([\d\(\)][[\d\s\+\-\*\/\%\(\)\.]+)", q)
    if m:
        expr = m.group(1).strip()
        if any(op in expr for op in ['+','-','*','/',',']):
            return f"print({expr})"

    # Factorial
    m = re.search(r"factorial\s+of\s+(\d+)", q) or re.search(r"(\d+)\s*!", q)
    if m:
        return f"import math\nprint(math.factorial({m.group(1)}))"

    # Fallback: ask distilgpt2 to draft code and hope for the best
    prompt = f"# Python code to solve: {query}\nprint("
    raw = pipe(prompt, max_new_tokens=60)[0]["generated_text"]
    return raw.split("# Python")[1] if "# Python" in raw else f"# Could not parse: {query}"

# Initialize FastMCP


# Load a lightweight model using a high-level pipeline for less code
# This replaces manual tokenizer and model loading
pipe = pipeline(
    "text-generation",
    model="AdaptLLM/finance-chat",
    device=0 if torch.cuda.is_available() else -1
)

# In-memory memory (MCP Context)
conversations = {}


@mcp.tool()
def chat(user_input: str, session_id: str="user_123") -> str:
    """Chat with the AI. Remembers up to 5 turns of conversation per session."""

    # Simple Context Management
    history = conversations.get(session_id, [])
    history.append(f"User: {user_input}")

    # Keep last 5 exchanges (MCP Logic)
    context = "\n".join(history[-5:]) + "\nAI:"

    if needs_computation(user_input):
        code = build_code_from_query(user_input)
        print(f"\n[Tool call → execute_python_code]\n{code}\n")

        result = execute_python_code(code)
        if result["success"]:
            return f"Result: {result['stdout']}"
        else:
            return f"Error: {result['stderr']}"
    else:
        # Pure language task — delegate to distilgpt2
        out = pipe(user_input, max_new_tokens=80, pad_token_id=50256)[0]["generated_text"]


    #     # Generate response
    # result = pipe(context, max_new_tokens=250, pad_token_id=50256)
    # ai_response = result[0]['generated_text'].split("AI:")[-1].strip()

    # Save to memory
    history.append(f"AI: {out}")
    conversations[session_id] = history

    return out

def compute():
    pass

if __name__ == "__main__":
    tests = [
        "Solve the following (1234*5678)*910",
        "(1234 * 5678) / 910",
        "Is 2025 divisible by 5?",
        "Is 97 a prime number?",
        "What is the factorial of 10?",
    ]
    for q in tests:
        print(f"Q: {q}")
        print(f"A: {chat(q)}\n")
    mcp.run()