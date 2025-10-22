from __future__ import annotations
import os
import json
from typing import Any
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from .xl_readers import load_workbook
from .schemas import ControllerSummary


# =========================
# LLM: Gemma via OpenRouter
# =========================
def _make_llm():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("❌ Missing OPENROUTER_API_KEY in environment!")

    model = os.getenv("OPENROUTER_MODEL", "google/gemma-3-12b-it")
    if "gemma" not in model.lower():
        raise ValueError(f"❌ Invalid model: {model} — must be Gemma only!")

    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    headers = {
        "HTTP-Referer": os.getenv("HTTP_REFERER", "http://localhost:8000"),
        "X-Title": os.getenv("X_TITLE", "Excel-Agent"),
    }
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        openai_api_key=api_key,
        default_headers=headers,
        temperature=0.2,
        timeout=90,
    )


# =========================
# Tools (single string input)
# =========================
def _parse_json_input(input_str: str) -> dict:
    input_str = (input_str or "").strip()
    if not input_str:
        return {}
    try:
        return json.loads(input_str)
    except Exception as e:
        raise ValueError(f"Tool expected JSON string input, got: {input_str!r}. Error: {e}")

def summarize_sheet_tool(input_str: str) -> str:
    """
    Expect JSON: {"file_path": "...", "sheet_name": "..."}
    """
    try:
        data = _parse_json_input(input_str)
        file_path = data.get("file_path")
        sheet_name = data.get("sheet_name")
        if not file_path or not sheet_name:
            return "❌ Missing required keys. Provide JSON with keys: file_path, sheet_name."

        xls = load_workbook(file_path)
        if sheet_name not in xls.sheet_names:
            return f"❌ Sheet '{sheet_name}' not found. Available: {xls.sheet_names}"

        import pandas as pd
        df = xls.parse(sheet_name)
        preview = df.head(20).to_csv(index=False)
        return (
            f"📊 Sheet: {sheet_name}\n"
            f"Columns: {', '.join(map(str, df.columns))}\n"
            f"Rows: {len(df)}\n\nPreview (first 20 rows):\n{preview}"
        )
    except Exception as e:
        return f"❌ Error in SummarizeSheet: {e}"

def summarize_workbook_tool(input_str: str) -> str:
    """
    Expect JSON: {"file_path": "..."}
    """
    try:
        data = _parse_json_input(input_str)
        file_path = data.get("file_path")
        if not file_path:
            return "❌ Missing required key 'file_path' in JSON."

        xls = load_workbook(file_path)
        parts = []
        for name in xls.sheet_names:
            df = xls.parse(name)
            parts.append(
                f"Sheet: {name} | rows={len(df)} | cols={len(df.columns)} | "
                f"columns={', '.join(map(str, df.columns))}"
            )
        return "\n".join(parts)
    except Exception as e:
        return f"❌ Error in SummarizeWorkbook: {e}"


# =========================
# Create ReAct Agent
# =========================
def create_excel_agent() -> AgentExecutor:
    llm = _make_llm()

    tools = [
        Tool(
            name="SummarizeWorkbook",
            func=summarize_workbook_tool,
            description=(
                "Summarize the structure of an Excel workbook: list sheets, row/col counts, and columns. "
                "INPUT MUST BE JSON like: {\"file_path\":\"/path/to/file.xlsx\"}."
            ),
        ),
        Tool(
            name="SummarizeSheet",
            func=summarize_sheet_tool,
            description=(
                "Summarize a specific sheet in the Excel file (columns, rows, small preview). "
                "INPUT MUST BE JSON like: {\"file_path\":\"/path/to/file.xlsx\",\"sheet_name\":\"P&L Insurance YOY\"}."
            ),
        ),
    ]

    template = """You are an expert Excel analyst. You have tools you can call.

TOOLS:
{tools}

When you need to use a tool, follow EXACTLY this format:
Thought: what you think you need to do
Action: the tool name (one of: {tool_names})
Action Input: a single JSON string with the tool arguments
Observation: the tool result
... (repeat Thought/Action/Action Input/Observation as needed)
Final Answer: a concise, helpful answer to the user's question.

IMPORTANT:
- Tools expect a JSON string as Action Input. Do NOT pass Python dicts. Do NOT pass plain text.
- For SummarizeWorkbook, pass: {{ "file_path": "<path>" }}.
- For SummarizeSheet, pass: {{ "file_path": "<path>", "sheet_name": "<sheet>" }}.

Question:
{input}

{agent_scratchpad}"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor


# =========================
# Run Excel agent
# =========================
def run_excel_agent(file_path: str, controller_summary: ControllerSummary, query: str) -> str:
    agent = create_excel_agent()

    context = f"""You are analyzing an Excel workbook uploaded by the user.

Here is the controller summary (metadata and per-sheet structure):
{controller_summary.model_dump_json(indent=2)}

When you need details, use the tools with JSON input as documented.
File path to use: {file_path}
User Query: {query}"""

    try:
        result = agent.invoke({"input": context})
        return result.get("output", "❌ No output returned from agent.")
    except Exception as e:
        return f"❌ Agent failed to process query: {e}"


# =========================
# Run text-only agent (no Excel required)
# =========================
def run_text_agent(query: str) -> str:
    """
    Answers general questions that do NOT require reading an Excel file.
    Output is short (<= 120 words).
    """
    llm = _make_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a concise business analyst. "
             "If the user's question can be answered without reading a file, answer directly in under 120 words. "
             "If the question clearly requires specific data from a file, briefly state what data is needed."),
            ("user", "{q}"),
        ]
    )
    chain = prompt | llm
    msg = chain.invoke({"q": query})
    # ChatOpenAI via LangChain returns an AIMessage with .content
    return getattr(msg, "content", "").strip() or "Done."
