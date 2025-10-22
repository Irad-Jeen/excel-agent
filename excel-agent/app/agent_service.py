# app/agent_service.py
from __future__ import annotations
import os, json
from typing import Any, Dict, Callable
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

from .schemas import ControllerSummary

# כלים מקוריים
from .tools import (
    summarize_workbook_tool,
    summarize_sheet_tool,
    list_sheets_tool,
    sheet_columns_tool,
    sheet_preview_tool,
    detect_year_columns_tool,
    find_rows_tool,
    totals_row_tool,
    yoy_for_label_tool,
    quality_report_tool,
    column_stats_tool,
    compute_aggregate_tool,
    compute_ratio_tool,
    yoy_table_tool,
    topn_changes_tool,
    pivot_mini_tool,
    filter_rows_tool,
    detect_total_columns_tool,
    detect_subtotal_rows_tool,
    detect_tables_tool,
    explain_sheet_purpose_tool,
)

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
# Router Tool — Invoke
# =========================
def _normalize_tool_name(name: str) -> str:
    if not name:
        return ""
    t = name.strip().strip("`").strip('"').strip("'")
    if t.lower().startswith("action:"):
        t = t.split(":", 1)[-1].strip()
    return t

# מיפוי שם->פונקציה (שמור על סנכרון עם הכלים)
TOOL_REGISTRY: Dict[str, Callable[[str], str]] = {
    "SummarizeWorkbook":      summarize_workbook_tool,
    "SummarizeSheet":         summarize_sheet_tool,
    "ExplainSheetPurpose":    explain_sheet_purpose_tool,
    "ListSheets":             list_sheets_tool,
    "SheetColumns":           sheet_columns_tool,
    "SheetPreview":           sheet_preview_tool,
    "DetectYearColumns":      detect_year_columns_tool,
    "DetectTables":           detect_tables_tool,
    "FindRows":               find_rows_tool,
    "TotalsRow":              totals_row_tool,
    "YoYForLabel":            yoy_for_label_tool,
    "QualityReport":          quality_report_tool,
    "DetectTotalColumns":     detect_total_columns_tool,
    "DetectSubtotalRows":     detect_subtotal_rows_tool,
    "ColumnStats":            column_stats_tool,
    "ComputeAggregate":       compute_aggregate_tool,
    "ComputeRatio":           compute_ratio_tool,
    "YoYTable":               yoy_table_tool,
    "TopNChanges":            topn_changes_tool,
    "PivotMini":              pivot_mini_tool,
    "FilterRows":             filter_rows_tool,
}

def _invoke_router(input_str: str) -> str:
    """
    קלט צפוי (מחרוזת JSON):
    {"tool":"SheetColumns","input":{"file_path":"...","sheet_name":"..."}}
    """
    try:
        data = json.loads((input_str or "").strip() or "{}")
    except Exception as e:
        return json.dumps({"error": f"Invoke: invalid JSON string input. {e}"}, ensure_ascii=False)

    tool_name = _normalize_tool_name(str(data.get("tool", "")))
    tool_input = data.get("input")

    if not tool_name:
        return json.dumps({"error": "Invoke: missing 'tool'."}, ensure_ascii=False)
    if tool_name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Invoke: unknown tool '{tool_name}'."}, ensure_ascii=False)

    # הכלים שלנו מצפים תמיד למחרוזת JSON (string), לא ל-dict
    if isinstance(tool_input, dict):
        tool_input = json.dumps(tool_input, ensure_ascii=False)
    elif tool_input is None:
        tool_input = "{}"
    elif not isinstance(tool_input, str):
        return json.dumps({"error": "Invoke: 'input' must be a JSON string or object."}, ensure_ascii=False)

    try:
        fn = TOOL_REGISTRY[tool_name]
        return fn(tool_input)
    except Exception as e:
        return json.dumps({"error": f"Invoke: tool '{tool_name}' failed: {e}"}, ensure_ascii=False)

# =========================
# Create ReAct Agent (Invoke-only)
# =========================
def create_excel_agent() -> AgentExecutor:
    llm = _make_llm()

    # נחשוף למודל רק כלי אחד — Invoke — והוא יעשה dispatch לכל השאר.
    tools = [
        Tool(
            name="Invoke",
            func=_invoke_router,
            description=(
                "Call any Excel tool via a router.\n"
                "INPUT (JSON string): {\"tool\":\"<One of: "
                + ", ".join(sorted(TOOL_REGISTRY.keys()))
                + ">\", \"input\": {<tool-args>}}\n"
                "Example: {\"tool\":\"SheetColumns\",\"input\":{\"file_path\":\"/path.xlsx\",\"sheet_name\":\"P&L Insurance YOY\"}}"
            ),
        )
    ]

    template = """You are an expert Excel analyst. Use the single tool "Invoke" to run any operation.

AVAILABLE TOOLS (via Invoke):
{tools}

IMPORTANT CALL FORMAT (no backticks anywhere):
Thought: what you need to do
Action: Invoke
Action Input: a single JSON string like "{{\"tool\":\"SheetColumns\",\"input\":{{\"file_path\":\"/path.xlsx\",\"sheet_name\":\"P&L Insurance YOY\"}}}}"
Observation: the tool JSON result
... (iterate Thought/Action/Action Input/Observation)
Final Answer: concise answer; include a small inline JSON object if numeric results are central (no code fences).

Rules:
- Always use Action: Invoke (exactly). Never call other tool names directly.
- In Action Input, 'tool' must be one of: {tool_names}.
- 'input' may be a JSON object; the router will stringify it for the target tool.
- If a tool fails, read the error JSON and try a corrective next step (e.g., fix sheet_name).

Question:
{input}

{agent_scratchpad}"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=(
            "Your last message did not follow the tool-call format. "
            "Use ONLY:\nThought\nAction: Invoke\nAction Input: \"{\\\"tool\\\":\\\"<ToolName>\\\",\\\"input\\\":{...}}\""
        ),
    )
    return executor

# =========================
# Run Excel agent
# =========================
def run_excel_agent(file_path: str, controller_summary: ControllerSummary, query: str) -> str:
    agent = create_excel_agent()
    context = f"""You are analyzing an Excel workbook uploaded by the user.

Here is the controller summary (metadata and per-sheet structure):
{controller_summary.model_dump_json(indent=2)}

When you need details, use Invoke with JSON input as documented.
File path to use: {file_path}
User Query: {query}"""
    try:
        result = agent.invoke({"input": context}, handle_parsing_errors=True)
        return result.get("output", "❌ No output returned from agent.")
    except Exception as e:
        return f"❌ Agent failed to process query: {e}"

# =========================
# Run text-only agent
# =========================
def run_text_agent(query: str) -> str:
    llm = _make_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a concise business analyst. "
             "If the user's question can be answered without reading a file, answer directly in under 120 words. "
             "If the question clearly requires specific data from a file, briefly state what data is needed. "
             "Never use markdown code fences in your final output."),
            ("user", "{q}"),
        ]
    )
    chain = prompt | llm
    msg = chain.invoke({"q": query})
    return getattr(msg, "content", "").strip() or "Done."
