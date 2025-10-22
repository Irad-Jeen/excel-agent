# app/agent_service.py
from __future__ import annotations
import os, json, re, ast
from typing import Any, Dict, Callable, Union
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from .schemas import ControllerSummary

# ====== ייבוא הכלים ======
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
    describe_tools_tool,
)

# =========================
# LLM: Gemma via OpenRouter  (עם stop tokens נגד ``` וקוד-פנסינג)
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
    # ✅ stop מונע מהמודל לפתוח/לסגור ``` וליפול על parser
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        openai_api_key=api_key,
        default_headers=headers,
        temperature=0.0,
        timeout=90,
        stop=["```", "\n```", "```json", "```JSON"],
    )

# =========================
# Router Tool — Invoke (סלחן ל-JSON “מלוכלך”)
# =========================
def _normalize_tool_name(name: str) -> str:
    if not name:
        return ""
    t = str(name).strip().strip("`").strip('"').strip("'")
    if t.lower().startswith("action:"):
        t = t.split(":", 1)[-1].strip()
    return t

TOOL_ALIASES = {
    "describetools": "DescribeTools",
    "listsheets": "ListSheets",
    "sheetcolumns": "SheetColumns",
    "columns": "SheetColumns",
    "sheetpreview": "SheetPreview",
    "detectyearcolumns": "DetectYearColumns",
    "detecttables": "DetectTables",
    "findrows": "FindRows",
    "totalsrow": "TotalsRow",
    "yoyforlabel": "YoYForLabel",
    "qualityreport": "QualityReport",
    "detecttotalcolumns": "DetectTotalColumns",
    "detectsubtotalrows": "DetectSubtotalRows",
    "columnstats": "ColumnStats",
    "computeaggregate": "ComputeAggregate",
    "computeratio": "ComputeRatio",
    "yoytable": "YoYTable",
    "topnchanges": "TopNChanges",
    "pivotmini": "PivotMini",
    "filterrows": "FilterRows",
    "detectyear": "DetectYearColumns",
    "preview": "SheetPreview",
}

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
    "DescribeTools":          describe_tools_tool,
}

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

def _coerce_to_dict(input_data: Union[str, dict]) -> Dict[str, Any]:
    if isinstance(input_data, dict):
        return input_data
    s = (input_data or "").strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
    s = s.replace("\u200b", "").strip()
    m = _JSON_OBJECT_RE.search(s)
    if m:
        s = m.group(0).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        maybe = ast.literal_eval(s)
        if isinstance(maybe, dict):
            return maybe
    except Exception:
        pass
    try:
        s2 = re.sub(r"(?<!\\)'", '"', s)
        return json.loads(s2)
    except Exception as e:
        raise ValueError(f"Invoke: invalid JSON string input. {e}")

def _invoke_router(input_any: Union[str, Dict[str, Any]]) -> str:
    try:
        data = _coerce_to_dict(input_any)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    tool_name = _normalize_tool_name(str(data.get("tool", "")))
    tool_name = TOOL_ALIASES.get(tool_name.replace(" ", "").lower(), tool_name)
    tool_input = data.get("input")

    if not tool_name:
        return json.dumps({"error": "Invoke: missing 'tool'."}, ensure_ascii=False)
    if tool_name not in TOOL_REGISTRY:
        return json.dumps({
            "error": f"Invoke: unknown tool '{tool_name}'.",
            "hint": f"Use one of: {sorted(TOOL_REGISTRY.keys())}"
        }, ensure_ascii=False)

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

    tools = [
        Tool(
            name="Invoke",
            func=_invoke_router,
            description=(
                "Call any Excel tool via a router.\n"
                "INPUT: either a JSON object or a JSON string with fields {tool, input}.\n"
                "Format: {\"tool\":\"<One of: "
                + ", ".join(sorted(TOOL_REGISTRY.keys()))
                + ">\", \"input\": {<tool-args>}}\n"
                "Example (object): {\"tool\":\"SheetColumns\",\"input\":{\"file_path\":\"/path.xlsx\",\"sheet_name\":\"P&L Insurance YOY\"}}\n"
                "Example (string): \"{\\\"tool\\\":\\\"SheetColumns\\\",\\\"input\\\":{\\\"file_path\\\":\\\"/path.xlsx\\\",\\\"sheet_name\\\":\\\"P&L Insurance YOY\\\"}}\""
            ),
        )
    ]

    # ✅ פרומפט מוקשח — אין פנסינג, אין טקסט מסביב, 4 שורות בלבד
    template = """You are an expert Excel analyst. Use the single tool "Invoke" to run any operation.

AVAILABLE TOOLS (via Invoke):
{tools}

FORMAT (produce exactly these 4 blocks, no extras, no code fences, no markdown):
Thought: brief reasoning
Action: Invoke
Action Input: {{"tool":"<OneOf:{tool_names}>","input":{{"file_path":"{example_path}","sheet_name":"<SheetNameOrOtherArgs>"}}}}
Observation: (the tool JSON result will appear here; then continue if needed)

Rules:
- NEVER use backticks or code fences.
- NEVER include extra prose on the Action Input line; it must be a raw JSON object.
- If a tool fails, read its error JSON and try a corrective next step.
- Finish with: Final Answer: <concise answer to the user>.

Question:
{input}

{agent_scratchpad}"""

    example_path = "/path/to/workbook.xlsx"
    tool_names = ", ".join(sorted(TOOL_REGISTRY.keys()))

    prompt = PromptTemplate(
        template=template,
        input_variables=["input", "tools", "agent_scratchpad"],
        partial_variables={"example_path": example_path, "tool_names": tool_names},
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=8,                  # לא להסתחרר בלופים
        early_stopping_method="generate",  # אם נתקע, לייצר Final Answer
        handle_parsing_errors=(
            "Your last message did not follow the exact 4-block format.\n"
            "Produce ONLY:\n"
            "Thought: ...\nAction: Invoke\nAction Input: {\"tool\":\"<ToolName>\",\"input\":{...}}\nObservation: ...\n"
            "No backticks. No code fences. No extra prose around Action Input."
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
    from langchain_core.prompts import ChatPromptTemplate
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
