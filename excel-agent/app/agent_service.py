# app/agent_service.py
from __future__ import annotations
import os, json, re, ast
from typing import Any, Dict, Callable, Union, Optional, Set, List
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
    infer_sheet_relations_tool,
)

# =========================
# Per-request guardrails and dynamic tool profiles
# =========================
# These module-level variables are set just-in-time by run_excel_agent and
# cleared immediately after the agent invocation finishes.
_REQUEST_FILE_PATH: Optional[str] = None
_REQUEST_SHEET_LOCK: Optional[str] = None
_ALLOWED_TOOLS_CURRENT: Optional[Set[str]] = None

# Profiles: each request exposes ONLY these 5 tools to the agent
_PROFILE_TOOLSETS: Dict[str, Set[str]] = {
    "general_explore": {
        "ListSheets", "SummarizeSheet", "SheetPreview", "FindRows", "ComputeAggregate"
    },
    "time_series": {
        "DetectYearColumns", "YoYTable", "FindRows", "TotalsRow", "ComputeAggregate"
    },
    "quality_structure": {
        "SheetColumns", "QualityReport", "DetectTables", "DetectTotalColumns", "DetectSubtotalRows"
    },
    "transform_pivot": {
        "FilterRows", "ComputeAggregate", "PivotMini", "TotalsRow", "SheetPreview"
    },
    "relationships": {
        "InferSheetRelations", "ListSheets", "SheetColumns", "FindRows", "DetectTables"
    },
}

def _select_profile(query: str) -> Set[str]:
    t = (query or "").lower()
    # Prefer totals-aware toolkit when user asks for sum/total
    if re.search(r"\b(total|totals|grand\s*total|sum)\b", t):
        return _PROFILE_TOOLSETS["time_series"]
    # Relationship intent: map/link/relate/structure/schema/dimension/reference between sheets
    if re.search(r"\b(relationship|relat(e|ion)|map(ping)?|link(age)?|structure|schema|dimension|reference)\b", t):
        return _PROFILE_TOOLSETS["relationships"]
    if re.search(r"\b(yoy|trend|growth|delta|202[0-9])\b", t):
        return _PROFILE_TOOLSETS["time_series"]
    if re.search(r"\b(missing|null|quality|schema|structure|subtotal|total column|tables?)\b", t):
        return _PROFILE_TOOLSETS["quality_structure"]
    if re.search(r"\b(pivot|group|aggregate|mean|median|top\s*n|filter)\b", t):
        return _PROFILE_TOOLSETS["transform_pivot"]
    return _PROFILE_TOOLSETS["general_explore"]

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
    "InferSheetRelations":    infer_sheet_relations_tool,
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

    # Enforce per-request allowed tools (dynamic 5-tool profile)
    allowed: Set[str] = _ALLOWED_TOOLS_CURRENT or set(TOOL_REGISTRY.keys())
    if tool_name and tool_name not in allowed:
        return json.dumps({
            "error": f"Invoke: tool '{tool_name}' is not allowed in this request.",
            "allowed": sorted(list(allowed)),
        }, ensure_ascii=False)

    if not tool_name:
        return json.dumps({"error": "Invoke: missing 'tool'."}, ensure_ascii=False)
    if tool_name not in TOOL_REGISTRY:
        return json.dumps({
            "error": f"Invoke: unknown tool '{tool_name}'.",
            "hint": f"Use one of: {sorted(TOOL_REGISTRY.keys())}"
        }, ensure_ascii=False)

    # Sanitize and lock inputs to the current request (file path, optional sheet)
    try:
        input_dict: Dict[str, Any]
        if isinstance(tool_input, dict):
            input_dict = tool_input
        elif tool_input is None:
            input_dict = {}
        elif isinstance(tool_input, str):
            input_dict = _coerce_to_dict(tool_input)
        else:
            return json.dumps({"error": "Invoke: 'input' must be a JSON string or object."}, ensure_ascii=False)

        if _REQUEST_FILE_PATH:
            input_dict["file_path"] = _REQUEST_FILE_PATH
        if _REQUEST_SHEET_LOCK is not None:
            # Enforce sheet lock if set; auto-inject if missing
            sheet = input_dict.get("sheet_name")
            if sheet is None:
                input_dict["sheet_name"] = _REQUEST_SHEET_LOCK
            elif str(sheet) != str(_REQUEST_SHEET_LOCK):
                return json.dumps({
                    "error": "Invoke: sheet_name is locked for this request.",
                    "required_sheet_name": _REQUEST_SHEET_LOCK,
                }, ensure_ascii=False)

        tool_input = json.dumps(input_dict, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Invoke: failed to sanitize input: {e}"}, ensure_ascii=False)

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
                "Only the 5 allowed tools for this request are available.\n"
            ),
        )
    ]

    # ✅ פרומפט מוקשח — אין פנסינג, אין טקסט מסביב, 4 שורות בלבד
    template = """You are an expert Excel analyst. Use the single tool "Invoke" to run operations.

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
- Use at most 5 Invoke calls total (prefer 2–4). If still uncertain, finalize.
- Operate only on the current workbook and, if specified, only the locked sheet.
- Before your first Action, include a one-line Tool Plan: name up to 3 candidate tools with 1-line reasons and pick 1.
- Totals-first: if the user asks for totals, try TotalsRow (or YoYForLabel) before ComputeAggregate.
 - Totals-first: if the user asks for totals or sums, try TotalsRow (or YoYForLabel) before ComputeAggregate. If ComputeAggregate is used, prefer existing total rows when present.
 - Totals-first: if the user asks for totals or sums, try TotalsRow (or YoYForLabel) before ComputeAggregate. When multiple totals exist (e.g., Revenues vs Expenses), choose the one whose label best matches the user’s target (e.g., contains "expenses"). If not specified, prefer Expenses over Revenues.
 - If TotalsRow returns null on a sheet, compute the total: use ComputeAggregate with detected year columns (e.g., year=2024) and sum across values; return both per-column sums and total_sum.
 - If TotalsRow returns null on a sheet, compute the total: use ComputeAggregate with detected year columns (e.g., year=2024) and, when appropriate, restrict rows by label_regex (e.g., (?i)\bExpenses\b) derived from the user intent or sheet structure. Return both per-column sums and total_sum.
 - If TotalsRow returns null on a sheet, compute the total: use ComputeAggregate with detected year columns (e.g., year=2024), prefer monthly columns, and restrict rows by target intent (e.g., target:"expenses") or label_regex. Return both per-column sums and total_sum.
- Do not use ComputeRatio for sums; it is for ratios only.
- If router returns a misuse hint, do exactly one corrective retry with the suggested tool and stay within 5 calls.
 - If a tool requires specific parameter names (e.g., ComputeAggregate expects 'values'), correct the parameter names and retry once.
 - For sheet relationships: first use InferSheetRelations to propose column mappings and sheet roles; then optionally verify with SheetColumns/FindRows/DetectTables.
- If a total is already present in the sheet (TotalsRow returns it), do not recompute via ComputeAggregate unless TotalsRow fails or is absent.
- Finish with: Final Answer: <concise answer to the user>.

Question:
{input}

{agent_scratchpad}"""

    example_path = "/path/to/workbook.xlsx"
    # Only show the 5 tools allowed for this request in the prompt
    allowed_for_prompt: List[str] = sorted(list(_ALLOWED_TOOLS_CURRENT or set(TOOL_REGISTRY.keys())))
    tool_names = ", ".join(allowed_for_prompt)

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
        max_iterations=6,                  # Up to 5 tool calls + final
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
    # Determine a reasonable sheet lock: single-sheet workbooks or explicit mention in query.
    # If the query clearly requests a cross-sheet comparison, do NOT set a sheet lock.
    sheet_lock: Optional[str] = None
    try:
        sheet_names = [s.name for s in controller_summary.per_sheet]
        q = (query or "").strip().lower()
        # Detect cross-sheet intent: mentions of 2+ known sheets or compare-like phrasing
        mentions = [name for name in sheet_names if name.lower() in q]
        cross_sheet = (len(mentions) >= 2) or bool(re.search(r"\b(compare|vs\.?|versus|between)\b", q))

        if cross_sheet:
            sheet_lock = None
        elif len(sheet_names) == 1:
            sheet_lock = sheet_names[0]
        else:
            # try exact (case-insensitive) match of a single sheet name mentioned in the query
            for name in sheet_names:
                if name.lower() in q:
                    sheet_lock = name
                    break
    except Exception:
        sheet_lock = None

    # Choose the dynamic 5-tool profile for this request
    allowed_tools = _select_profile(query)

    # Set per-request globals
    global _REQUEST_FILE_PATH, _REQUEST_SHEET_LOCK, _ALLOWED_TOOLS_CURRENT
    _REQUEST_FILE_PATH = file_path
    _REQUEST_SHEET_LOCK = sheet_lock
    _ALLOWED_TOOLS_CURRENT = allowed_tools

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
    finally:
        # Clear per-request globals to avoid cross-request leakage
        _REQUEST_FILE_PATH = None
        _REQUEST_SHEET_LOCK = None
        _ALLOWED_TOOLS_CURRENT = None

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
