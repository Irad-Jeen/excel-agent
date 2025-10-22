from __future__ import annotations
import os
from typing import Any
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

from .schemas import ControllerSummary

# ייבוא כל הפונקציות מקובץ הכלים
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
# Create ReAct Agent
# =========================
def create_excel_agent() -> AgentExecutor:
    llm = _make_llm()

    tools = [
        # Smart summaries
        Tool(name="SummarizeWorkbook", func=summarize_workbook_tool,
             description="Summarize workbook structure and quick insights. INPUT: {\"file_path\":\"/path/to/file.xlsx\"}."),
        Tool(name="SummarizeSheet", func=summarize_sheet_tool,
             description="Smart sheet summary: columns, totals detection, YOY, numeric stats, quality, highlights. INPUT: {\"file_path\":\"/path/to/file.xlsx\",\"sheet_name\":\"<name>\"}."),

        # Utilities
        Tool(name="ListSheets",        func=list_sheets_tool,        description="List sheet names. INPUT: {\"file_path\":\"...\"}."),
        Tool(name="SheetColumns",      func=sheet_columns_tool,      description="Return headers. INPUT: {\"file_path\":\"...\",\"sheet_name\":\"...\"}."),
        Tool(name="SheetPreview",      func=sheet_preview_tool,      description="First N rows. INPUT: {\"file_path\":\"...\",\"sheet_name\":\"...\",\"n\":10}."),
        Tool(name="DetectYearColumns", func=detect_year_columns_tool,description="Detect year columns. INPUT: {\"file_path\":\"...\",\"sheet_name\":\"...\"}."),
        Tool(name="FindRows",          func=find_rows_tool,          description="Find rows by label regex. INPUT: {\"file_path\":\"...\",\"sheet_name\":\"...\",\"label_columns\":[...],\"query\":\"...\"}."),
        Tool(name="TotalsRow",         func=totals_row_tool,         description="Locate 'Total' row and return values per year. INPUT: {...}."),
        Tool(name="YoYForLabel",       func=yoy_for_label_tool,      description="Compute YoY for a label row. INPUT: {...}."),
        Tool(name="QualityReport",     func=quality_report_tool,     description="Nulls and numeric parse rate. INPUT: {...}."),

        # Computational
        Tool(name="ColumnStats",       func=column_stats_tool,       description="Numeric stats per column. INPUT: {\"file_path\":\"...\",\"sheet_name\":\"...\",\"columns\":[...]}."),
        Tool(name="ComputeAggregate",  func=compute_aggregate_tool,  description="Group & aggregate. INPUT: {\"file_path\":\"...\",\"sheet_name\":\"...\",\"group_by\":[...],\"values\":[...],\"agg\":\"sum|avg|min|max|median|count\"}."),
        Tool(name="ComputeRatio",      func=compute_ratio_tool,      description="Ratio of columns or totals by label. INPUT: {...}."),
        Tool(name="YoYTable",          func=yoy_table_tool,          description="YoY for many labels. INPUT: {\"file_path\":\"...\",\"sheet_name\":\"...\",\"label_columns\":[...],\"label_regex\":\"...\"}."),
        Tool(name="TopNChanges",       func=topn_changes_tool,       description="Top-N YoY deltas. INPUT: {\"file_path\":\"...\",\"sheet_name\":\"...\",\"n\":10}."),
        Tool(name="PivotMini",         func=pivot_mini_tool,         description="Minimal pivot. INPUT: {\"file_path\":\"...\",\"sheet_name\":\"...\",\"index\":[...],\"columns\":[...],\"values\":[...],\"agg\":\"sum\"}."),
        Tool(name="FilterRows",        func=filter_rows_tool,        description="Generic filtering. INPUT: {\"file_path\":\"...\",\"sheet_name\":\"...\",\"where\":[...],\"select\":[...],\"limit\":50}."),
    ]

    template = """You are an expert Excel analyst. Use tools when the query requires file-specific data.

TOOLS:
{tools}

Call tools with EXACTLY this format:
Thought: what you think you need to do
Action: the tool name (one of: {tool_names})
Action Input: a single JSON string with the tool arguments
Observation: the tool result (JSON string)
... (repeat Thought/Action/Action Input/Observation as needed)
Final Answer: a concise answer. If numeric results are central, include a short JSON object inline (no code fences).

IMPORTANT:
- Tools expect a JSON string as Action Input. Do NOT pass Python dicts. Do NOT pass plain text.
- Prefer these flows:
  • ListSheets -> SheetColumns/SheetPreview
  • DetectYearColumns -> TotalsRow / YoYForLabel
  • For analysis: YoYTable / TopNChanges / ComputeAggregate / ComputeRatio / PivotMini
  • Use QualityReport when data issues are suspected.
- Never use markdown code fences in your Final Answer.
- If a tool fails or data is insufficient, explain clearly and suggest the next best tool.

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
        handle_parsing_errors=True,
        # return_intermediate_steps=True,
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

When you need details, use the tools with JSON input as documented.
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
