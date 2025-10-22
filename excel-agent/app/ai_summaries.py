# app/ai_summaries.py
from __future__ import annotations
import os, json
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

"""
Purpose
-------
LLM-based semantic summaries for the Excel Agent:
1) analyze_sheet_with_ai  -> short, bullet-style purpose for a single sheet
2) analyze_workbook_overview -> structured semantic summary for the whole workbook
   (purpose, key_entities, key_metrics, time_ranges, data_quality_notes, suggested_questions)

These functions are called from tools.py:
- SummarizeWorkbook -> analyze_workbook_overview (adds "semantic" object to result)
- ExplainSheetPurpose -> analyze_sheet_with_ai
"""

def _make_llm():
    """
    Construct the ChatOpenAI client via OpenRouter (Gemma only).
    Uses the same env vars as the rest of the project.
    """
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", "google/gemma-3-12b-it")

    # Fail fast like the rest of the codebase, so misconfig is obvious
    if not api_key:
        raise ValueError("❌ Missing OPENROUTER_API_KEY in environment!")
    if "gemma" not in (model or "").lower():
        raise ValueError(f"❌ Invalid model: {model} — must be Gemma only!")

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
        timeout=60,
    )

# ---------- Single-sheet semantic summary ----------
SHEET_SYSTEM = (
    "You analyze Excel sheets. Be concise and specific. "
    "Given a small CSV preview, infer the sheet's purpose, main entities/metrics, and any obvious data issues. "
    "Return 3–6 bullet points. No markdown code fences."
)

def analyze_sheet_with_ai(sheet_name: str, preview_csv: str) -> str:
    llm = _make_llm()
    messages = [
        SystemMessage(content=SHEET_SYSTEM),
        HumanMessage(content=(
            f"Sheet name: {sheet_name}\n"
            f"Preview (first rows, CSV):\n{preview_csv}\n\n"
            "Give a short analysis (3–6 bullet points)."
        ))
    ]
    resp = llm.invoke(messages)
    return (getattr(resp, "content", "") or "").strip()

# ---------- Workbook-level semantic summary ----------
WORKBOOK_SYSTEM = (
    "You are a data analyst. You receive previews for multiple Excel sheets (CSV snippets). "
    "Infer what the overall workbook is about. Be concrete and concise."
)

def analyze_workbook_overview(summaries: Dict[str, str]) -> Dict[str, List[str] | str]:
    llm = _make_llm()
    joined = "\n---\n".join([f"Sheet: {k}\n{v}" for k, v in summaries.items()])
    messages = [
        SystemMessage(content=WORKBOOK_SYSTEM),
        HumanMessage(content=(
            "Below are short CSV previews of each sheet in an Excel workbook.\n"
            "Infer the overall purpose, key entities, key metrics, time ranges, obvious data quality issues,\n"
            "and propose 5–8 follow-up questions/analyses the user might ask.\n\n"
            f"{joined}\n\n"
            "Return a JSON object with keys: purpose (string), key_entities (string[]), key_metrics (string[]), "
            "time_ranges (string[]), data_quality_notes (string[]), suggested_questions (string[]). "
            "Do not include extra text."
        )),
    ]
    resp = llm.invoke(messages)
    text = (getattr(resp, "content", "") or "").strip()
    try:
        data = json.loads(text)
        for k in ["purpose","key_entities","key_metrics","time_ranges","data_quality_notes","suggested_questions"]:
            if k not in data:
                data[k] = [] if k != "purpose" else ""
        data["purpose"] = str(data.get("purpose", ""))
        for key in ["key_entities","key_metrics","time_ranges","data_quality_notes","suggested_questions"]:
            if not isinstance(data.get(key), list):
                data[key] = []
            data[key] = [str(x) for x in data[key]]
        return data
    except Exception:
        return {
            "purpose": text,
            "key_entities": [],
            "key_metrics": [],
            "time_ranges": [],
            "data_quality_notes": [],
            "suggested_questions": [],
        }
