from __future__ import annotations
import os, json
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

def _make_llm():
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", "google/gemma-3-12b-it")
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

# ניתוח גיליון בודד
SHEET_SYSTEM = (
    "You analyze Excel sheets. Be concise and specific. "
    "Point out columns, row counts, obvious quality issues, and interesting patterns."
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
    return llm.invoke(messages).content.strip()

# ניתוח כל הקובץ
WORKBOOK_SYSTEM = (
    "You are a data analyst. You receive previews for multiple Excel sheets and must infer what the workbook is about. "
    "Be concrete and concise."
)

def analyze_workbook_overview(summaries: Dict[str, str]) -> Dict[str, List[str] | str]:
    llm = _make_llm()
    joined = "\n---\n".join([f"Sheet: {k}\n{v}" for k, v in summaries.items()])
    messages = [
        SystemMessage(content=WORKBOOK_SYSTEM),
        HumanMessage(content=(
            "Below are short previews of each sheet in an Excel workbook.\n"
            "Infer the overall purpose, key entities, key metrics, time ranges, obvious data quality issues,\n"
            "and propose 5–8 follow-up questions/analyses the user might ask.\n\n"
            f"{joined}\n\n"
            "Return a JSON object with keys: purpose (string), key_entities (string[]), key_metrics (string[]), "
            "time_ranges (string[]), data_quality_notes (string[]), suggested_questions (string[]). "
            "Do not include extra text."
        )),
    ]
    resp = llm.invoke(messages).content.strip()
    try:
        data = json.loads(resp)
        for k in ["purpose","key_entities","key_metrics","time_ranges","data_quality_notes","suggested_questions"]:
            if k not in data:
                data[k] = [] if k != "purpose" else ""
        return data
    except Exception:
        return {
            "purpose": resp,
            "key_entities": [],
            "key_metrics": [],
            "time_ranges": [],
            "data_quality_notes": [],
            "suggested_questions": [],
        }
