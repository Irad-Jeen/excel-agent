from __future__ import annotations
import os
import pandas as pd
from fastapi import HTTPException

from .schemas import (
    ControllerSummary, FileMetadata, SheetQuickSummary,
    SheetAIAnalysis, WorkbookAIOverview, AgentPayload
)
from .utils import get_file_metadata
from .xl_readers import load_workbook
from .llm_client import analyze_sheet_with_ai, analyze_workbook_overview
from .agent_service import run_excel_agent


# ----------------------------------------------------------
# 🧠 פונקציה ראשית – סיכום הקובץ והגיליונות
# ----------------------------------------------------------

def summarize_workbook(file_path: str):
    size, mime = get_file_metadata(file_path)
    xls = load_workbook(file_path)

    per_sheet: list[SheetQuickSummary] = []
    ai_results: list[SheetAIAnalysis] = []
    workbook_previews: dict[str, str] = {}

    for name in xls.sheet_names:
        df = xls.parse(name)
        n_rows, n_cols = df.shape
        columns = [str(c) for c in df.columns.tolist()]

        per_sheet.append(SheetQuickSummary(
            name=name, n_rows=n_rows, n_cols=n_cols, columns=columns
        ))

        preview_csv = df.head(10).to_csv(index=False)
        workbook_previews[name] = (
            f"Columns: {', '.join(columns[:30])}\n"
            f"Preview CSV:\n{preview_csv}"
        )

        analysis = analyze_sheet_with_ai(name, preview_csv)
        ai_results.append(SheetAIAnalysis(sheet=name, analysis=analysis))

    overview = analyze_workbook_overview(workbook_previews)
    workbook_ai = WorkbookAIOverview(
        purpose=overview.get("purpose",""),
        key_entities=overview.get("key_entities",[]),
        key_metrics=overview.get("key_metrics",[]),
        time_ranges=overview.get("time_ranges",[]),
        data_quality_notes=overview.get("data_quality_notes",[]),
        suggested_questions=overview.get("suggested_questions",[]),
    )

    controller_summary = ControllerSummary(
        metadata=FileMetadata(filename=os.path.basename(file_path), size_bytes=size, mime_type=mime),
        per_sheet=per_sheet
    )
    agent_payload = AgentPayload(
        file_path=file_path,
        controller_summary=controller_summary,
        workbook_ai=workbook_ai,
    )
    return controller_summary, ai_results, workbook_ai, agent_payload


# ----------------------------------------------------------
# 🤖 פונקציה חדשה – הרצת הסוכן על הקובץ
# ----------------------------------------------------------

def run_agent_on_excel(file_path: str, query: str):
    """
    מפעיל את סוכן ה-AI (LangChain ReAct) על קובץ Excel.
    מקבל שאילתה חופשית ומחזיר תשובת AI על סמך הנתונים.
    """
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    controller_summary, _, _, _ = summarize_workbook(file_path)

    # מפעיל את ה-Agent עם הנתיב והסיכום
    try:
        answer = run_excel_agent(file_path, controller_summary, query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {e}")

    return {"query": query, "answer": answer}
