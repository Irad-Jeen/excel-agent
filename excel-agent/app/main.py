from __future__ import annotations
import os, tempfile, shutil, time, uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse

from .controller import summarize_workbook, run_agent_on_excel
from .schemas import AnalyzeResponse
from .agent_service import run_excel_agent

app = FastAPI(title="Excel Agent • Unified Endpoint")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

@app.post("/analyze")
async def analyze(
    query: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Unified endpoint:
    1. Accepts Excel file and user query
    2. Summarizes the workbook and its sheets
    3. Runs the Excel Agent (Gemma 3 12B via OpenRouter)
    4. Returns combined structured JSON response
    """
    if not file.filename.lower().endswith((".xlsx", ".xlsm", ".xls")):
        raise HTTPException(status_code=400, detail="Please upload a valid Excel file (.xlsx/.xlsm/.xls)")

    # --- שלב 1: שמירה זמנית ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # --- שלב 2: יצירת שם ייחודי בתיקיית data ---
    base_name = os.path.basename(file.filename)
    safe_name = base_name.replace(" ", "_")
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    final_name = f"{timestamp}_{unique_id}_{safe_name}"
    final_path = os.path.join(DATA_DIR, final_name)
    shutil.copyfile(tmp_path, final_path)

    try:
        # --- שלב 3: סיכום טכני וניתוח ראשוני ---
        controller_summary, per_sheet_ai, workbook_ai, _ = summarize_workbook(final_path)

        # --- שלב 4: הפעלת הסוכן על הקובץ והשאילתה ---
        agent_answer = run_excel_agent(final_path, controller_summary, query)

        # --- שלב 5: תשובה מאוחדת ---
        response = {
            "file_name": final_name,
            "query": query,
            "controller_summary": controller_summary.model_dump(),
            "per_sheet_ai": [a.model_dump() for a in per_sheet_ai],
            "workbook_ai": workbook_ai.model_dump(),
            "agent_answer": agent_answer,
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
