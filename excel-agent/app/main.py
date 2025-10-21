from __future__ import annotations
import os, tempfile, shutil, time, uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from .controller import summarize_workbook
from .schemas import AnalyzeResponse

app = FastAPI(title="Excel Agent • Sheets + Workbook Overview")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(query: str = Form(""), file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".xlsx", ".xlsm", ".xls")):
        raise HTTPException(status_code=400, detail="Please upload an Excel file (.xlsx/.xlsm/.xls)")

    # שמירה זמנית
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # יצירת שם ייחודי בתיקיית data/
    base_name = os.path.basename(file.filename)
    safe_name = base_name.replace(" ", "_")
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    final_name = f"{timestamp}_{unique_id}_{safe_name}"
    final_path = os.path.join(DATA_DIR, final_name)
    shutil.copyfile(tmp_path, final_path)

    try:
        controller_summary, sheet_ai, workbook_ai, agent_payload = summarize_workbook(final_path)
        controller_summary.metadata.filename = final_name
        agent_payload.file_path = final_path

        resp = AnalyzeResponse(
            controller_summary=controller_summary,
            per_sheet_ai=sheet_ai,
            workbook_ai=workbook_ai,
        )
        return JSONResponse(content=resp.model_dump())
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
