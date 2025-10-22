# app/controller.py
from __future__ import annotations
import os, json, uuid, base64, re
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Body

from .xl_readers import load_workbook
from .agent_service import run_excel_agent, run_text_agent
from .schemas import (
    ControllerSummary,
    ControllerSheetSummary,
    ControllerMetadata,
    ChatPayload,
)

router = APIRouter()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def _unique_filename(original_name: str) -> str:
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    rand = uuid.uuid4().hex[:8]
    base, ext = os.path.splitext(original_name)
    safe_base = base.replace(" ", "_")
    return f"{stamp}_{rand}_{safe_base}{ext}"

def _build_controller_summary(saved_path: str, original_name: str) -> ControllerSummary:
    xls = load_workbook(saved_path)
    per_sheet: List[ControllerSheetSummary] = []
    for name in xls.sheet_names:
        df = xls.parse(name)
        per_sheet.append(
            ControllerSheetSummary(
                name=name,
                n_rows=int(df.shape[0]),
                n_cols=int(df.shape[1]),
                columns=[str(c) for c in df.columns.tolist()],
            )
        )
    meta = ControllerMetadata(
        filename=os.path.basename(saved_path),
        size_bytes=os.path.getsize(saved_path),
        mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        original_name=original_name,
    )
    return ControllerSummary(metadata=meta, per_sheet=per_sheet)

# ---------- Heuristic: does the query REQUIRE the actual Excel data? ----------
_REQUIRE_DATA_PATTERNS = re.compile(
    r"(analy[sz]e|inspect|process|compute|calculate|find|list|show|extract|detect|"
    r"top|largest|variance|yoy|trend|pivot|vlookup|sum|avg|median|"
    r"compare|correlate|forecast|outlier|inconsistent|mismatch)",
    re.IGNORECASE,
)
_EXCEL_TARGET_PATTERNS = re.compile(
    r"(my\s+(excel|workbook|sheet|file)|this\s+(excel|workbook|sheet|file)|"
    r"the\s+workbook|the\s+sheet|in\s+the\s+file|attached\s+excel)",
    re.IGNORECASE,
)

def _query_requires_excel(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if re.search(r"(analy[sz]e|inspect|process)\s+(the|my|this)\s+(excel|workbook|sheet|file)", t, re.IGNORECASE):
        return True
    if _REQUIRE_DATA_PATTERNS.search(t) and _EXCEL_TARGET_PATTERNS.search(t):
        return True
    if re.match(r"^\s*(what|how|why|when|which)\b", t, re.IGNORECASE):
        return False
    return False

# ------------------------------ JSON-only: /chat/analyze ------------------------------
@router.post("/chat/analyze")
async def chat_analyze(payload: ChatPayload = Body(...)):
    """
    קלט יחיד: JSON עם messages.
    אם מצורף אקסל ב-Base64 -> ננתח. אחרת:
      - אם לא חובה אקסל -> נענה מיד (בעזרת Gemma).
      - אם חובה -> נחזיר בקשה לצרף קובץ.
    פלט: {"agent_answer": "..."} בלבד.
    """
    texts: List[str] = []
    excel_file_bytes: Optional[bytes] = None
    excel_original_name: Optional[str] = None

    for msg in payload.messages:
        for part in msg.content:
            if part.type == "text" and getattr(part, "text", None):
                texts.append(part.text.strip())
            elif part.type == "input_file" and getattr(part, "file_data", None):
                mt = (part.file_data.mime_type or "").lower()
                if "spreadsheetml" in mt and excel_file_bytes is None:
                    try:
                        excel_file_bytes = base64.b64decode(part.file_data.data)
                        excel_original_name = part.file_data.name or "uploaded.xlsx"
                    except Exception:
                        excel_file_bytes = None

    query = " ".join([t for t in texts if t]).strip() or "Analyze the Excel workbook."

    if excel_file_bytes:
        try:
            original_name = excel_original_name or "uploaded.xlsx"
            save_name = _unique_filename(original_name)
            save_path = os.path.join(DATA_DIR, save_name)
            with open(save_path, "wb") as f:
                f.write(excel_file_bytes)

            controller_summary = _build_controller_summary(save_path, original_name)
            agent_answer = run_excel_agent(
                file_path=save_path,
                controller_summary=controller_summary,
                query=query,
            )
            return {"agent_answer": agent_answer}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    if _query_requires_excel(query):
        return {
            "agent_answer": (
                "This request requires the actual Excel data to compute or verify results. "
                "Please attach a .xlsx file (Base64 in messages) so I can analyze the workbook."
            )
        }

    try:
        direct_answer = run_text_agent(query)
        return {"agent_answer": direct_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
