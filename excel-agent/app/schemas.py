from __future__ import annotations
from pydantic import BaseModel
from typing import List, Dict, Optional

class SheetQuickSummary(BaseModel):
    name: str
    n_rows: int
    n_cols: int
    columns: List[str]

class SheetAIAnalysis(BaseModel):
    sheet: str
    analysis: str

class FileMetadata(BaseModel):
    filename: str
    size_bytes: int
    mime_type: Optional[str] = None

class ControllerSummary(BaseModel):
    metadata: FileMetadata
    per_sheet: List[SheetQuickSummary]

class WorkbookAIOverview(BaseModel):
    purpose: str
    key_entities: List[str] = []
    key_metrics: List[str] = []
    time_ranges: List[str] = []
    data_quality_notes: List[str] = []
    suggested_questions: List[str] = []

class AnalyzeResponse(BaseModel):
    controller_summary: ControllerSummary
    per_sheet_ai: List[SheetAIAnalysis]
    workbook_ai: WorkbookAIOverview

class AgentPayload(BaseModel):
    file_path: str
    controller_summary: ControllerSummary
    workbook_ai: WorkbookAIOverview
