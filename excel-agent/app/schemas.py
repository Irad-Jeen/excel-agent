# app/schemas.py
from __future__ import annotations
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field

# ---------------------------- Controller Summary ----------------------------
class ControllerMetadata(BaseModel):
    filename: str
    size_bytes: int
    mime_type: str
    original_name: Optional[str] = None

class ControllerSheetSummary(BaseModel):
    name: str
    n_rows: int
    n_cols: int
    columns: List[str]

class ControllerSummary(BaseModel):
    metadata: ControllerMetadata
    per_sheet: List[ControllerSheetSummary]

# ---------------------------- Chat Payload ----------------------------
class FileData(BaseModel):
    name: Optional[str] = None
    mime_type: Optional[str] = None
    data: str  # base64 string

class ContentItemText(BaseModel):
    type: Literal["text"]
    text: str

class ContentItemFile(BaseModel):
    type: Literal["input_file"]
    file_data: FileData

ContentItem = Union[ContentItemText, ContentItemFile]

class Message(BaseModel):
    role: Literal["user", "assistant", "system"] = "user"
    content: List[ContentItem]

class ChatPayload(BaseModel):
    messages: List[Message] = Field(default_factory=list)
