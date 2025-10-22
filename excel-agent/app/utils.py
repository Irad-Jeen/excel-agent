# app/utils.py
import os, mimetypes
from typing import Tuple

def get_file_metadata(path: str) -> Tuple[int, str]:
    size = os.path.getsize(path)
    mime, _ = mimetypes.guess_type(path)
    return size, mime or "application/octet-stream"
