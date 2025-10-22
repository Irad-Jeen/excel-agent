# app/xl_readers.py
from __future__ import annotations
import pandas as pd

def load_workbook(path: str):
    """
    Returns an object with .sheet_names and .parse(name) like pandas.ExcelFile.
    Tries pandas first; falls back to pyexcel if needed (for .xls files).
    """
    try:
        return pd.ExcelFile(path, engine="openpyxl")
    except Exception:
        import pyexcel as p
        book = p.get_book(file_name=path)

        class _Shim:
            def __init__(self, book):
                self._book = book
                self.sheet_names = [s.name for s in book]

            def parse(self, name):
                s = self._book[name]
                data = list(s.rows())
                import pandas as pd
                if not data:
                    return pd.DataFrame()
                max_len = max(len(r) for r in data) if data else 0
                data = [list(r)+[None]*(max_len-len(r)) for r in data]
                df = pd.DataFrame(data)
                # choose first non-empty row as header
                header_idx = 0
                for i in range(len(df)):
                    if df.iloc[i].notna().any():
                        header_idx = i
                        break
                header = df.iloc[header_idx].astype(str).tolist()
                body = df.iloc[header_idx+1:].reset_index(drop=True)
                body.columns = header
                return body

        return _Shim(book)
