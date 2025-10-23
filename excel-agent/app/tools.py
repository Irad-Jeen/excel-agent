# app/tools.py
from __future__ import annotations
import json
import re
from typing import Any, List, Dict
import pandas as pd

from .xl_readers import load_workbook
from .ai_summaries import analyze_workbook_overview, analyze_sheet_with_ai

# =========================
# Helpers
# =========================
def _parse_json_input(input_str: str) -> dict:
    input_str = (input_str or "").strip()
    if not input_str:
        return {}
    try:
        return json.loads(input_str)
    except Exception as e:
        raise ValueError(f"Tool expected JSON string input, got: {input_str!r}. Error: {e}")

def _load_df(file_path: str, sheet_name: str) -> pd.DataFrame:
    xls = load_workbook(file_path)
    if sheet_name not in xls.sheet_names:
        raise ValueError(f"Sheet '{sheet_name}' not found. Available: {xls.sheet_names}")
    df = xls.parse(sheet_name)
    return df

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

_YEAR_PAT = re.compile(r"(?:^|[^0-9])(20\d{2})(?:[^0-9]|$)")

def _detect_year_columns(df: pd.DataFrame) -> Dict[str, Any]:
    cols = list(map(str, df.columns))
    years: Dict[str, List[str]] = {}
    actual_by_year: Dict[str, List[str]] = {}
    budget_by_year: Dict[str, List[str]] = {}

    for c in cols:
        m = _YEAR_PAT.search(c)
        if not m:
            continue
        y = m.group(1)
        years.setdefault(y, []).append(c)
        lc = c.lower()
        if "actual" in lc or "act " in lc or lc.strip() == y:
            actual_by_year.setdefault(y, []).append(c)
        if "budget" in lc or "plan" in lc:
            budget_by_year.setdefault(y, []).append(c)

    return {
        "years": sorted(years.keys()),
        "map": years,
        "actual": actual_by_year,
        "budget": budget_by_year,
    }

def _find_rows_by_label(
    df: pd.DataFrame,
    label_columns: List[str],
    query: str,
    regex: bool = True,
    case_insensitive: bool = True,
) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)
    flags = re.IGNORECASE if case_insensitive else 0
    for col in label_columns:
        if col not in df.columns:
            continue
        if regex:
            m = df[col].astype(str).str.contains(query, flags=flags, regex=True, na=False)
        else:
            m = df[col].astype(str).str.strip().str.lower() == str(query).strip().lower()
        mask |= m
    return df[mask]

def _numeric_cols(df: pd.DataFrame) -> List[str]:
    num_cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            num_cols.append(str(c))
    return num_cols

def _basic_numeric_stats(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
    out = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        out[str(c)] = {
            "count": int(s.notna().sum()),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if s.notna().sum() > 1 else 0.0,
            "min": float(s.min()),
            "q25": float(s.quantile(0.25)),
            "median": float(s.median()),
            "q75": float(s.quantile(0.75)),
            "max": float(s.max()),
            "missing": int(df[c].isna().sum()),
        }
    return out

def _first_label_cols(df: pd.DataFrame, k: int = 2) -> List[str]:
    labels = []
    for c in df.columns:
        if not _YEAR_PAT.search(str(c)) and df[c].dtype == "O":
            labels.append(str(c))
        if len(labels) >= k:
            break
    if not labels:
        labels = [str(df.columns[0])]
    return labels

def _infer_total_row(df: pd.DataFrame, label_cols: List[str]) -> pd.DataFrame:
    return _find_rows_by_label(df, label_cols, r"(?i)^\s*(total|sum|סה\"כ|סך הכל)\b", regex=True)

def _series_from_row_yearmap(row: pd.Series, info: Dict[str, Any]) -> List[Dict[str, Any]]:
    years = sorted(info["years"])
    series = []
    for y in years:
        cols = info["actual"].get(y) or info["map"].get(y) or []
        val = 0.0
        for c in cols:
            v = pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0]
            if pd.notna(v):
                val += float(v)
        series.append({"year": int(y), "value": val})
    return series

def _yoy_from_series(series: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    yoy = []
    for i in range(1, len(series)):
        curr, prev = series[i]["value"], series[i-1]["value"]
        delta = curr - (prev or 0.0)
        pct = (delta / prev) if (prev not in (None, 0)) else None
        yoy.append({"year": series[i]["year"], "delta": delta, "pct": pct})
    return yoy

# =========================
# Tools — Summaries (smart)
# =========================
def summarize_sheet_tool(input_str: str) -> str:
    try:
        d = _parse_json_input(input_str)
        file_path, sheet_name = d.get("file_path"), d.get("sheet_name")
        if not file_path or not sheet_name:
            return json.dumps({"error": "Provide file_path, sheet_name"}, ensure_ascii=False)

        df = _normalize_columns(_load_df(file_path, sheet_name))
        rows, cols = int(len(df)), int(len(df.columns))
        columns = list(map(str, df.columns))

        years_info = _detect_year_columns(df)
        label_cols = d.get("label_columns") or _first_label_cols(df)
        total_df = _infer_total_row(df, label_cols)
        inferred_total = None
        if not total_df.empty:
            if years_info["years"]:
                row = total_df.iloc[0]
                series = _series_from_row_yearmap(row, years_info)
                yoy = _yoy_from_series(series)
                inferred_total = {"series": series, "yoy": yoy}
            else:
                row = total_df.iloc[0]
                nums = pd.to_numeric(row, errors="coerce")
                inferred_total = {"total_all_numeric": float(nums.dropna().sum())}

        num_cols = _numeric_cols(df)
        numeric_stats = _basic_numeric_stats(df, num_cols)

        nulls = {str(c): int(df[c].isna().sum()) for c in df.columns}
        parse_rate = {}
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            parse_rate[str(c)] = (int(s.notna().sum()) / rows) if rows else 0.0

        highlights = []
        if years_info["years"]:
            highlights.append(f"Detected years: {', '.join(years_info['years'])}")
        if inferred_total and "series" in inferred_total:
            last = inferred_total["series"][-1]["value"]
            highlights.append(f"Total (latest year): {last:,.2f}")
            if inferred_total["yoy"]:
                last_yoy = inferred_total["yoy"][-1]
                pct = last_yoy["pct"]
                if pct is not None:
                    highlights.append(f"YoY delta: {last_yoy['delta']:,.2f} ({pct:.1%})")
        if not num_cols:
            highlights.append("No numeric columns detected; sheet may be metadata or categories.")

        preview = df.head(20).to_dict(orient="records")

        return json.dumps({
            "sheet": sheet_name,
            "rows": rows,
            "cols": cols,
            "columns": columns,
            "years_info": years_info,
            "inferred_total": inferred_total,
            "numeric_stats": numeric_stats,
            "quality": {"nulls_per_column": nulls, "numeric_parse_rate": parse_rate},
            "highlights": highlights,
            "preview_first_20": preview
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"SummarizeSheet: {e}"}, ensure_ascii=False)

def summarize_workbook_tool(input_str: str) -> str:
    try:
        d = _parse_json_input(input_str)
        file_path = d.get("file_path")
        if not file_path:
            return json.dumps({"error": "Missing 'file_path'."}, ensure_ascii=False)

        xls = load_workbook(file_path)
        overview = []
        previews = {}
        for name in xls.sheet_names:
            df = _normalize_columns(xls.parse(name))
            years_info = _detect_year_columns(df)
            label_cols = _first_label_cols(df)
            has_total = not _infer_total_row(df, label_cols).empty
            notes = []
            if years_info["years"]:
                notes.append(f"Detected years: {', '.join(years_info['years'])}")
            if has_total:
                notes.append("Contains a 'Total' row candidate")
            overview.append({
                "sheet": name,
                "rows": int(len(df)),
                "cols": int(len(df.columns)),
                "detected_years": years_info["years"],
                "has_total": bool(has_total),
                "notes": notes
            })
            previews[name] = df.head(12).to_csv(index=False)

        semantic = analyze_workbook_overview(previews)
        return json.dumps({"sheets": overview, "semantic": semantic}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"SummarizeWorkbook: {e}"}, ensure_ascii=False)

# =========================
# Tools — Utilities
# =========================
def list_sheets_tool(input_str: str) -> str:
    try:
        data = _parse_json_input(input_str)
        file_path = data.get("file_path")
        if not file_path:
            return json.dumps({"error": "Missing 'file_path'."}, ensure_ascii=False)
        xls = load_workbook(file_path)
        return json.dumps({"sheets": xls.sheet_names}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"ListSheets: {e}"}, ensure_ascii=False)

def sheet_columns_tool(input_str: str) -> str:
    try:
        d = _parse_json_input(input_str)
        df = _normalize_columns(_load_df(d["file_path"], d["sheet_name"]))
        return json.dumps({"columns": list(map(str, df.columns))}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"SheetColumns: {e}"}, ensure_ascii=False)

def sheet_preview_tool(input_str: str) -> str:
    try:
        d = _parse_json_input(input_str)
        n = int(d.get("n", 10))
        df = _normalize_columns(_load_df(d["file_path"], d["sheet_name"]))
        return json.dumps({"rows": df.head(n).to_dict(orient="records")}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"SheetPreview: {e}"}, ensure_ascii=False)

def detect_year_columns_tool(input_str: str) -> str:
    try:
        d = _parse_json_input(input_str)
        df = _normalize_columns(_load_df(d["file_path"], d["sheet_name"]))
        info = _detect_year_columns(df)
        return json.dumps(info, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"DetectYearColumns: {e}"}, ensure_ascii=False)

def find_rows_tool(input_str: str) -> str:
    try:
        d = _parse_json_input(input_str)
        df = _normalize_columns(_load_df(d["file_path"], d["sheet_name"]))
        # Use top label-like columns by default (e.g., Category, Subcategory)
        label_cols = d.get("label_columns") or _first_label_cols(df, k=2)
        # Support alias: label_regex
        query = d.get("query") or d.get("label_regex") or ".*"
        out = _find_rows_by_label(
            df, label_cols,
            query,
            regex=bool(d.get("regex", True) or d.get("label_regex") is not None),
            case_insensitive=True
        )
        # Optional column selection
        select = d.get("select")
        if select:
            cols = [c for c in select if c in out.columns]
            if cols:
                out = out[cols]
        # Optional row limit
        limit = int(d.get("limit") or d.get("row_limit") or 0)
        if limit and limit > 0:
            out = out.head(limit)
        return json.dumps({
            "matched": out.to_dict(orient="records"),
            "idx": out.index.tolist()
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"FindRows: {e}"}, ensure_ascii=False)

def totals_row_tool(input_str: str) -> str:
    try:
        d = _parse_json_input(input_str)
        df = _normalize_columns(_load_df(d["file_path"], d["sheet_name"]))
        label_cols = d.get("label_columns") or _first_label_cols(df)
        total_regex = d.get("total_regex", r"(?i)total|סה\"כ|סך הכל")
        cand = _find_rows_by_label(df, label_cols, total_regex, regex=True)
        if cand.empty:
            return json.dumps({"totals": None, "note": "No total row found"}, ensure_ascii=False)
        info = _detect_year_columns(df)
        years = info["years"]
        selection = {}
        cols_used = []

        # Heuristic: choose the most appropriate total row among candidates
        # 1) Exact match: "Total Expenses" preferred, then "Total Revenues", then any "Total ... Expenses", then "Total ... Revenues"
        # 2) If none, choose the candidate with the shortest label text that contains 'total'
        def _label_text(r) -> str:
            return " | ".join(str(r[c]) for c in label_cols if c in df.columns).strip()

        labels = [(_label_text(r), idx) for idx, r in cand.iterrows()]
        # Normalize
        norm = [(re.sub(r"\s+", " ", (t or "").strip()), idx) for t, idx in labels]
        # Ranking rules
        def _rank(label: str) -> tuple:
            l = label.lower()
            # exact priority
            if re.match(r"^\s*total\s+expenses\b", l):
                return (0, len(l))
            if re.match(r"^\s*total\s+revenue(s)?\b", l):
                return (1, len(l))
            if re.match(r"^\s*total\b.*\bexpenses\b", l):
                return (2, len(l))
            if re.match(r"^\s*total\b.*\brevenue(s)?\b", l):
                return (3, len(l))
            # generic total
            if re.match(r"^\s*total\b", l):
                return (4, len(l))
            return (9, len(l))

        best_idx = min(norm, key=lambda x: _rank(x[0]))[1]
        row = cand.loc[best_idx]
        if years:
            for y in years:
                cols = info["actual"].get(y) or info["map"].get(y) or []
                if not cols:
                    continue
                vals = []
                for c in cols:
                    v = pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0]
                    if pd.notna(v):
                        vals.append(float(v))
                        cols_used.append(c)
                if vals:
                    selection[y] = sum(vals)
            # Also return all candidate labels with ranks to help the agent disambiguate
            candidates_meta = [{"label": t, "rank": _rank(t)[0]} for t, _ in norm]
            return json.dumps({
                "totals": selection,
                "columns_used": cols_used,
                "label_used": _label_text(row),
                "candidates": candidates_meta
            }, ensure_ascii=False)
        else:
            nums = pd.to_numeric(row, errors="coerce")
            candidates_meta = [{"label": t, "rank": _rank(t)[0]} for t, _ in norm]
            return json.dumps({
                "totals": {"all_numeric_sum": float(nums.dropna().sum())},
                "label_used": _label_text(row),
                "candidates": candidates_meta
            }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"TotalsRow: {e}"}, ensure_ascii=False)

def yoy_for_label_tool(input_str: str) -> str:
    try:
        d = _parse_json_input(input_str)
        df = _normalize_columns(_load_df(d["file_path"], d["sheet_name"]))
        label_cols = d.get("label_columns") or _first_label_cols(df)
        label_q = d.get("label_query", r"(?i)^total")
        cand = _find_rows_by_label(df, label_cols, label_q, regex=bool(d.get("regex", True)))
        if cand.empty:
            return json.dumps({"error": "Label not found"}, ensure_ascii=False)

        info = _detect_year_columns(df)
        years = sorted(info["years"])
        row = cand.iloc[0]
        series = []
        for y in years:
            cols = info["actual"].get(y) or info["map"].get(y) or []
            val = 0.0
            for c in cols:
                v = pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0]
                if pd.notna(v):
                    val += float(v)
            series.append({"year": int(y), "value": val})

        yoy = []
        for i in range(1, len(series)):
            curr = series[i]["value"]
            prev = series[i-1]["value"]
            if prev is None or prev == 0:
                delta = curr - (prev or 0.0)
                pct = None
            else:
                delta = curr - prev
                pct = delta / prev
            yoy.append({"year": series[i]["year"], "delta": delta, "pct": pct})

        return json.dumps({"series": series, "yoy": yoy}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"YoYForLabel: {e}"}, ensure_ascii=False)

def quality_report_tool(input_str: str) -> str:
    try:
        d = _parse_json_input(input_str)
        df = _normalize_columns(_load_df(d["file_path"], d["sheet_name"]))
        row_count = int(len(df))
        col_count = int(len(df.columns))
        nulls = {str(c): int(df[c].isna().sum()) for c in df.columns}

        rate = {}
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            parsed = int(s.notna().sum())
            rate[str(c)] = parsed / row_count if row_count else 0.0

        return json.dumps({
            "row_count": row_count,
            "col_count": col_count,
            "nulls_per_column": nulls,
            "numeric_parse_rate": rate
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"QualityReport: {e}"}, ensure_ascii=False)

# =========================
# Tools — Computational
# =========================
def column_stats_tool(input_str: str) -> str:
    try:
        d = _parse_json_input(input_str)
        df = _normalize_columns(_load_df(d["file_path"], d["sheet_name"]))
        cols = d.get("columns") or _numeric_cols(df)
        return json.dumps({"stats": _basic_numeric_stats(df, cols)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"ColumnStats: {e}"}, ensure_ascii=False)

def compute_aggregate_tool(input_str: str) -> str:
    try:
        d = _parse_json_input(input_str)
        df = _normalize_columns(_load_df(d["file_path"], d["sheet_name"]))
        # Accept aliases for values
        values = d.get("values") or d.get("columns")
        # Normalize target intent (e.g., expenses, revenues)
        target = str(d.get("target") or d.get("label") or "").strip().lower()
        target_regex: str | None = d.get("label_regex")
        if not target_regex and target:
            if "expense" in target:
                target_regex = r"(?i)\bexpenses\b"
            elif "revenue" in target or "income" in target:
                target_regex = r"(?i)\brevenue(s)?\b|\bincome\b"
            elif "profit" in target:
                target_regex = r"(?i)\bprofit\b"
        # If values not provided but year is, auto-detect columns for that year
        # If values not provided but year is, auto-detect columns for that year
        if not values and d.get("year") is not None:
            info = _detect_year_columns(df)
            y = str(int(d.get("year")))
            detected = info["actual"].get(y) or info["map"].get(y) or []
            # Prefer monthly columns when present
            only_months = bool(d.get("only_months")) or any(re.search(r"[_\s-](0[1-9]|1[0-2])\b", str(c)) for c in detected)
            if only_months:
                detected = [c for c in detected if re.search(r"[_\s-](0[1-9]|1[0-2])\b", str(c))]
            values = detected
        # Optional pre-filter by label regex to restrict rows (e.g., only Expenses)
        if target_regex:
            label_cols = d.get("label_columns") or _first_label_cols(df)
            df = _find_rows_by_label(df, label_cols, target_regex, regex=True)
            if df.empty:
                return json.dumps({"rows": [], "note": "Filter produced no rows"}, ensure_ascii=False)
        elif d.get("label_regex"):
            label_cols = d.get("label_columns") or _first_label_cols(df)
            df = _find_rows_by_label(df, label_cols, d.get("label_regex"), regex=True)
            if df.empty:
                return json.dumps({"rows": [], "note": "Filter produced no rows"}, ensure_ascii=False)
        if not values:
            return json.dumps({"error": "ComputeAggregate: provide 'values' (list of COLUMN NAMES). Example: {\"values\":[\"Actual 2024\"]}"},
                              ensure_ascii=False)

        if any(not isinstance(c, str) for c in values):
            return json.dumps({"error": "ComputeAggregate: 'values' must be column NAMES (strings), not raw numbers. Example: {\"values\":[\"Actual 2024\"]}"},
                              ensure_ascii=False)

        missing = [c for c in values if c not in df.columns]
        if missing:
            return json.dumps({"error": f"ComputeAggregate: columns not found: {missing}. "
                                        f"Available (first 20): {list(map(str, df.columns))[:20]}"},
                              ensure_ascii=False)

        # Accept aliases for agg
        agg = (d.get("agg") or d.get("aggregate_type") or d.get("aggregate_function") or "sum").lower()
        agg_map = {"sum":"sum","avg":"mean","mean":"mean","min":"min","max":"max","median":"median","count":"count"}
        if agg not in agg_map:
            return json.dumps({"error": f"ComputeAggregate: unsupported agg '{agg}'. Use one of {list(agg_map)}"},
                              ensure_ascii=False)

        # Prefer existing total row if present for simple sums without grouping
        gb = d.get("group_by") or None
        if agg == "sum" and not gb:
            label_cols = d.get("label_columns") or _first_label_cols(df)
            total_df = _infer_total_row(df, label_cols)
            if not total_df.empty:
                row = total_df.iloc[0]
                out_row: Dict[str, Any] = {}
                for c in values:
                    v = pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0]
                    if pd.notna(v):
                        out_row[str(c)] = float(v)
                if out_row:
                    return json.dumps({"rows": [out_row], "source": "existing_total_row"}, ensure_ascii=False)

        for c in values:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        if gb:
            if any(g not in df.columns for g in gb):
                bad = [g for g in gb if g not in df.columns]
                return json.dumps({"error": f"ComputeAggregate: group_by columns not found: {bad}"},
                                  ensure_ascii=False)
            out = getattr(df.groupby(gb)[values], agg_map[agg])().reset_index()
        else:
            out = getattr(df[values], agg_map[agg])()
            out = out.to_frame().T
            # Return a single total across the provided values when summing
            if agg == "sum":
                total_sum = float(out[values].sum(axis=1).iloc[0])
                return json.dumps({"rows": out.to_dict(orient="records"), "total_sum": total_sum}, ensure_ascii=False)

        return json.dumps({"rows": out.to_dict(orient="records")}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"ComputeAggregate: {e}"}, ensure_ascii=False)

def compute_ratio_tool(input_str: str) -> str:
    try:
        d = _parse_json_input(input_str)
        df = _normalize_columns(_load_df(d["file_path"], d["sheet_name"]))

        if d.get("numerator_col") and d.get("denominator_col"):
            num = pd.to_numeric(df[d["numerator_col"]], errors="coerce").sum()
            den = pd.to_numeric(df[d["denominator_col"]], errors="coerce").sum()
            ratio = (num / den) if den not in (None, 0) else None
            return json.dumps({"ratio": ratio, "details": {"numerator": num, "denominator": den}}, ensure_ascii=False)

        label_cols = d.get("label_columns") or _first_label_cols(df)
        num_rows = _find_rows_by_label(df, label_cols, d.get("numerator_label","(?i)^total revenues$"), regex=True)
        den_rows = _find_rows_by_label(df, label_cols, d.get("denominator_label","(?i)^total expenses$"), regex=True)
        if num_rows.empty or den_rows.empty:
            return json.dumps({"ratio": None, "details": "Label rows not found"}, ensure_ascii=False)

        num = pd.to_numeric(num_rows.select_dtypes(exclude=['object']).stack(), errors="coerce").sum()
        den = pd.to_numeric(den_rows.select_dtypes(exclude=['object']).stack(), errors="coerce").sum()
        ratio = (num / den) if den not in (None, 0) else None
        return json.dumps({"ratio": ratio, "details": {"numerator": num, "denominator": den}}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"ComputeRatio: {e}"}, ensure_ascii=False)

def yoy_table_tool(input_str: str) -> str:
    try:
        d = _parse_json_input(input_str)
        df = _normalize_columns(_load_df(d["file_path"], d["sheet_name"]))
        label_cols = d.get("label_columns") or _first_label_cols(df)
        info = _detect_year_columns(df)
        if not info["years"]:
            return json.dumps({"rows": [], "note": "No year columns detected"}, ensure_ascii=False)

        if d.get("label_regex"):
            labels_df = _find_rows_by_label(df, label_cols, d["label_regex"], regex=True)
        else:
            labels_df = df

        rows = []
        for _, r in labels_df.iterrows():
            label_val = " | ".join(str(r[c]) for c in label_cols if c in df.columns)
            if not label_val.strip():
                continue
            series = _series_from_row_yearmap(r, info)
            if all((s["value"] == 0 or s["value"] is None) for s in series):
                continue
            rows.append({"label": label_val, "series": series, "yoy": _yoy_from_series(series)})

        return json.dumps({"rows": rows}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"YoYTable: {e}"}, ensure_ascii=False)

def topn_changes_tool(input_str: str) -> str:
    try:
        d = _parse_json_input(input_str)
        n = int(d.get("n", 10))
        res = json.loads(yoy_table_tool(json.dumps({
            "file_path": d["file_path"],
            "sheet_name": d["sheet_name"],
            "label_columns": d.get("label_columns"),
        })))
        if "rows" not in res:
            return json.dumps(res, ensure_ascii=False)

        top = []
        for row in res["rows"]:
            for y in row["yoy"]:
                top.append({
                    "label": row["label"],
                    "year": y["year"],
                    "delta": y["delta"],
                    "pct": y["pct"],
                })
        top.sort(key=lambda x: abs(x["delta"] or 0), reverse=True)
        return json.dumps({"top": top[:n]}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"TopNChanges: {e}"}, ensure_ascii=False)

def pivot_mini_tool(input_str: str) -> str:
    try:
        d = _parse_json_input(input_str)
        df = _normalize_columns(_load_df(d["file_path"], d["sheet_name"]))
        idx = d.get("index") or []
        cols = d.get("columns")
        vals = d.get("values")
        agg = (d.get("agg") or "sum").lower()
        agg_map = {"sum":"sum","avg":"mean","mean":"mean","min":"min","max":"max","median":"median","count":"count"}
        if not vals:
            return json.dumps({"error": "Provide 'values' list"}, ensure_ascii=False)
        for c in vals:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        pv = pd.pivot_table(df, index=idx, columns=cols, values=vals, aggfunc=agg_map.get(agg, "sum"), fill_value=0)
        pv = pv.reset_index()
        return json.dumps({"rows": json.loads(pv.to_json(orient="records"))}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"PivotMini: {e}"}, ensure_ascii=False)

def filter_rows_tool(input_str: str) -> str:
    try:
        d = _parse_json_input(input_str)
        df = _normalize_columns(_load_df(d["file_path"], d["sheet_name"]))
        where = d.get("where") or []

        def _normalize_cond(cond):
            if "op" in cond:
                return cond["op"], cond.get("value")
            for k, op in [("gt", ">"), ("gte", ">="), ("lt", "<"), ("lte", "<="), ("eq", "=="), ("ne", "!=")]:
                if k in cond:
                    return op, cond[k]
            return None, None

        for cond in where:
            col = cond["column"]
            op, val = _normalize_cond(cond)

            if op == "regex":
                val = cond.get("value", "")
                df = df[df[col].astype(str).str.contains(val, regex=True, na=False)]
                continue

            if op in (">", "<", ">=", "<=", "==", "!="):
                s = pd.to_numeric(df[col], errors="coerce")
                df = df[eval(f"s {op} @val")]
            else:
                return json.dumps({"error": f"Unsupported or missing operator for condition on '{col}'"}, ensure_ascii=False)

        select = d.get("select")
        if select:
            df = df[[c for c in select if c in df.columns]]
        limit = int(d.get("limit", 100))
        return json.dumps({"rows": df.head(limit).to_dict(orient="records")}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"FilterRows: {e}"}, ensure_ascii=False)

# =========================
# Tools — New (Structure & Semantics)
# =========================
def detect_total_columns_tool(input_str: str) -> str:
    try:
        d = _parse_json_input(input_str)
        df = _normalize_columns(_load_df(d["file_path"], d["sheet_name"]))
        cols = list(df.columns)
        candidates = []
        for c in cols:
            lc = str(c).lower()
            if any(k in lc for k in ["total", "sum", "all", "סה", "סך", "סה\"כ", "Σ"]):
                candidates.append(str(c))
        verified = []
        for tc in candidates:
            t = pd.to_numeric(df[tc], errors="coerce")
            others = [c for c in cols if c != tc]
            s = pd.to_numeric(df[others], errors="coerce").sum(axis=1)
            match_rate = float(((t - s).abs() <= (abs(s) * 0.01 + 1e-9)).mean())
            verified.append({"column": tc, "approx_sum_match_rate": match_rate})
        return json.dumps({"candidates": verified}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"DetectTotalColumns: {e}"}, ensure_ascii=False)

def detect_subtotal_rows_tool(input_str: str) -> str:
    try:
        d = _parse_json_input(input_str)
        df = _normalize_columns(_load_df(d["file_path"], d["sheet_name"]))
        label_cols = d.get("label_columns") or _first_label_cols(df)
        subt = _find_rows_by_label(df, label_cols, r"(?i)\b(sub[- ]?total|תת[- ]?סך|ביניים)\b", regex=True)
        return json.dumps({"rows": subt.to_dict(orient="records"), "idx": subt.index.tolist()}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"DetectSubtotalRows: {e}"}, ensure_ascii=False)

def detect_tables_tool(input_str: str) -> str:
    try:
        d = _parse_json_input(input_str)
        df = _normalize_columns(_load_df(d["file_path"], d["sheet_name"]))
        is_empty_row = df.isna().all(axis=1)
        boundaries = [0]
        for i in range(1, len(df)):
            if is_empty_row.iloc[i] and not is_empty_row.iloc[i-1]:
                boundaries.append(i)
            if not is_empty_row.iloc[i] and is_empty_row.iloc[i-1]:
                boundaries.append(i)
        boundaries = sorted(set(boundaries + [len(df)]))

        regions = []
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            block = df.iloc[a:b]
            if block.dropna(how="all").empty:
                continue
            header_idx = 0
            cols = block.columns.tolist()
            regions.append({
                "top": int(a),
                "bottom": int(b - 1),
                "header_row": int(header_idx),
                "columns": list(map(str, cols)),
                "rows": int(len(block.dropna(how='all')))
            })
        return json.dumps({"regions": regions}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"DetectTables: {e}"}, ensure_ascii=False)

def explain_sheet_purpose_tool(input_str: str) -> str:
    try:
        d = _parse_json_input(input_str)
        file_path, sheet_name = d.get("file_path"), d.get("sheet_name")
        if not file_path or not sheet_name:
            return json.dumps({"error": "Provide file_path, sheet_name"}, ensure_ascii=False)
        df = _normalize_columns(_load_df(file_path, sheet_name))
        preview_csv = df.head(30).to_csv(index=False)
        analysis = analyze_sheet_with_ai(sheet_name, preview_csv)
        return json.dumps({"sheet": sheet_name, "analysis": analysis}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"ExplainSheetPurpose: {e}"}, ensure_ascii=False)

# =========================
# Tools — Relationships
# =========================
def _text_like_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        s = df[c]
        # consider object dtype or mixed columns as label-like
        if s.dtype == "O":
            cols.append(str(c))
    return cols

def _value_set(series: pd.Series, limit: int = 10000) -> set:
    vals = set()
    for v in series.dropna().astype(str).head(limit):
        vv = v.strip()
        if vv:
            vals.add(vv)
    return vals

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter / union) if union else 0.0

def infer_sheet_relations_tool(input_str: str) -> str:
    """
    Infer likely relationships between sheets via column-name similarity and value overlaps.
    Input: {"file_path": "/path.xlsx", "limit_per_sheet": 500}
    Output: {"relations": [{"sheet_a":..., "sheet_b":..., "column_mappings":[{"a_col":...,"b_col":...,"name_similarity":...,"value_overlap":...}], "notes":[...] }], "summary": "..."}
    """
    try:
        d = _parse_json_input(input_str)
        file_path = d.get("file_path")
        if not file_path:
            return json.dumps({"error": "InferSheetRelations: Missing 'file_path'"}, ensure_ascii=False)
        limit = int(d.get("limit_per_sheet", 1000))

        xls = load_workbook(file_path)
        sheets = xls.sheet_names
        dfs: Dict[str, pd.DataFrame] = {}
        for name in sheets:
            df = xls.parse(name)
            df = _normalize_columns(df)
            if len(df) > limit:
                df = df.head(limit)
            dfs[name] = df

        relations: List[Dict[str, Any]] = []
        notes_global: List[str] = []

        # Precompute text-like columns and year info
        label_cols_by_sheet: Dict[str, List[str]] = {}
        years_by_sheet: Dict[str, Dict[str, Any]] = {}
        for name, df in dfs.items():
            label_cols_by_sheet[name] = _first_label_cols(df, k=3)
            years_by_sheet[name] = _detect_year_columns(df)

        # Pairwise compare sheets
        for i in range(len(sheets)):
            for j in range(i + 1, len(sheets)):
                a, b = sheets[i], sheets[j]
                dfa, dfb = dfs[a], dfs[b]
                a_labels = [c for c in _text_like_columns(dfa) if c in label_cols_by_sheet[a]]
                b_labels = [c for c in _text_like_columns(dfb) if c in label_cols_by_sheet[b]]
                if not a_labels or not b_labels:
                    continue

                mappings: List[Dict[str, Any]] = []
                for ca in a_labels:
                    set_a = _value_set(dfa[ca])
                    for cb in b_labels:
                        set_b = _value_set(dfb[cb])
                        vo = _jaccard(set_a, set_b)
                        if vo >= 0.2:  # heuristic threshold
                            # name similarity (simple, lowercased token Jaccard)
                            na = set(str(ca).lower().split())
                            nb = set(str(cb).lower().split())
                            name_sim = _jaccard(na, nb)
                            samples = list((set_a & set_b))[:5]
                            mappings.append({
                                "a_col": str(ca),
                                "b_col": str(cb),
                                "name_similarity": float(name_sim),
                                "value_overlap": float(vo),
                                "sample_overlap_values": samples,
                            })

                rel_notes: List[str] = []
                # If one sheet has years and the other is mostly labels, suggest structure→fact relation
                a_has_years = bool(years_by_sheet[a].get("years"))
                b_has_years = bool(years_by_sheet[b].get("years"))
                if mappings:
                    if a_has_years and not b_has_years:
                        rel_notes.append("Sheet B likely defines hierarchy/labels used by sheet A.")
                    if b_has_years and not a_has_years:
                        rel_notes.append("Sheet A likely defines hierarchy/labels used by sheet B.")
                    if a_has_years and b_has_years:
                        rel_notes.append("Both sheets have time-series; mappings may represent shared dimensions.")

                    relations.append({
                        "sheet_a": a,
                        "sheet_b": b,
                        "column_mappings": mappings,
                        "notes": rel_notes,
                    })

        summary = "Found {} relation candidates across {} sheets.".format(len(relations), len(sheets))
        if not relations:
            notes_global.append("No strong label overlaps detected; try adjusting label columns or increase limit.")

        return json.dumps({"relations": relations, "summary": summary, "notes": notes_global}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"InferSheetRelations: {e}"}, ensure_ascii=False)

# =========================
# Tools — Meta (Docs)
# =========================
def _tool_specs() -> Dict[str, Any]:
    return {
        "SummarizeWorkbook": {
            "purpose": "High-level workbook summary + semantic intent.",
            "required": ["file_path"],
            "optional": [],
            "example_input": {"file_path": "/path/file.xlsx"},
            "returns": {"sheets": "...", "semantic": "..."}
        },
        "SummarizeSheet": {
            "purpose": "Smart sheet summary: columns, totals detection, YoY, stats, quality, preview.",
            "required": ["file_path", "sheet_name"],
            "optional": ["label_columns"],
            "example_input": {"file_path": "/path/file.xlsx", "sheet_name": "P&L Insurance YOY"},
            "returns": {"columns": "...", "years_info": "...", "inferred_total": "...", "preview_first_20": "..."}
        },
        "ExplainSheetPurpose": {
            "purpose": "3–6 bullets: semantic purpose of a sheet from preview.",
            "required": ["file_path", "sheet_name"],
            "optional": [],
            "example_input": {"file_path": "/path/file.xlsx", "sheet_name": "P&L Insurance YOY"},
            "returns": {"sheet": "...", "analysis": "..."}
        },
        "ListSheets": {
            "purpose": "List sheet names in workbook.",
            "required": ["file_path"],
            "optional": [],
            "example_input": {"file_path": "/path/file.xlsx"},
            "returns": {"sheets": ["..."]}
        },
        "SheetColumns": {
            "purpose": "Return column headers of a sheet.",
            "required": ["file_path", "sheet_name"],
            "optional": [],
            "example_input": {"file_path": "/path/file.xlsx", "sheet_name": "P&L Insurance YOY"},
            "returns": {"columns": ["..."]}
        },
        "SheetPreview": {
            "purpose": "Return first N rows (records).",
            "required": ["file_path", "sheet_name"],
            "optional": ["n"],
            "example_input": {"file_path": "/path/file.xlsx", "sheet_name": "P&L Insurance YOY", "n": 10},
            "returns": {"rows": [{"col": "..."}, "..."]}
        },
        "DetectYearColumns": {
            "purpose": "Detect year columns and classify as 'actual' / 'budget'.",
            "required": ["file_path", "sheet_name"],
            "optional": [],
            "example_input": {"file_path": "/path/file.xlsx", "sheet_name": "P&L Insurance YOY"},
            "returns": {"years": ["2021","2022","..."], "actual": {"2024": ["Actual 2024"]}}
        },
        "DetectTables": {
            "purpose": "Detect multiple table regions (by blank row blocks).",
            "required": ["file_path", "sheet_name"],
            "optional": [],
            "example_input": {"file_path": "/path/file.xlsx", "sheet_name": "Sheet1"},
            "returns": {"regions": [{"top":0,"bottom":20,"columns":[...]}]}
        },
        "FindRows": {
            "purpose": "Find rows by label regex/exact in label columns.",
            "required": ["file_path", "sheet_name"],
            "optional": ["label_columns", "query", "regex"],
            "example_input": {"file_path": "/path/file.xlsx", "sheet_name": "P&L Insurance YOY", "label_columns": ["Category (Level 1)", "Subcategory (Level 2)"], "query": "(?i)^Total Expenses$", "regex": True},
            "returns": {"matched": [{"...": "..."}], "idx": [0,1]}
        },
        "TotalsRow": {
            "purpose": "Locate a total row and return per-year totals (if years exist).",
            "required": ["file_path", "sheet_name"],
            "optional": ["label_columns", "total_regex"],
            "example_input": {"file_path": "/path/file.xlsx", "sheet_name": "P&L Insurance YOY", "total_regex": "(?i)^Total Expenses$"},
            "returns": {"totals": {"2024": 494}}
        },
        "YoYForLabel": {
            "purpose": "YoY series for a label row across detected years.",
            "required": ["file_path", "sheet_name"],
            "optional": ["label_columns", "label_query", "regex"],
            "example_input": {"file_path": "/path/file.xlsx", "sheet_name": "P&L Insurance YOY", "label_query": "(?i)^Total Expenses$"},
            "returns": {"series":[{"year":2024,"value":494}],"yoy":[...]}
        },
        "QualityReport": {
            "purpose": "Nulls per column + numeric parse rates.",
            "required": ["file_path", "sheet_name"],
            "optional": [],
            "example_input": {"file_path": "/path/file.xlsx", "sheet_name": "P&L Insurance YOY"},
            "returns": {"row_count": 50, "numeric_parse_rate": {"Actual 2024": 1.0}}
        },
        "DetectTotalColumns": {
            "purpose": "Find total-like columns and approx sum match rate.",
            "required": ["file_path", "sheet_name"],
            "optional": [],
            "example_input": {"file_path": "/path/file.xlsx", "sheet_name": "Sheet1"},
            "returns": {"candidates": [{"column":"Total","approx_sum_match_rate":0.97}]}
        },
        "DetectSubtotalRows": {
            "purpose": "Detect subtotal rows by heuristics.",
            "required": ["file_path", "sheet_name"],
            "optional": ["label_columns"],
            "example_input": {"file_path": "/path/file.xlsx", "sheet_name": "Sheet1"},
            "returns": {"rows": [{"...":"..."}]}
        },
        "ColumnStats": {
            "purpose": "Basic numeric stats per given columns (or inferred numeric).",
            "required": ["file_path", "sheet_name"],
            "optional": ["columns"],
            "example_input": {"file_path": "/path/file.xlsx", "sheet_name": "P&L Insurance YOY", "columns": ["Actual 2024"]},
            "returns": {"stats": {"Actual 2024": {"mean": 47.64}}}
        },
        "ComputeAggregate": {
            "purpose": "Aggregate numeric columns. IMPORTANT: 'values' must be column NAMES, not raw numbers.",
            "required": ["file_path","sheet_name","values"],
            "optional": ["group_by","agg"],
            "example_input": {"file_path":"/path/file.xlsx","sheet_name":"P&L Insurance YOY","values":["Actual 2024"],"agg":"sum"},
            "returns": {"rows":[{"Actual 2024": 494}]}
        },
        "ComputeRatio": {
            "purpose": "Compute ratio between two columns OR between labeled totals.",
            "required": ["file_path","sheet_name"],
            "optional": ["numerator_col","denominator_col","label_columns","numerator_label","denominator_label"],
            "example_input": {"file_path":"/path.xlsx","sheet_name":"P&L Insurance YOY","numerator_label":"(?i)^Total Revenues$","denominator_label":"(?i)^Total Expenses$"},
            "returns": {"ratio": 0.83, "details": {"numerator":..., "denominator":...}}
        },
        "YoYTable": {
            "purpose": "YoY for many labels (optionally label_regex).",
            "required": ["file_path","sheet_name"],
            "optional": ["label_columns","label_regex"],
            "example_input": {"file_path":"/path.xlsx","sheet_name":"P&L Insurance YOY","label_regex":"(?i)Expenses|Revenues"},
            "returns": {"rows":[{"label":"Total Expenses","series":[...]}]}
        },
        "TopNChanges": {
            "purpose": "Top-N YoY deltas by absolute change.",
            "required": ["file_path","sheet_name"],
            "optional": ["label_columns","n"],
            "example_input": {"file_path":"/path.xlsx","sheet_name":"P&L Insurance YOY","n":10},
            "returns": {"top":[{"label":"...","year":2024,"delta":...}]}
        },
        "PivotMini": {
            "purpose": "Minimal pivot.",
            "required": ["file_path","sheet_name","values"],
            "optional": ["index","columns","agg"],
            "example_input": {"file_path":"/path.xlsx","sheet_name":"P&L Insurance YOY","index":["Category (Level 1)"],"values":["Actual 2024"],"agg":"sum"},
            "returns": {"rows":[{"Category (Level 1)":"Expenses","Actual 2024":494}]}
        },
        "FilterRows": {
            "purpose": "Filter by numeric comparisons or regex; then select and limit.",
            "required": ["file_path","sheet_name"],
            "optional": ["where","select","limit"],
            "example_input": {"file_path":"/path.xlsx","sheet_name":"P&L Insurance YOY","where":[{"column":"Category (Level 1)","op":"regex","value":"^(?i)\\s*Expenses\\s*$"}],"select":["Category (Level 1)","Actual 2024"],"limit":1000},
            "returns": {"rows":[{"Category (Level 1)":"Expenses","Actual 2024":...}]}
        },
    }

def describe_tools_tool(input_str: str) -> str:
    try:
        d = _parse_json_input(input_str)
        names = d.get("names")
        specs = _tool_specs()
        if names:
            out = {k: specs[k] for k in names if k in specs}
        else:
            out = specs
        return json.dumps({"tools": out}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"DescribeTools: {e}"}, ensure_ascii=False)
