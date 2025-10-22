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
                # אין עמודות שנים — נחזיר סכום across כל העמודות המספריות
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

        semantic = analyze_workbook_overview(previews)  # purpose/key_entities/key_metrics/...
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
        label_cols = d.get("label_columns") or [str(df.columns[0])]
        out = _find_rows_by_label(
            df, label_cols,
            d.get("query", ".*"),
            regex=bool(d.get("regex", True)),
            case_insensitive=True
        )
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
        row = cand.iloc[0]
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
            return json.dumps({"totals": selection, "columns_used": cols_used}, ensure_ascii=False)
        else:
            # ללא עמודות שנים — נחזיר סכום across עמודות מספריות
            nums = pd.to_numeric(row, errors="coerce")
            return json.dumps({"totals": {"all_numeric_sum": float(nums.dropna().sum())}}, ensure_ascii=False)
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
        values = d.get("values")
        if not values:
            return json.dumps({"error": "Provide 'values' list"}, ensure_ascii=False)
        agg = (d.get("agg") or "sum").lower()
        agg_map = {"sum":"sum","avg":"mean","mean":"mean","min":"min","max":"max","median":"median","count":"count"}
        if agg not in agg_map:
            return json.dumps({"error": f"Unsupported agg '{agg}'"}, ensure_ascii=False)

        for c in values:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        gb = d.get("group_by") or None
        if gb:
            out = getattr(df.groupby(gb)[values], agg_map[agg])().reset_index()
        else:
            out = getattr(df[values], agg_map[agg])()
            out = out.to_frame().T
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
            # explicit op
            if "op" in cond:
                return cond["op"], cond.get("value")
            # shorthand ops
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
            # התאמה בקירוב (1% טולרנס)
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
        # היוריסטיקה בסיסית: בלוקים מופרדים ע"י שורות ריקות
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
    """
    LLM-עזר: 3–6 בולטים על מטרת הגיליון והישויות/מדדים בו.
    """
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
