
from __future__ import annotations

import io
import math
import os
import re
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ENGINE_VERSION = "2026-03-20-parity2"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil import parser as dateparser
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment


MAIN_OUTPUT_COLUMNS = [
    "S_No",
    "Gate_Reg_No_Raw",
    "Gate_Reg_No_Cleaned",
    "Gate_Reg_No_Final",
    "Gate_Reg_Strict_Format_Flag",
    "Final_Reg_Strict_Format_Flag",
    "Reg_Format_Cleaned_Flag",
    "Reg_Corrected_via_System_Flag",
    "Reporting_Date_Clean",
    "Reporting_Time_Clean",
    "Workshop_In_Date_Clean",
    "Workshop_In_Time_Clean",
    "Workshop_Out_Date_Clean",
    "Workshop_Out_Time_Clean",
    "Gate_In_Source",
    "Base_Gate_In_Date",
    "Base_Gate_In_Time",
    "Corrected_Gate_Out_Date",
    "Corrected_Gate_Out_Time",
    "Gate_Register_GIGO_TAT_Hours",
    "Original_TAT_Hours",
    "Corrected_TAT_Hours",
    "TAT_Corrected_Flag",
    "Gate_Time_Corrected_Flag",
    "TAT_Correction_Source",
    "Matched_Job_Card_No",
    "Matched_System_Reg_No",
    "Match_Type",
    "Match_Score",
    "Match_Time_Diff_Hours",
    "Match_Bill_Diff_Hours",
    "Matched_System_Gate_In_Date",
    "Matched_System_Gate_In_Time",
    "Matched_System_Bill_Date",
    "Matched_System_Bill_Time",
    "ROT_Hours",
    "GIGO_TAT_Delay_Hours",
    "System_Validation_Available_Flag",
    "Top_Suggested_Job_Card_No",
    "Top_Suggested_System_Reg_No",
    "Top_Suggestion_Score",
    "Top_Suggestion_Time_Diff_Hours",
    "Gate_In_Before_Gate_Out_Flag",
    "Outlier_Flag",
    "ROT_Work_Mismatch_Flag",
    "Business_Validation_Status",
    "Final_Considered_Flag",
    "Any_Correction_Flag",
    "Correction_Flags",
    "Final_Remarks",
]

OPERATING_START = (11, 1)
OPERATING_END = (12, 15)

ALPHA_POS = {0, 1, 4, 5}
NUM_POS = {2, 3, 6, 7, 8, 9}
OCR_ALPHA_TO_NUM = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2"}
OCR_NUM_TO_ALPHA = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z"}

GATE_HINTS = {
    "reg": [
        ["registration"],
        ["vehicle", "registration"],
        ["vehicle", "reg"],
        ["vehicle", "reg", "no"],
        ["vehicle", "number"],
        ["vehicle", "no"],
        ["reg", "no"],
        ["reg", "number"],
        ["regn", "no"],
        ["plate"],
        ["truck", "no"],
    ],
    "reporting_datetime": [
        ["reporting", "datetime"],
        ["reporting", "date", "time"],
        ["report", "datetime"],
        ["report", "date", "time"],
        ["reporting", "dt"],
        ["report", "dt"],
    ],
    "reporting_date": [
        ["reporting", "date"],
        ["report", "date"],
        ["reporting", "dt"],
        ["report", "dt"],
    ],
    "reporting_time": [
        ["reporting", "time"],
        ["report", "time"],
        ["reporting", "tm"],
        ["report", "tm"],
    ],
    "workshop_in_datetime": [
        ["workshop", "in", "datetime"],
        ["workshop", "in", "date", "time"],
        ["gate", "in", "datetime"],
        ["gate", "in", "date", "time"],
        ["vehicle", "in", "datetime"],
        ["check", "in", "datetime"],
        ["workshop", "in", "dt"],
        ["gate", "in", "dt"],
    ],
    "workshop_in_date": [
        ["workshop", "in", "date"],
        ["gate", "in", "date"],
        ["vehicle", "in", "date"],
        ["workshop", "in", "dt"],
        ["gate", "in", "dt"],
    ],
    "workshop_in_time": [
        ["workshop", "in", "time"],
        ["gate", "in", "time"],
        ["vehicle", "in", "time"],
        ["workshop", "in", "tm"],
        ["gate", "in", "tm"],
    ],
    "workshop_out_datetime": [
        ["workshop", "out", "datetime"],
        ["workshop", "out", "date", "time"],
        ["gate", "out", "datetime"],
        ["gate", "out", "date", "time"],
        ["vehicle", "out", "datetime"],
        ["check", "out", "datetime"],
        ["workshop", "out", "dt"],
        ["gate", "out", "dt"],
    ],
    "workshop_out_date": [
        ["workshop", "out", "date"],
        ["gate", "out", "date"],
        ["vehicle", "out", "date"],
        ["workshop", "out", "dt"],
        ["gate", "out", "dt"],
    ],
    "workshop_out_time": [
        ["workshop", "out", "time"],
        ["gate", "out", "time"],
        ["vehicle", "out", "time"],
        ["workshop", "out", "tm"],
        ["gate", "out", "tm"],
    ],
}

SYSTEM_HINTS = {
    "reg": [
        ["registration"],
        ["vehicle", "registration"],
        ["vehicle", "reg"],
        ["vehicle", "reg", "no"],
        ["vehicle", "number"],
        ["vehicle", "no"],
        ["reg", "no"],
        ["reg", "number"],
        ["regn", "no"],
        ["plate"],
        ["vrn"],
    ],
    "job_card": [
        ["job", "card"],
        ["job", "card", "no"],
        ["jc"],
        ["jc", "no"],
        ["repair", "order"],
        ["repair", "order", "no"],
        ["ro", "no"],
        ["ro", "number"],
        ["order", "no"],
        ["invoice", "no"],
    ],
    "gate_in_datetime": [
        ["gate", "in", "datetime"],
        ["gate", "in", "date", "time"],
        ["vehicle", "in", "datetime"],
        ["check", "in", "datetime"],
        ["arrival", "datetime"],
        ["gate", "in", "dt"],
        ["check", "in", "dt"],
        ["arrival", "dt"],
    ],
    "gate_in_date": [
        ["gate", "in", "date"],
        ["vehicle", "in", "date"],
        ["check", "in", "date"],
        ["arrival", "date"],
        ["gate", "in", "dt"],
        ["arrival", "dt"],
    ],
    "gate_in_time": [
        ["gate", "in", "time"],
        ["vehicle", "in", "time"],
        ["check", "in", "time"],
        ["arrival", "time"],
        ["gate", "in", "tm"],
        ["arrival", "tm"],
    ],
    "reporting_datetime": [
        ["reporting", "datetime"],
        ["reporting", "date", "time"],
        ["job", "open", "datetime"],
        ["opening", "datetime"],
        ["open", "datetime"],
        ["create", "datetime"],
        ["created", "datetime"],
        ["reporting", "dt"],
        ["open", "dt"],
        ["created", "dt"],
    ],
    "reporting_date": [
        ["reporting", "date"],
        ["job", "open", "date"],
        ["opening", "date"],
        ["open", "date"],
        ["created", "date"],
        ["reporting", "dt"],
        ["open", "dt"],
        ["created", "dt"],
    ],
    "reporting_time": [
        ["reporting", "time"],
        ["job", "open", "time"],
        ["opening", "time"],
        ["open", "time"],
        ["created", "time"],
        ["reporting", "tm"],
        ["open", "tm"],
        ["created", "tm"],
    ],
    "bill_datetime": [
        ["bill", "datetime"],
        ["billing", "datetime"],
        ["invoice", "datetime"],
        ["close", "datetime"],
        ["bill", "date", "time"],
        ["invoice", "date", "time"],
        ["billing", "dt"],
        ["invoice", "dt"],
        ["close", "dt"],
    ],
    "bill_date": [
        ["bill", "date"],
        ["billing", "date"],
        ["invoice", "date"],
        ["close", "date"],
        ["billing", "dt"],
        ["invoice", "dt"],
        ["close", "dt"],
    ],
    "bill_time": [
        ["bill", "time"],
        ["billing", "time"],
        ["invoice", "time"],
        ["close", "time"],
        ["billing", "tm"],
        ["invoice", "tm"],
        ["close", "tm"],
    ],
    "rot_hours": [
        ["rot"],
        ["rot", "hours"],
        ["rot", "hrs"],
        ["repair", "time"],
        ["labour", "hours"],
        ["labor", "hours"],
        ["work", "hours"],
        ["work", "hrs"],
        ["actual", "hours"],
        ["actual", "hrs"],
    ],
}



@dataclass
class ParsedDate:
    value: Optional[date]
    ambiguous: bool = False
    unresolved: bool = False
    parse_failed: bool = False
    raw_nonempty: bool = False


@dataclass
class ParsedTime:
    value: Optional[time]
    parse_failed: bool = False
    raw_nonempty: bool = False


def list_sheets(file_path: str) -> List[str]:
    path = Path(file_path)
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        xl = pd.ExcelFile(file_path)
        return xl.sheet_names
    return []


def _read_raw_table(file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    path = Path(file_path)
    chosen_sheet = 0 if sheet_name in (None, "") else sheet_name
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(file_path, sheet_name=chosen_sheet, header=None, dtype=object, engine=None)
    return pd.read_csv(file_path, header=None, dtype=object)


def _trim_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return out.reset_index(drop=True)


def _make_unique_headers(values: Iterable[Any]) -> List[str]:
    headers: List[str] = []
    seen: Dict[str, int] = {}
    for idx, value in enumerate(values):
        raw = "" if pd.isna(value) else str(value).strip()
        if not raw:
            raw = f"Unnamed_{idx}"
        base = raw
        if base in seen:
            seen[base] += 1
            raw = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        headers.append(raw)
    return headers


def read_table(file_path: str, sheet_name: Optional[str] = None, kind: Optional[str] = None) -> pd.DataFrame:
    raw = _trim_table(_read_raw_table(file_path, sheet_name=sheet_name))
    if raw.empty:
        return pd.DataFrame()
    if kind not in {"gate", "system"}:
        kind = None
    header_row = _choose_best_header_row(raw, kind=kind)
    headers = _make_unique_headers(raw.iloc[header_row].tolist())
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = headers
    df = _trim_table(df)
    # remove fully unnamed empty columns after header promotion
    keep_cols = []
    for col in df.columns:
        ser = df[col]
        if str(col).startswith("Unnamed_") and ser.fillna("").astype(str).str.strip().eq("").all():
            continue
        keep_cols.append(col)
    df = df[keep_cols]
    return df.reset_index(drop=True)


def _norm_header(col: Any) -> str:
    text = "" if pd.isna(col) else str(col)
    text = text.replace("\n", " ").replace("\r", " ").replace("&", " and ")
    text = text.strip().lower()
    replacements = {
        r"\bregn\b": "reg",
        r"\bveh\b": "vehicle",
        r"\bvehicl\b": "vehicle",
        r"\bdt\b": "date time",
        r"\btm\b": "time",
        r"\bjc\b": "job card",
        r"\bro\b": "repair order",
        r"\bhrs\b": "hours",
        r"\bhr\b": "hours",
        r"\bno\.\b": "no",
        r"\bnum\b": "number",
    }
    for pat, repl in replacements.items():
        text = re.sub(pat, repl, text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _best_header_match(headers: List[str], pattern_groups: List[List[str]]) -> Optional[str]:
    best_col = None
    best_score = -1
    for col in headers:
        norm = _norm_header(col)
        score = 0
        for group in pattern_groups:
            hits = sum(1 for word in group if word in norm)
            if hits == len(group):
                score = max(score, len(group) * 20)
                joined = " ".join(group)
                if norm == joined:
                    score += 5
            elif hits >= max(1, len(group) - 1):
                score = max(score, hits * 6)
        if score > best_score:
            best_col = col
            best_score = score
    return best_col if best_score > 0 else None


def _registration_like_share(series: pd.Series) -> float:
    cleaned = (
        series.fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )
    if len(cleaned) == 0:
        return 0.0
    pattern_like = cleaned.str.len().between(6, 12) & cleaned.str.contains(r"[A-Z]") & cleaned.str.contains(r"\d")
    return float(pattern_like.mean())


def _job_card_like_share(series: pd.Series) -> float:
    vals = series.fillna("").astype(str).str.strip().head(100)
    vals = vals[vals != ""]
    if len(vals) == 0:
        return 0.0
    cleaned = vals.str.replace(r"[^A-Za-z0-9]", "", regex=True)
    looks = cleaned.str.len().between(6, 14) & (cleaned.str.contains(r"\d"))
    return float(looks.mean())


def _datetime_like_share(series: pd.Series) -> float:
    vals = series.fillna("").astype(str).str.strip().head(100)
    vals = vals[vals != ""]
    if len(vals) == 0:
        return 0.0
    ok = 0
    for v in vals:
        try:
            parsed = pd.to_datetime(v, errors="coerce", dayfirst=True)
            if not pd.isna(parsed):
                ok += 1
                continue
        except Exception:
            pass
        if re.search(r"\b\d{1,2}[:/-]\d{1,2}[:/-]\d{2,4}\b", v) or re.search(r"\b\d{1,2}:\d{2}(:\d{2})?\b", v):
            ok += 1
    return ok / len(vals)


def _time_like_share(series: pd.Series) -> float:
    vals = series.fillna("").astype(str).str.strip().head(100)
    vals = vals[vals != ""]
    if len(vals) == 0:
        return 0.0
    ok = 0
    for v in vals:
        if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?([ ]?[APMapm]{2})?", v):
            ok += 1
            continue
        try:
            num = float(v)
            if 0 <= num < 1:
                ok += 1
        except Exception:
            pass
    return ok / len(vals)


def _fallback_reg_column(df: pd.DataFrame) -> Optional[str]:
    best_col = None
    best_score = 0.0
    for col in df.columns:
        share = _registration_like_share(df[col])
        if share > best_score:
            best_col = col
            best_score = share
    return best_col if best_score >= 0.2 else None


def _fallback_job_card_column(df: pd.DataFrame, reg_col: Optional[str] = None) -> Optional[str]:
    best_col = None
    best_score = 0.0
    for col in df.columns:
        if reg_col is not None and col == reg_col:
            continue
        score = _job_card_like_share(df[col])
        if score > best_score:
            best_col = col
            best_score = score
    return best_col if best_score >= 0.2 else None


def _fallback_numeric_hours_column(df: pd.DataFrame) -> Optional[str]:
    best_col = None
    best_score = 0.0
    for col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        share = float(vals.notna().mean())
        if share > best_score:
            best_col = col
            best_score = share
    return best_col if best_score >= 0.4 else None


def _header_candidate_score(raw: pd.DataFrame, header_row: int, kind: Optional[str]) -> float:
    headers = _make_unique_headers(raw.iloc[header_row].tolist())
    cand = raw.iloc[header_row + 1 :].copy()
    if cand.empty:
        return -1.0
    cand.columns = headers
    cand = _trim_table(cand)
    if cand.empty:
        return -1.0

    hints = GATE_HINTS if kind == "gate" else SYSTEM_HINTS if kind == "system" else {**GATE_HINTS, **SYSTEM_HINTS}
    mapping = detect_columns(cand.head(100), hints, kind=kind or "gate", validate_only=True)
    score = 0.0
    if mapping.get("reg"):
        score += 120 + _registration_like_share(cand[mapping["reg"]]) * 100
    if kind == "system":
        if mapping.get("job_card"):
            score += 70 + _job_card_like_share(cand[mapping["job_card"]]) * 40
        if mapping.get("gate_in_datetime") or mapping.get("gate_in_date"):
            score += 30
        if mapping.get("reporting_datetime") or mapping.get("reporting_date"):
            score += 20
        if mapping.get("bill_datetime") or mapping.get("bill_date"):
            score += 10
    elif kind == "gate":
        for key in ("reporting_date", "reporting_datetime", "workshop_in_date", "workshop_in_datetime", "workshop_out_date", "workshop_out_datetime"):
            if mapping.get(key):
                score += 20
    header_text = " ".join(_norm_header(h) for h in headers)
    score += sum(2 for term in ("reg", "registration", "job card", "reporting", "workshop", "gate in", "gate out", "bill", "invoice", "rot") if term in header_text)
    # prefer earlier header rows when close
    score -= header_row * 0.5
    return score


def _choose_best_header_row(raw: pd.DataFrame, kind: Optional[str]) -> int:
    max_rows = min(len(raw), 8)
    best_row = 0
    best_score = float("-inf")
    for row_idx in range(max_rows):
        score = _header_candidate_score(raw, row_idx, kind)
        if score > best_score:
            best_score = score
            best_row = row_idx
    return best_row


def _prefer_split_date_time_mapping(mapping: Dict[str, Optional[str]], bases: Iterable[str]) -> Dict[str, Optional[str]]:
    mapping = dict(mapping)
    for base in bases:
        dt_key = f"{base}_datetime"
        d_key = f"{base}_date"
        t_key = f"{base}_time"
        if mapping.get(dt_key) is not None and mapping.get(t_key) is not None:
            if mapping.get(dt_key) == mapping.get(d_key) or mapping.get(dt_key) == mapping.get(t_key):
                mapping[dt_key] = None
    return mapping


def _mapping_value_valid(df: pd.DataFrame, col: Optional[str], target: str) -> bool:
    if col is None or col not in df.columns:
        return False
    if target == "reg":
        return _registration_like_share(df[col]) >= 0.15
    if target == "job_card":
        return _job_card_like_share(df[col]) >= 0.15
    if target == "rot_hours":
        return float(pd.to_numeric(df[col], errors="coerce").notna().mean()) >= 0.2
    if target.endswith("_datetime") or target.endswith("_date"):
        return _datetime_like_share(df[col]) >= 0.12
    if target.endswith("_time"):
        return _time_like_share(df[col]) >= 0.12
    return True


def detect_columns(df: pd.DataFrame, hints: Dict[str, List[List[str]]], kind: str, validate_only: bool = False) -> Dict[str, Optional[str]]:
    headers = list(df.columns)
    mapping: Dict[str, Optional[str]] = {}
    for target, pattern_groups in hints.items():
        cand = _best_header_match(headers, pattern_groups)
        if validate_only:
            mapping[target] = cand
        else:
            mapping[target] = cand if _mapping_value_valid(df, cand, target) else None

    if not mapping.get("reg"):
        mapping["reg"] = _fallback_reg_column(df)

    if kind == "system":
        if not mapping.get("job_card"):
            mapping["job_card"] = _fallback_job_card_column(df, reg_col=mapping.get("reg"))
        if not mapping.get("rot_hours"):
            mapping["rot_hours"] = _fallback_numeric_hours_column(df)

    return mapping

def is_strict_reg(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{2}\d{2}[A-Z]{2}\d{4}", value or ""))


def _best_reg_substring(s: str) -> str:
    if len(s) <= 10:
        return s
    best = s[:10]
    best_score = -1
    for i in range(len(s) - 9):
        sub = s[i:i+10]
        score = 0
        for idx, ch in enumerate(sub):
            if idx in ALPHA_POS and ch.isalpha():
                score += 2
            elif idx in NUM_POS and ch.isdigit():
                score += 2
            elif ch.isalnum():
                score += 1
        if score > best_score:
            best = sub
            best_score = score
    return best


def _positional_reg_fix(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # Only apply positional OCR correction when we truly have a 10-character candidate.
    # For shorter malformed registrations, preserve the cleaned text rather than forcing letters/numbers.
    if len(s) != 10:
        return s
    chars = list(s)
    for idx, ch in enumerate(chars):
        if idx in ALPHA_POS:
            if ch.isdigit():
                chars[idx] = OCR_NUM_TO_ALPHA.get(ch, ch)
        elif idx in NUM_POS:
            if ch.isalpha():
                chars[idx] = OCR_ALPHA_TO_NUM.get(ch, ch)
    return "".join(chars)

def clean_registration(raw: Any) -> Tuple[str, str, bool, bool, bool]:
    raw_text = "" if pd.isna(raw) else str(raw).strip()
    raw_alnum = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()

    # Match agent behavior:
    # - preserve the cleaned alphanumeric text as-is for overlength values
    # - apply positional OCR correction only when the candidate is already 10 characters
    # - do not force-substring long values like RJ021GD4816
    cleaned = raw_alnum
    if len(cleaned) == 10:
        cleaned = _positional_reg_fix(cleaned)

    final = cleaned
    strict_before = is_strict_reg(cleaned)
    strict_after = strict_before
    format_cleaned = (
        (cleaned != raw_alnum)
        or (raw_text != raw_text.upper())
        or bool(re.search(r"[^A-Za-z0-9]", raw_text))
    )
    return cleaned, final, strict_before, strict_after, format_cleaned

def _is_nonempty(value: Any) -> bool:
    return value is not None and not (pd.isna(value)) and str(value).strip() != ""


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if value is pd.NaT:
            return True
    except Exception:
        pass
    if isinstance(value, str):
        txt = value.strip().upper()
        if txt in {"", "NAT", "NAN", "NONE", "NULL"}:
            return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _normalize_datetime_like(value: Any) -> Optional[datetime]:
    if _is_missing(value):
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.to_pydatetime().replace(microsecond=0)
    if isinstance(value, datetime):
        if pd.isna(value):
            return None
        return value.replace(microsecond=0)
    if isinstance(value, np.datetime64):
        try:
            ts = pd.to_datetime(value, errors="coerce")
            if pd.isna(ts):
                return None
            return ts.to_pydatetime().replace(microsecond=0)
        except Exception:
            return None
    return None


def _safe_date_like(value: Any) -> Optional[date]:
    if _is_missing(value):
        return None
    try:
        dt = _normalize_datetime_like(value)
        if dt is not None:
            try:
                return date(dt.year, dt.month, dt.day)
            except Exception:
                return None
        if isinstance(value, date):
            try:
                return date(int(value.year), int(value.month), int(value.day))
            except Exception:
                return None
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return None
        py_dt = ts.to_pydatetime()
        try:
            return date(py_dt.year, py_dt.month, py_dt.day)
        except Exception:
            return None
    except Exception:
        return None


def _safe_time_like(value: Any) -> Optional[time]:
    if _is_missing(value):
        return None
    try:
        dt = _normalize_datetime_like(value)
        if dt is not None:
            try:
                return time(dt.hour, dt.minute, dt.second)
            except Exception:
                return None
        if isinstance(value, time):
            try:
                return time(int(value.hour), int(value.minute), int(value.second))
            except Exception:
                return None
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return None
        py_dt = ts.to_pydatetime()
        try:
            return time(py_dt.hour, py_dt.minute, py_dt.second)
        except Exception:
            return None
    except Exception:
        return None


def _safe_dt_date(value: Any) -> Optional[date]:
    return _safe_date_like(value)


def _safe_dt_time(value: Any) -> Optional[time]:
    return _safe_time_like(value)


def _try_excel_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    try:
        if float(value) > 1:
            return pd.to_datetime(float(value), unit="D", origin="1899-12-30").to_pydatetime()
    except Exception:
        return None
    return None


def _time_from_fraction(value: Any) -> Optional[time]:
    try:
        f = float(value)
    except Exception:
        return None
    if 0 <= f < 1:
        total_seconds = int(round(f * 24 * 60 * 60))
        total_seconds %= 24 * 60 * 60
        hh = total_seconds // 3600
        mm = (total_seconds % 3600) // 60
        ss = total_seconds % 60
        return time(hh, mm, ss)
    return None


def _within_operating_window(d: date) -> bool:
    md = (d.month, d.day)
    return OPERATING_START <= md <= OPERATING_END


def _format_date(d: Optional[date]) -> Optional[str]:
    try:
        d2 = _safe_date_like(d)
        if d2 is None or _is_missing(d2):
            return None
        return f"{int(d2.month):02d}/{int(d2.day):02d}/{int(d2.year):04d}"
    except Exception:
        return None


def _format_time(t: Optional[time]) -> Optional[str]:
    try:
        t2 = _safe_time_like(t)
        if t2 is None or _is_missing(t2):
            return None
        return f"{int(t2.hour):02d}:{int(t2.minute):02d}:{int(t2.second):02d}"
    except Exception:
        return None


def _datetime_helper_selftest() -> bool:
    return _format_date(pd.NaT) is None and _format_time(pd.NaT) is None


def _combine_date_time(d: Optional[date], t: Optional[time]) -> Optional[datetime]:
    d2 = _safe_date_like(d)
    t2 = _safe_time_like(t)
    if d2 is None or t2 is None:
        return None
    return datetime.combine(d2, t2)


def _parse_gate_date_candidate(value: Any) -> ParsedDate:
    if not _is_nonempty(value):
        return ParsedDate(None, raw_nonempty=False)

    normalized_dt = _normalize_datetime_like(value)
    if normalized_dt is not None:
        return ParsedDate(normalized_dt.date(), raw_nonempty=True)
    if isinstance(value, date):
        return ParsedDate(value, raw_nonempty=True)

    excel_dt = _try_excel_datetime(value)
    if excel_dt is not None:
        return ParsedDate(excel_dt.date(), raw_nonempty=True)

    text = str(value).strip()
    if not text:
        return ParsedDate(None, raw_nonempty=False)

    # if a time got mixed in the date field
    m = re.search(r"(\d{1,2})[\/\-\.:](\d{1,2})[\/\-\.:](\d{2,4})", text)
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        y = int(m.group(3))
        if y < 100:
            y += 2000
        candidates: List[date] = []
        for month, day in [(a, b), (b, a)]:
            try:
                candidates.append(date(y, month, day))
            except Exception:
                pass
        if len(candidates) == 1:
            return ParsedDate(candidates[0], raw_nonempty=True)
        if len(candidates) == 2:
            c1, c2 = candidates
            in_window = [_within_operating_window(c1), _within_operating_window(c2)]
            if in_window[0] and not in_window[1]:
                return ParsedDate(c1, raw_nonempty=True, ambiguous=True)
            if in_window[1] and not in_window[0]:
                return ParsedDate(c2, raw_nonempty=True, ambiguous=True)
            # unresolved for later neighbor smoothing
            return ParsedDate(c1, ambiguous=True, unresolved=True, raw_nonempty=True)

    for dayfirst in (False, True):
        try:
            dt = dateparser.parse(text, dayfirst=dayfirst, fuzzy=True, default=datetime(2024, 11, 1))
            if dt:
                return ParsedDate(dt.date(), raw_nonempty=True)
        except Exception:
            pass
    return ParsedDate(None, parse_failed=True, raw_nonempty=True)


def resolve_gate_date_series(series: pd.Series) -> Tuple[List[Optional[date]], List[bool], List[bool]]:
    initial = [_parse_gate_date_candidate(v) for v in series.tolist()]
    resolved = [p.value for p in initial]
    parse_fail_flags = [bool(p.parse_failed or (p.unresolved and p.value is None)) for p in initial]
    ambiguous_unresolved_flags = [False] * len(initial)

    # derive neighbor anchors from non-ambiguous rows
    anchor_indices = [i for i, p in enumerate(initial) if p.value is not None and not p.unresolved]

    for idx, p in enumerate(initial):
        if not p.unresolved:
            continue
        raw = str(series.iloc[idx]).strip()
        m = re.search(r"(\d{1,2})[\/\-\.:](\d{1,2})[\/\-\.:](\d{2,4})", raw)
        if not m:
            ambiguous_unresolved_flags[idx] = True
            continue
        a = int(m.group(1))
        b = int(m.group(2))
        y = int(m.group(3))
        if y < 100:
            y += 2000
        candidates: List[date] = []
        for month, day in [(a, b), (b, a)]:
            try:
                candidates.append(date(y, month, day))
            except Exception:
                pass
        if not candidates:
            parse_fail_flags[idx] = True
            ambiguous_unresolved_flags[idx] = True
            resolved[idx] = None
            continue
        if len(candidates) == 1:
            resolved[idx] = candidates[0]
            continue

        # local sequential smoothing with nearby anchors
        neighbor_dates: List[date] = []
        for j in range(max(0, idx - 3), min(len(initial), idx + 4)):
            if j == idx:
                continue
            if resolved[j] is not None and not initial[j].unresolved:
                neighbor_dates.append(resolved[j])

        if neighbor_dates:
            scores = []
            for cand in candidates:
                # smaller is better; keep rows sequential-ish
                score = sum(abs((cand - nd).days) for nd in neighbor_dates)
                # prefer operating window
                if not _within_operating_window(cand):
                    score += 25
                scores.append(score)
            best_idx = int(np.argmin(scores))
            resolved[idx] = candidates[best_idx]
            ambiguous_unresolved_flags[idx] = scores.count(scores[best_idx]) > 1
        else:
            # if no anchors, prefer candidate inside operating window
            in_window = [_within_operating_window(c) for c in candidates]
            if in_window[0] and not in_window[1]:
                resolved[idx] = candidates[0]
            elif in_window[1] and not in_window[0]:
                resolved[idx] = candidates[1]
            else:
                resolved[idx] = candidates[0]
                ambiguous_unresolved_flags[idx] = True

    return resolved, parse_fail_flags, ambiguous_unresolved_flags


def parse_time_value(value: Any) -> ParsedTime:
    if not _is_nonempty(value):
        return ParsedTime(None, raw_nonempty=False)

    normalized_dt = _normalize_datetime_like(value)
    if normalized_dt is not None:
        return ParsedTime(normalized_dt.time().replace(microsecond=0), raw_nonempty=True)
    if isinstance(value, time):
        return ParsedTime(value.replace(microsecond=0), raw_nonempty=True)

    maybe_time = _time_from_fraction(value)
    if maybe_time is not None:
        return ParsedTime(maybe_time, raw_nonempty=True)

    text = str(value).strip()
    if not text:
        return ParsedTime(None, raw_nonempty=False)

    # handle values like "7.30 PM"
    text2 = re.sub(r"(?<=\d)\.(?=\d{2}\b)", ":", text)
    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%I:%M %p", "%I %p"):
        try:
            return ParsedTime(datetime.strptime(text2, fmt).time(), raw_nonempty=True)
        except Exception:
            pass
    try:
        dt = dateparser.parse(text2, fuzzy=True)
        if dt is not None:
            return ParsedTime(dt.time(), raw_nonempty=True)
    except Exception:
        pass
    return ParsedTime(None, parse_failed=True, raw_nonempty=True)


def _combine_from_columns(df: pd.DataFrame, date_col: Optional[str], time_col: Optional[str], dt_col: Optional[str], gate_dates: bool = False):
    dates: List[Optional[date]] = [None] * len(df)
    date_fail_flags = [False] * len(df)
    date_amb_flags = [False] * len(df)
    times: List[Optional[time]] = [None] * len(df)
    time_fail_flags = [False] * len(df)

    if dt_col:
        dt_series = df[dt_col]
        split_dates: List[Optional[date]] = []
        split_times: List[Optional[time]] = []
        for v in dt_series.tolist():
            if not _is_nonempty(v):
                split_dates.append(None)
                split_times.append(None)
                continue
            excel_dt = _try_excel_datetime(v)
            if excel_dt is not None:
                split_dates.append(excel_dt.date())
                split_times.append(excel_dt.time().replace(microsecond=0))
                continue
            text = str(v).strip()
            if not text:
                split_dates.append(None)
                split_times.append(None)
                continue
            try:
                if gate_dates:
                    # date part may still be ambiguous
                    p = _parse_gate_date_candidate(text)
                    # parse time separately from the same string
                    t = parse_time_value(text)
                    split_dates.append(p.value)
                    split_times.append(t.value)
                else:
                    dt = pd.to_datetime(text, dayfirst=True, errors="coerce")
                    if pd.isna(dt):
                        # strict system format DD:MM:YYYY HH:MM:SS fallback
                        dt = pd.to_datetime(text, format="%d:%m:%Y %H:%M:%S", errors="coerce")
                    if pd.isna(dt):
                        split_dates.append(None)
                        split_times.append(None)
                    else:
                        split_dates.append(dt.to_pydatetime().date())
                        split_times.append(dt.to_pydatetime().time().replace(microsecond=0))
            except Exception:
                split_dates.append(None)
                split_times.append(None)
        if gate_dates:
            # resolve ambiguous numeric dates using same smoother
            dates, date_fail_flags, date_amb_flags = resolve_gate_date_series(pd.Series(split_dates))
        else:
            dates = split_dates
            date_fail_flags = [False] * len(df)
            date_amb_flags = [False] * len(df)
        times = split_times
        time_fail_flags = [False if t is not None else False for t in split_times]
    else:
        if date_col:
            if gate_dates:
                dates, date_fail_flags, date_amb_flags = resolve_gate_date_series(df[date_col])
            else:
                split_dates = []
                fail_flags = []
                for v in df[date_col].tolist():
                    if not _is_nonempty(v):
                        split_dates.append(None)
                        fail_flags.append(False)
                        continue
                    if isinstance(v, (datetime, pd.Timestamp, np.datetime64)):
                        safe_d = _safe_date_like(v)
                        split_dates.append(safe_d)
                        fail_flags.append(safe_d is None and _is_nonempty(v))
                        continue
                    if isinstance(v, date):
                        split_dates.append(v)
                        fail_flags.append(False)
                        continue
                    excel_dt = _try_excel_datetime(v)
                    if excel_dt is not None:
                        split_dates.append(excel_dt.date())
                        fail_flags.append(False)
                        continue
                    try:
                        dt = pd.to_datetime(str(v), dayfirst=True, errors="coerce")
                        if pd.isna(dt):
                            dt = pd.to_datetime(str(v), errors="coerce")
                        if pd.isna(dt):
                            split_dates.append(None)
                            fail_flags.append(True)
                        else:
                            split_dates.append(dt.to_pydatetime().date())
                            fail_flags.append(False)
                    except Exception:
                        split_dates.append(None)
                        fail_flags.append(True)
                dates = split_dates
                date_fail_flags = fail_flags
                date_amb_flags = [False] * len(df)
        if time_col:
            parsed = [parse_time_value(v) for v in df[time_col].tolist()]
            times = [p.value for p in parsed]
            time_fail_flags = [p.parse_failed for p in parsed]

    dts = [_combine_date_time(d, t) for d, t in zip(dates, times)]
    return dates, times, dts, date_fail_flags, date_amb_flags, time_fail_flags


def _safe_hours(diff: Optional[timedelta]) -> Optional[float]:
    if _is_missing(diff):
        return None
    try:
        seconds = diff.total_seconds()
    except Exception:
        return None
    if _is_missing(seconds):
        return None
    return round(float(seconds) / 3600.0, 4)


def _time_diff_hours(a: Optional[datetime], b: Optional[datetime]) -> Optional[float]:
    a = _normalize_datetime_like(a)
    b = _normalize_datetime_like(b)
    if a is None or b is None:
        return None
    return round(abs((a - b).total_seconds()) / 3600.0, 4)


def _to_float(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return float(str(value).strip())
    except Exception:
        return None


def standardize_gate(gate_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    df = gate_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    mapping = detect_columns(df, GATE_HINTS, kind="gate")
    mapping = _prefer_split_date_time_mapping(mapping, ["reporting", "workshop_in", "workshop_out"])

    reg_col = mapping.get("reg")
    cleaned_info = [clean_registration(v) for v in df[reg_col].tolist()] if reg_col else [("", "", False, False, False) for _ in range(len(df))]

    rep_dates, rep_times, rep_dt, rep_dfail, rep_damb, rep_tfail = _combine_from_columns(
        df, mapping.get("reporting_date"), mapping.get("reporting_time"), mapping.get("reporting_datetime"), gate_dates=True
    )
    win_dates, win_times, win_dt, win_dfail, win_damb, win_tfail = _combine_from_columns(
        df, mapping.get("workshop_in_date"), mapping.get("workshop_in_time"), mapping.get("workshop_in_datetime"), gate_dates=True
    )
    wout_dates, wout_times, wout_dt, wout_dfail, wout_damb, wout_tfail = _combine_from_columns(
        df, mapping.get("workshop_out_date"), mapping.get("workshop_out_time"), mapping.get("workshop_out_datetime"), gate_dates=True
    )

    rows = []
    for i in range(len(df)):
        cleaned, final, strict_before, strict_after, fmt_cleaned = cleaned_info[i]
        rep_dt_i = _normalize_datetime_like(rep_dt[i])
        win_dt_i = _normalize_datetime_like(win_dt[i])
        wout_dt_i = _normalize_datetime_like(wout_dt[i])
        gate_in_source = "Workshop In" if win_dt_i is not None else ("Reporting" if rep_dt_i is not None else None)
        base_gate_in_dt = win_dt_i if win_dt_i is not None else rep_dt_i
        timeline_anchor = _safe_dt_date(base_gate_in_dt) or rep_dates[i] or win_dates[i] or wout_dates[i]
        any_gate_parse_fail = any([
            rep_dfail[i], rep_damb[i],
            win_dfail[i], win_damb[i],
            wout_dfail[i], wout_damb[i],
            rep_tfail[i], win_tfail[i], wout_tfail[i],
        ])
        rows.append(
            {
                "__gate_idx__": i,
                "Gate_Reg_No_Raw": None if reg_col is None else df.iloc[i][reg_col],
                "Gate_Reg_No_Cleaned": cleaned,
                "Gate_Reg_No_Final": final,
                "Gate_Reg_Strict_Format_Flag": "Yes" if strict_before else "No",
                "Final_Reg_Strict_Format_Flag": "Yes" if strict_after else "No",
                "Reg_Format_Cleaned_Flag": "Yes" if fmt_cleaned else "No",
                "Reg_Corrected_via_System_Flag": "No",
                "Reporting_Date_Clean": _format_date(rep_dates[i]),
                "Reporting_Time_Clean": _format_time(rep_times[i]),
                "Workshop_In_Date_Clean": _format_date(win_dates[i]),
                "Workshop_In_Time_Clean": _format_time(win_times[i]),
                "Workshop_Out_Date_Clean": _format_date(wout_dates[i]),
                "Workshop_Out_Time_Clean": _format_time(wout_times[i]),
                "__reporting_dt__": rep_dt_i,
                "__workshop_in_dt__": win_dt_i,
                "__workshop_out_dt__": wout_dt_i,
                "__base_gate_in_dt__": base_gate_in_dt,
                "__gate_in_source__": gate_in_source,
                "__timeline_anchor_date__": timeline_anchor,
                "__gate_parse_fail__": any_gate_parse_fail,
            }
        )

    return pd.DataFrame(rows), mapping

def _parse_system_datetime_text(value: Any) -> Optional[datetime]:
    normalized_dt = _normalize_datetime_like(value)
    if normalized_dt is not None:
        return normalized_dt
    if not _is_nonempty(value):
        return None
    excel_dt = _try_excel_datetime(value)
    if excel_dt is not None:
        return excel_dt
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%d:%m:%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%d-%m-%Y %H:%M:%S", "%d:%m:%Y %H:%M", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            pass
    # broader fallback for imperfect real-world data
    for dayfirst in (True, False):
        try:
            dt = dateparser.parse(text, dayfirst=dayfirst, fuzzy=True)
            if dt is not None:
                return dt
        except Exception:
            pass
    return None


def _combine_system_datetime(df: pd.DataFrame, date_col: Optional[str], time_col: Optional[str], dt_col: Optional[str]):
    dts: List[Optional[datetime]] = []
    parse_fail_flags: List[bool] = []
    for i in range(len(df)):
        if dt_col:
            dt = _parse_system_datetime_text(df.iloc[i][dt_col])
            if dt is None and _is_nonempty(df.iloc[i][dt_col]):
                parse_fail_flags.append(True)
            else:
                parse_fail_flags.append(False)
            dts.append(dt)
        else:
            d = None
            t = None
            fail = False
            if date_col and _is_nonempty(df.iloc[i][date_col]):
                dt_date = _parse_system_datetime_text(str(df.iloc[i][date_col]) + " 00:00:00")
                if dt_date is not None:
                    d = _safe_dt_date(dt_date)
                else:
                    try:
                        d_parsed = pd.to_datetime(str(df.iloc[i][date_col]), dayfirst=True, errors="coerce")
                        if pd.isna(d_parsed):
                            fail = True
                        else:
                            d = d_parsed.to_pydatetime().date()
                    except Exception:
                        fail = True
            if time_col and _is_nonempty(df.iloc[i][time_col]):
                t_parsed = parse_time_value(df.iloc[i][time_col])
                if t_parsed.value is not None:
                    t = t_parsed.value
                elif t_parsed.raw_nonempty:
                    fail = True
            dts.append(_combine_date_time(d, t))
            parse_fail_flags.append(fail)
    dates = [_safe_dt_date(dt) for dt in dts]
    times = [_safe_dt_time(dt) for dt in dts]
    return dates, times, dts, parse_fail_flags


def standardize_system(sys_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    df = sys_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    mapping = detect_columns(df, SYSTEM_HINTS, kind="system")
    mapping = _prefer_split_date_time_mapping(mapping, ["gate_in", "reporting", "bill"])

    reg_col = mapping.get("reg")
    jc_col = mapping.get("job_card")
    rot_col = mapping.get("rot_hours")

    cleaned_info = [clean_registration(v) for v in df[reg_col].tolist()] if reg_col else [("", "", False, False, False) for _ in range(len(df))]
    gate_dates, gate_times, gate_dt, gate_fail = _combine_system_datetime(df, mapping.get("gate_in_date"), mapping.get("gate_in_time"), mapping.get("gate_in_datetime"))
    rep_dates, rep_times, rep_dt, rep_fail = _combine_system_datetime(df, mapping.get("reporting_date"), mapping.get("reporting_time"), mapping.get("reporting_datetime"))
    bill_dates, bill_times, bill_dt, bill_fail = _combine_system_datetime(df, mapping.get("bill_date"), mapping.get("bill_time"), mapping.get("bill_datetime"))

    rows = []
    for i in range(len(df)):
        cleaned, final, _, strict_after, _ = cleaned_info[i]
        job_card = None if jc_col is None else df.iloc[i][jc_col]
        rot_hours = _to_float(df.iloc[i][rot_col]) if rot_col else None
        gate_dt_i = _normalize_datetime_like(gate_dt[i])
        rep_dt_i = _normalize_datetime_like(rep_dt[i])
        bill_dt_i = _normalize_datetime_like(bill_dt[i])
        timeline_anchor = _safe_dt_date(gate_dt_i) or _safe_dt_date(rep_dt_i) or _safe_dt_date(bill_dt_i)

        # Keep separate "any parse issue" vs "matching blocked".
        # Bill parse issues should not destroy otherwise valid matching.
        parse_fail_any = bool(gate_fail[i] or rep_fail[i] or bill_fail[i])
        matching_blocked = False
        if not (final or cleaned):
            matching_blocked = True
        elif gate_fail[i] and rep_fail[i]:
            matching_blocked = True
        elif gate_dt_i is None and rep_dt_i is None and (gate_fail[i] or rep_fail[i]):
            matching_blocked = True

        rows.append(
            {
                "__sys_idx__": i,
                "Matched_Job_Card_No": job_card,
                "Matched_System_Reg_No": final if final else cleaned,
                "__system_reg_raw__": None if reg_col is None else df.iloc[i][reg_col],
                "__system_reg_cleaned__": cleaned,
                "__system_reg_final__": final if final else cleaned,
                "__system_reg_strict__": strict_after,
                "__system_gate_in_dt__": gate_dt_i,
                "__system_reporting_dt__": rep_dt_i,
                "__system_bill_dt__": bill_dt_i,
                "Matched_System_Gate_In_Date": _format_date(gate_dates[i]),
                "Matched_System_Gate_In_Time": _format_time(gate_times[i]),
                "Matched_System_Bill_Date": _format_date(bill_dates[i]),
                "Matched_System_Bill_Time": _format_time(bill_times[i]),
                "ROT_Hours": rot_hours,
                "__system_parse_fail__": parse_fail_any,
                "__system_matching_blocked__": matching_blocked,
                "__timeline_anchor_date__": timeline_anchor,
                "__system_order__": i,
            }
        )
    return pd.DataFrame(rows), mapping

def _candidate_time_anchor(gate_row: pd.Series) -> Optional[datetime]:
    base_dt = _normalize_datetime_like(gate_row["__base_gate_in_dt__"])
    return base_dt if base_dt is not None else _normalize_datetime_like(gate_row["__reporting_dt__"])


def _system_time_anchor(sys_row: pd.Series) -> Optional[datetime]:
    gate_dt = _normalize_datetime_like(sys_row["__system_gate_in_dt__"])
    return gate_dt if gate_dt is not None else _normalize_datetime_like(sys_row["__system_reporting_dt__"])


def _registration_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a = str(a).upper()
    b = str(b).upper()
    return float(fuzz.ratio(a, b))

def _bill_diff_hours(gate_row: pd.Series, sys_row: pd.Series) -> Optional[float]:
    gate_out = gate_row["__workshop_out_dt__"]
    sys_bill = sys_row["__system_bill_dt__"]
    return _time_diff_hours(gate_out, sys_bill)


def classify_match_type(score: Optional[float], exact: bool, promoted: bool = False) -> str:
    if score is None:
        return "No Match"
    if exact:
        return "Exact"
    if score >= 90:
        return "Fuzzy_High"
    if score >= 80:
        return "Fuzzy_Medium"
    return "No Match"

def build_candidate_table(gate_df: pd.DataFrame, sys_df: pd.DataFrame, purpose: str = "match") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, g in gate_df.iterrows():
        g_reg = g["Gate_Reg_No_Final"] or g["Gate_Reg_No_Cleaned"]
        if not g_reg:
            continue
        g_anchor = _candidate_time_anchor(g)

        for _, s in sys_df.iterrows():
            if s.get("__system_matching_blocked__", False):
                continue
            s_reg = s["__system_reg_final__"] or s["__system_reg_cleaned__"]
            if not s_reg:
                continue

            sim = _registration_similarity(g_reg, s_reg)
            exact = (g_reg == s_reg and g_reg != "")
            time_diff = _time_diff_hours(g_anchor, _system_time_anchor(s))
            bill_diff = _bill_diff_hours(g, s)

            if purpose == "match":
                valid = False
                if exact:
                    valid = time_diff is not None and float(time_diff) <= 48
                else:
                    valid = sim >= 80 and time_diff is not None and float(time_diff) <= 48
                if not valid:
                    continue
            else:
                # Suggestions should mirror the agent's behavior:
                # keep a top suggestion for every gate row, even when the time gap is large
                # or the similarity is below matching threshold.
                if sim <= 0 and not exact:
                    continue

            reward = sim * 10000.0
            if time_diff is not None:
                reward -= float(time_diff) * 100.0
            if bill_diff is not None:
                reward -= float(bill_diff)
            if exact:
                reward += 50.0
            # deterministic tie-break toward earlier Gate Register occurrence.
            reward -= float(g["__gate_idx__"]) * 1e-6

            rows.append(
                {
                    "gate_idx": int(g["__gate_idx__"]),
                    "sys_idx": int(s["__sys_idx__"]),
                    "sim_score": float(sim),
                    "exact": exact,
                    "time_diff_hours": time_diff,
                    "bill_diff_hours": bill_diff,
                    "reward": float(reward),
                    "sys_order": int(s["__system_order__"]),
                }
            )
    return pd.DataFrame(rows)

def top_suggestions(candidate_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    suggestions: Dict[int, Dict[str, Any]] = {}
    if candidate_df.empty:
        return suggestions
    ranked = candidate_df.sort_values(
        ["gate_idx", "sim_score", "time_diff_hours", "bill_diff_hours", "sys_order"],
        ascending=[True, False, True, True, True],
        na_position="last",
    )
    for gate_idx, grp in ranked.groupby("gate_idx", sort=False):
        best = grp.iloc[0].to_dict()
        suggestions[int(gate_idx)] = best
    return suggestions


def global_one_to_one_assignment(gate_df: pd.DataFrame, sys_df: pd.DataFrame, candidate_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    n_gate = len(gate_df)
    n_sys = len(sys_df)
    assignments: Dict[int, Dict[str, Any]] = {}
    if n_gate == 0 or candidate_df.empty or n_sys == 0:
        return assignments

    cost = np.full((n_gate, n_sys + n_gate), 0.0)
    cost[:, :n_sys] = 10_000.0
    for _, row in candidate_df.iterrows():
        gi = int(row["gate_idx"])
        si = int(row["sys_idx"])
        cost[gi, si] = -float(row["reward"])

    row_ind, col_ind = linear_sum_assignment(cost)
    chosen = {(int(r), int(c)) for r, c in zip(row_ind, col_ind)}

    indexed = candidate_df.set_index(["gate_idx", "sys_idx"]).sort_index()
    for gi, ci in chosen:
        if ci >= n_sys or (gi, ci) not in indexed.index:
            continue
        rec = indexed.loc[(gi, ci)]
        if isinstance(rec, pd.DataFrame):
            rec = rec.iloc[0]
        rec_dict = rec.to_dict()
        rec_dict["gate_idx"] = gi
        rec_dict["sys_idx"] = ci
        assignments[gi] = rec_dict

    return assignments

def second_pass_promote(
    assignments: Dict[int, Dict[str, Any]],
    suggestions: Dict[int, Dict[str, Any]],
    gate_df: pd.DataFrame,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, str]]:
    reasons: Dict[int, str] = {}
    used_sys = {int(v["sys_idx"]) for v in assignments.values()}
    sys_to_gate = {int(v["sys_idx"]): g for g, v in assignments.items()}

    def _priority_tuple(cand: Dict[str, Any], gate_idx: int) -> Tuple[float, float, float, float, float]:
        return (
            float(cand.get("sim_score") or 0),
            -float(cand.get("time_diff_hours") if cand.get("time_diff_hours") is not None else 999999),
            -float(cand.get("bill_diff_hours") if cand.get("bill_diff_hours") is not None else 999999),
            -float(cand.get("sys_order") if cand.get("sys_order") is not None else 999999),
            -float(gate_idx),
        )

    for gi in range(len(gate_df)):
        if gi in assignments:
            continue
        sugg = suggestions.get(gi)
        if not sugg:
            continue
        score = float(sugg.get("sim_score") or 0)
        tdiff = sugg.get("time_diff_hours")
        if not (score > 85 and tdiff is not None and float(tdiff) <= 48):
            continue

        si = int(sugg["sys_idx"])
        if si not in used_sys:
            promoted = dict(sugg)
            promoted["promoted"] = True
            assignments[gi] = promoted
            used_sys.add(si)
            sys_to_gate[si] = gi
            reasons[gi] = "High-confidence suggestion reviewed and promoted."
            continue

        current_gate = sys_to_gate[si]
        current = assignments[current_gate]
        new_pri = _priority_tuple(sugg, gi)
        cur_pri = _priority_tuple(current, current_gate)

        if new_pri > cur_pri:
            reasons[gi] = "High-confidence suggestion reviewed and promoted."
            reasons[current_gate] = f"High-confidence suggestion reviewed but not promoted due to stronger competing row S_No {gi + 1}."
            assignments.pop(current_gate, None)
            promoted = dict(sugg)
            promoted["promoted"] = True
            assignments[gi] = promoted
            sys_to_gate[si] = gi
            continue

        if new_pri == cur_pri:
            earlier = min(gi, current_gate)
            later = max(gi, current_gate)
            owner = earlier
            if owner != current_gate:
                promoted = dict(sugg)
                promoted["promoted"] = True
                assignments.pop(current_gate, None)
                assignments[gi] = promoted
                sys_to_gate[si] = gi
                reasons[gi] = "High-confidence suggestion reviewed and promoted."
                reasons[current_gate] = f"High-confidence suggestion reviewed but not promoted due to equal competing row S_No {gi + 1} retained by earlier Gate Register occurrence."
            else:
                reasons[gi] = f"High-confidence suggestion reviewed but not promoted due to equal competing row S_No {current_gate + 1} retained by earlier Gate Register occurrence."
            continue

        reasons[gi] = f"High-confidence suggestion reviewed but not promoted due to stronger competing row S_No {current_gate + 1}."
    return assignments, reasons

def _toggle_ampm(t: Optional[time]) -> Optional[time]:
    if t is None:
        return None
    dt = datetime.combine(date(2024, 1, 1), t)
    if t.hour < 12:
        dt += timedelta(hours=12)
    else:
        dt -= timedelta(hours=12)
    return dt.time()


def _hours_value(val: Optional[float]) -> Optional[float]:
    num = _to_float(val)
    if num is None:
        return None
    rounded = round(num, 2)
    if abs(rounded) < 1e-9:
        rounded = 0.0
    return rounded


def _raw_num(val: Any) -> Optional[float]:
    num = _to_float(val)
    if num is None:
        return None
    if abs(num) < 1e-12:
        num = 0.0
    return float(num)

def _metric_num(series: pd.Series) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any():
        return float(s.mean())
    return None


def _sanitize_dataframe_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    for col in clean.columns:
        clean[col] = clean[col].apply(lambda x: None if _is_missing(x) else x)
    return clean


def _bucketize_tat(values: pd.Series) -> pd.Series:
    s = pd.to_numeric(values, errors="coerce")
    bins = [-np.inf, 2, 4, 8, 24, 48, 72, 168, np.inf]
    labels = ["<=2h", "2-4h", "4-8h", "8-24h", "24-48h", "48-72h", "72-168h", ">168h"]
    return pd.cut(s, bins=bins, labels=labels)


def _save_bar_chart(series: pd.Series, title: str, path: str):
    counts = series.value_counts(dropna=False).sort_index()
    plt.figure(figsize=(10, 4.8))
    counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _save_waterfall(summary_pairs: List[Tuple[str, float]], title: str, path: str):
    labels = [x[0] for x in summary_pairs]
    values = [x[1] for x in summary_pairs]
    plt.figure(figsize=(10, 4.8))
    pd.Series(values, index=labels).plot(kind="bar")
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _bool_to_yes_no(value: bool) -> str:
    return "Yes" if bool(value) else "No"


def _build_row_remark(
    row: Dict[str, Any],
    second_pass_reason: Optional[str],
    high_conf_rejected_reason: Optional[str],
) -> str:
    remarks: List[str] = []

    cleaned_reg = row.get("Gate_Reg_No_Cleaned")
    final_reg = row.get("Gate_Reg_No_Final")

    if row["Reg_Corrected_via_System_Flag"] == "Yes" and cleaned_reg and final_reg and cleaned_reg != final_reg:
        remarks.append(f"Registration corrected via matched system record from {cleaned_reg} to {final_reg}.")
    elif row["Reg_Format_Cleaned_Flag"] == "Yes":
        remarks.append("Registration cleaned for formatting and OCR noise.")
    else:
        remarks.append("Registration required no cleanup.")

    mt = row["Match_Type"]
    score = row.get("Match_Score")
    tdiff = row.get("Match_Time_Diff_Hours")
    jc = row.get("Matched_Job_Card_No")

    if mt == "No Match":
        remarks.append("No confident one-to-one job-card match found under exact/fuzzy time-aware rules.")
    elif mt == "Exact":
        remarks.append(f"Exact one-to-one system match confirmed with Job Card {jc}.")
    else:
        remarks.append(f"{mt} match confirmed with Job Card {jc} at score {score} and {tdiff}h gate-in difference.")

    top_score = row.get("Top_Suggestion_Score")
    top_tdiff = row.get("Top_Suggestion_Time_Diff_Hours")
    try:
        high_conf_window = top_score not in (None, "") and float(top_score) > 85 and top_tdiff not in (None, "") and float(top_tdiff) <= 48
    except Exception:
        high_conf_window = False

    if mt != "No Match":
        if second_pass_reason and "promoted" in second_pass_reason.lower():
            remarks.append("High-confidence suggestion reviewed and promoted.")
        elif high_conf_window:
            remarks.append("High-confidence suggestion reviewed and promoted.")
    else:
        if high_conf_rejected_reason:
            remarks.append(high_conf_rejected_reason)

    if row["Gate_Time_Corrected_Flag"] == "Yes":
        remarks.append("Gate-out time was corrected by AM/PM toggle.")
    elif row["TAT_Correction_Source"] == "System_BillDate_As_GateOut":
        remarks.append("Matched system bill date/time was used as corrected gate-out.")
    elif row["TAT_Correction_Source"] == "Gate_Out_AMPM_Toggled":
        remarks.append("Negative TAT was resolved using gate-out AM/PM toggle.")
    elif row["TAT_Correction_Source"] == "Unresolved_Negative_TAT":
        remarks.append("Negative TAT could not be resolved with allowed corrections.")
    else:
        remarks.append("No gate-time correction was required.")

    status = row["Business_Validation_Status"]
    if status in ("Pass", "", None):
        if row["Final_Considered_Flag"] == "Yes":
            remarks.append("No business-validation issues; row remains included in final count.")
        else:
            remarks.append("No business-validation issues; corrected TAT unavailable so row is excluded from final count.")
    else:
        if row["Final_Considered_Flag"] == "Yes":
            remarks.append(f"Business validation flagged {status}, but row remains included in final count.")
        else:
            remarks.append(f"Business validation flagged {status}; corrected TAT unavailable so row is excluded from final count.")

    return " ".join(remarks)

def build_main_output(
    gate_df: pd.DataFrame,
    sys_df: pd.DataFrame,
    assignments: Dict[int, Dict[str, Any]],
    suggestions: Dict[int, Dict[str, Any]],
    second_pass_reasons: Dict[int, str],
) -> pd.DataFrame:
    sys_lookup = sys_df.set_index("__sys_idx__", drop=False).to_dict(orient="index")
    rows: List[Dict[str, Any]] = []

    for i in range(len(gate_df)):
        g = gate_df.iloc[i].to_dict()
        out = {col: None for col in MAIN_OUTPUT_COLUMNS}
        out["S_No"] = i + 1
        for key in [
            "Gate_Reg_No_Raw", "Gate_Reg_No_Cleaned", "Gate_Reg_No_Final",
            "Gate_Reg_Strict_Format_Flag", "Final_Reg_Strict_Format_Flag",
            "Reg_Format_Cleaned_Flag", "Reg_Corrected_via_System_Flag",
            "Reporting_Date_Clean", "Reporting_Time_Clean",
            "Workshop_In_Date_Clean", "Workshop_In_Time_Clean",
            "Workshop_Out_Date_Clean", "Workshop_Out_Time_Clean",
        ]:
            out[key] = g.get(key)

        out["Gate_In_Source"] = g["__gate_in_source__"]
        out["Base_Gate_In_Date"] = _format_date(_safe_dt_date(g["__base_gate_in_dt__"]))
        out["Base_Gate_In_Time"] = _format_time(_safe_dt_time(g["__base_gate_in_dt__"]))

        sugg = suggestions.get(i)
        high_conf_rejected_reason = None
        if sugg:
            srow = sys_lookup.get(int(sugg["sys_idx"]))
            out["Top_Suggested_Job_Card_No"] = srow["Matched_Job_Card_No"] if srow else None
            out["Top_Suggested_System_Reg_No"] = srow["Matched_System_Reg_No"] if srow else None
            out["Top_Suggestion_Score"] = _raw_num(float(sugg["sim_score"]))
            out["Top_Suggestion_Time_Diff_Hours"] = _raw_num(float(sugg["time_diff_hours"])) if sugg["time_diff_hours"] is not None else None

        assn = assignments.get(i)
        matched_sys = None
        match_score = None
        match_time = None
        match_bill = None
        exact = False
        promoted = False

        if assn:
            matched_sys = sys_lookup.get(int(assn["sys_idx"]))
            match_score = float(assn["sim_score"])
            match_time = assn["time_diff_hours"]
            match_bill = assn["bill_diff_hours"]
            exact = bool(assn["exact"])
            promoted = bool(assn.get("promoted", False))
        # registration correction via matched system record:
        # only apply when the gate registration is not already strict-format
        # and the matched system registration is strict-format.
        if matched_sys and matched_sys["Matched_System_Reg_No"]:
            current_final = out["Gate_Reg_No_Final"] or ""
            system_final = matched_sys["Matched_System_Reg_No"] or ""
            gate_strict = out.get("Gate_Reg_Strict_Format_Flag") == "Yes"
            system_strict = bool(matched_sys.get("__system_reg_strict__", False))
            if system_final and current_final and system_final != current_final and (not gate_strict) and system_strict:
                out["Gate_Reg_No_Final"] = system_final
                out["Final_Reg_Strict_Format_Flag"] = "Yes"
                out["Reg_Corrected_via_System_Flag"] = "Yes"

        if matched_sys:
            out["Matched_Job_Card_No"] = matched_sys["Matched_Job_Card_No"]
            out["Matched_System_Reg_No"] = matched_sys["Matched_System_Reg_No"]
            out["Match_Type"] = classify_match_type(match_score, exact=exact, promoted=promoted)
            out["Match_Score"] = _raw_num(match_score)
            out["Match_Time_Diff_Hours"] = _raw_num(match_time)
            out["Match_Bill_Diff_Hours"] = _raw_num(match_bill) if match_bill is not None else None
            out["Matched_System_Gate_In_Date"] = matched_sys["Matched_System_Gate_In_Date"]
            out["Matched_System_Gate_In_Time"] = matched_sys["Matched_System_Gate_In_Time"]
            out["Matched_System_Bill_Date"] = matched_sys["Matched_System_Bill_Date"]
            out["Matched_System_Bill_Time"] = matched_sys["Matched_System_Bill_Time"]
            out["ROT_Hours"] = _raw_num(matched_sys["ROT_Hours"]) if matched_sys["ROT_Hours"] is not None else None
            out["System_Validation_Available_Flag"] = "Yes"
        else:
            out["Match_Type"] = "No Match"
            out["System_Validation_Available_Flag"] = "No"
            if sugg and float(sugg["sim_score"]) > 85 and sugg["time_diff_hours"] is not None and float(sugg["time_diff_hours"]) <= 48:
                high_conf_rejected_reason = second_pass_reasons.get(i) or "Not promoted in second-pass because a stronger competing row already owned the same system record."

        base_gate_in_dt = _normalize_datetime_like(g["__base_gate_in_dt__"])
        gate_out_dt = _normalize_datetime_like(g["__workshop_out_dt__"])

        original_tat = None
        if base_gate_in_dt is not None and gate_out_dt is not None:
            original_tat = _raw_num(_safe_hours(gate_out_dt - base_gate_in_dt))

        out["Gate_Register_GIGO_TAT_Hours"] = original_tat
        out["Original_TAT_Hours"] = original_tat

        corrected_gate_out_dt = gate_out_dt
        tat_corrected = False
        gate_time_corrected = False
        correction_source = None
        correction_flags: List[str] = []

        if out["Reg_Format_Cleaned_Flag"] == "Yes":
            correction_flags.append("Reg_Format_Cleaned")
        if out["Reg_Corrected_via_System_Flag"] == "Yes":
            correction_flags.append("Reg_Corrected_via_System")

        if original_tat is not None and original_tat < 0:
            gate_out_date = _safe_dt_date(gate_out_dt)
            gate_out_time = _safe_dt_time(gate_out_dt)
            toggled_time = _toggle_ampm(gate_out_time) if gate_out_time is not None else None
            toggled_dt = datetime.combine(gate_out_date, toggled_time) if gate_out_date is not None and toggled_time is not None else None
            toggled_tat = _safe_hours(toggled_dt - base_gate_in_dt) if toggled_dt is not None and base_gate_in_dt is not None else None
            if toggled_tat is not None and toggled_tat >= 0:
                corrected_gate_out_dt = toggled_dt
                tat_corrected = True
                gate_time_corrected = True
                correction_source = "Gate_Out_AMPM_Toggled"
                correction_flags.append("Gate_Out_AMPM_Toggled")
            elif matched_sys and _normalize_datetime_like(matched_sys["__system_bill_dt__"]) is not None:
                corrected_gate_out_dt = _normalize_datetime_like(matched_sys["__system_bill_dt__"])
                tat_corrected = True
                correction_source = "System_BillDate_As_GateOut"
                correction_flags.append("System_BillDate_As_GateOut")
            else:
                correction_source = "Unresolved_Negative_TAT"
                correction_flags.append("Negative_TAT_Unresolved")

        corrected_gate_out_dt = _normalize_datetime_like(corrected_gate_out_dt)
        corrected_tat = None
        if base_gate_in_dt is not None and corrected_gate_out_dt is not None:
            corrected_tat = _raw_num(_safe_hours(corrected_gate_out_dt - base_gate_in_dt))

        out["Corrected_Gate_Out_Date"] = _format_date(_safe_dt_date(corrected_gate_out_dt))
        out["Corrected_Gate_Out_Time"] = _format_time(_safe_dt_time(corrected_gate_out_dt))
        out["Corrected_TAT_Hours"] = corrected_tat
        out["TAT_Corrected_Flag"] = _bool_to_yes_no(tat_corrected)
        out["Gate_Time_Corrected_Flag"] = _bool_to_yes_no(gate_time_corrected)
        out["TAT_Correction_Source"] = correction_source

        rot_hours = _to_float(out["ROT_Hours"])
        if original_tat is not None and rot_hours is not None:
            out["GIGO_TAT_Delay_Hours"] = _raw_num(original_tat - rot_hours - 4)
        else:
            out["GIGO_TAT_Delay_Hours"] = None

        outlier = False
        rot_mismatch = False
        business_msgs: List[str] = []
        if corrected_tat is None:
            out["Gate_In_Before_Gate_Out_Flag"] = None
            out["Outlier_Flag"] = None
            out["ROT_Work_Mismatch_Flag"] = None
            business_msgs.append("Corrected TAT unavailable")
        else:
            gate_before_out = corrected_tat >= 0
            out["Gate_In_Before_Gate_Out_Flag"] = _bool_to_yes_no(gate_before_out)
            if corrected_tat < 0:
                business_msgs.append("Gate In after Gate Out")
            if corrected_tat < 0.25:
                outlier = True
                business_msgs.append("TAT below 0.25 hours")
                if rot_hours is not None and rot_hours > corrected_tat:
                    rot_mismatch = True
                    business_msgs.append("ROT exceeds TAT for low-duration job")
            if corrected_tat > 336:
                outlier = True
                business_msgs.append("TAT above 336 hours")
            out["Outlier_Flag"] = _bool_to_yes_no(outlier)
            out["ROT_Work_Mismatch_Flag"] = _bool_to_yes_no(rot_mismatch)

        out["Business_Validation_Status"] = "Pass" if business_msgs == [] else "; ".join(business_msgs)
        out["Final_Considered_Flag"] = "Yes" if corrected_tat is not None else "No"

        if g["__gate_parse_fail__"]:
            correction_flags.append("Gate_DateTime_Parse_Issue")
        if outlier:
            correction_flags.append("Outlier_Flag")
        if rot_mismatch:
            correction_flags.append("ROT_Work_Mismatch_Flag")

        out["Any_Correction_Flag"] = _bool_to_yes_no(bool(correction_flags))
        out["Correction_Flags"] = "; ".join(correction_flags) if correction_flags else None

        out["Final_Remarks"] = _build_row_remark(
            out,
            second_pass_reason=second_pass_reasons.get(i),
            high_conf_rejected_reason=high_conf_rejected_reason,
        )

        rows.append(out)

    main_df = pd.DataFrame(rows, columns=MAIN_OUTPUT_COLUMNS)
    return main_df

def build_summary(gate_df: pd.DataFrame, sys_df: pd.DataFrame, main_df: pd.DataFrame) -> pd.DataFrame:
    gate_timeline_mask = gate_df["__timeline_anchor_date__"].apply(lambda d: _within_operating_window(d) if isinstance(d, date) else False)
    sys_timeline_mask = sys_df["__timeline_anchor_date__"].apply(lambda d: _within_operating_window(d) if isinstance(d, date) else False)

    match_dist = main_df["Match_Type"].fillna("No Match").value_counts()
    score_100_share = None
    matched_scores = pd.to_numeric(main_df["Match_Score"], errors="coerce")
    if matched_scores.notna().any():
        score_100_share = round(float((matched_scores == 100).mean()) * 100, 2)

    matched_mask = main_df["Matched_Job_Card_No"].fillna("").astype(str).str.strip() != ""
    final_mask = main_df["Final_Considered_Flag"] == "Yes"

    # Client KPIs should stay within the Nov 1 to Dec 15 operating window.
    client_valid_mask = final_mask & gate_timeline_mask.values
    client_matched_mask = matched_mask & gate_timeline_mask.values

    summary_rows = [
        ("Total records processed", len(main_df)),
        ("Gate Register records in timeline", int(gate_timeline_mask.sum())),
        ("System Extract records in timeline", int(sys_timeline_mask.sum())),
        ("Delta between Gate Register and System Extract", int(gate_timeline_mask.sum()) - int(sys_timeline_mask.sum())),
        ("Unique registrations in Gate Register after correction", int(main_df["Gate_Reg_No_Final"].fillna("").replace("", np.nan).dropna().nunique())),
        ("Unique registrations in System Extract", int(sys_df["Matched_System_Reg_No"].fillna("").replace("", np.nan).dropna().nunique())),
        ("System date/time parse failures", int(sys_df["__system_parse_fail__"].sum())),
        ("Gate Register date parse failures", int(gate_df["__gate_parse_fail__"].sum())),
        ("Records with Original TAT calculated", int(pd.to_numeric(main_df["Original_TAT_Hours"], errors="coerce").notna().sum())),
        ("Records with Corrected TAT calculated", int(pd.to_numeric(main_df["Corrected_TAT_Hours"], errors="coerce").notna().sum())),
        ("Records with system validation available", int((main_df["System_Validation_Available_Flag"] == "Yes").sum())),
        ("Final count", int(final_mask.sum())),
        ("Average calculated TAT from Final count", _hours_value(pd.to_numeric(main_df.loc[final_mask, "Corrected_TAT_Hours"], errors="coerce").mean()) if final_mask.any() else None),
        ("Registration numbers corrected", int((main_df["Reg_Corrected_via_System_Flag"] == "Yes").sum())),
        ("Formatting-only registration cleanup", int((main_df["Reg_Format_Cleaned_Flag"] == "Yes").sum())),
        ("Entries with gate in or gate out details corrected", int((main_df["TAT_Corrected_Flag"] == "Yes").sum())),
        ("Correction impact", int((main_df["Any_Correction_Flag"] == "Yes").sum())),
        ("No confident job card match", int((main_df["Match_Type"] == "No Match").sum())),
        ("Outliers flagged", int((main_df["Outlier_Flag"] == "Yes").sum())),
        ("ROT hour work mismatch", int((main_df["ROT_Work_Mismatch_Flag"] == "Yes").sum())),
        ("Negative Original TAT detected", int((pd.to_numeric(main_df["Original_TAT_Hours"], errors="coerce") < 0).sum())),
        ("Gate out AM/PM toggles applied", int((main_df["TAT_Correction_Source"] == "Gate_Out_AMPM_Toggled").sum())),
        ("System bill date/time used as corrected gate out", int((main_df["TAT_Correction_Source"] == "System_BillDate_As_GateOut").sum())),
        ("Match distribution - Exact", int(match_dist.get("Exact", 0))),
        ("Match distribution - Fuzzy_High", int(match_dist.get("Fuzzy_High", 0))),
        ("Match distribution - Fuzzy_Medium", int(match_dist.get("Fuzzy_Medium", 0))),
        ("Match distribution - No Match", int(match_dist.get("No Match", 0))),
        ("Share of matched records with score 100 percent", score_100_share),
        ("Client KPI - Total gate register vehicles Nov 1 to Dec 15", int(gate_timeline_mask.sum())),
        ("Client KPI - Valid entries from gate register data", int(client_valid_mask.sum())),
        ("Client KPI - Number of JCs matched from system data", int(client_matched_mask.sum())),
        ("Client KPI - Average GIGO TAT for all valid registration numbers", _hours_value(pd.to_numeric(main_df.loc[client_valid_mask, "Corrected_TAT_Hours"], errors="coerce").mean()) if client_valid_mask.any() else None),
        ("Client KPI - Average GIGO TAT delay for all matched job cards", _hours_value(pd.to_numeric(main_df.loc[client_matched_mask, "GIGO_TAT_Delay_Hours"], errors="coerce").mean()) if client_matched_mask.any() else None),
    ]
    return pd.DataFrame(summary_rows, columns=["Metric", "Value"])

def build_suggestions_export(main_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "S_No", "Gate_Reg_No_Raw", "Gate_Reg_No_Final",
        "Top_Suggested_Job_Card_No", "Top_Suggested_System_Reg_No",
        "Top_Suggestion_Score", "Top_Suggestion_Time_Diff_Hours",
        "Matched_Job_Card_No", "Match_Type", "Final_Remarks",
    ]
    return main_df[cols].copy()


def build_manual_review_export(main_df: pd.DataFrame) -> pd.DataFrame:
    manual_mask = (
        (main_df["Match_Type"] == "No Match")
        | (main_df["TAT_Correction_Source"] == "Unresolved_Negative_TAT")
        | (main_df["Business_Validation_Status"] != "OK")
    )
    cols = [
        "S_No", "Gate_Reg_No_Raw", "Gate_Reg_No_Final",
        "Original_TAT_Hours", "Corrected_TAT_Hours",
        "Matched_Job_Card_No", "Match_Type", "Match_Score",
        "Business_Validation_Status", "Final_Considered_Flag",
        "Final_Remarks",
    ]
    return main_df.loc[manual_mask, cols].copy()


def _validate_output(main_df: pd.DataFrame):
    if list(main_df.columns) != MAIN_OUTPUT_COLUMNS:
        raise ValueError("Main output columns do not match the rulebook.")
    if list(main_df["S_No"]) != list(range(1, len(main_df) + 1)):
        raise ValueError("S_No is not sequential.")
    if len(main_df) == 0:
        return
    if "Original_TAT_Hours" not in main_df.columns or "Corrected_TAT_Hours" not in main_df.columns:
        raise ValueError("Required TAT columns are missing.")


def run_tat_pipeline(
    gate_path: str,
    sys_path: str,
    out_dir: str,
    gate_sheet: Optional[str] = None,
    sys_sheet: Optional[str] = None,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)

    gate_raw = read_table(gate_path, sheet_name=gate_sheet, kind="gate")
    sys_raw = read_table(sys_path, sheet_name=sys_sheet, kind="system")

    gate_df, gate_mapping = standardize_gate(gate_raw)
    sys_df, sys_mapping = standardize_system(sys_raw)

    suggestion_candidate_df = build_candidate_table(gate_df, sys_df, purpose="suggest")
    candidate_df = build_candidate_table(gate_df, sys_df, purpose="match")
    suggestions = top_suggestions(suggestion_candidate_df)
    assignments = global_one_to_one_assignment(gate_df, sys_df, candidate_df)
    assignments, second_pass_reasons = second_pass_promote(assignments, suggestions, gate_df)

    main_df = build_main_output(gate_df, sys_df, assignments, suggestions, second_pass_reasons)
    summary_df = build_summary(gate_df, sys_df, main_df)
    suggestions_df = build_suggestions_export(main_df)
    manual_df = build_manual_review_export(main_df)

    main_df = _sanitize_dataframe_missing_values(main_df)
    summary_df = _sanitize_dataframe_missing_values(summary_df)
    suggestions_df = _sanitize_dataframe_missing_values(suggestions_df)
    manual_df = _sanitize_dataframe_missing_values(manual_df)

    _validate_output(main_df)

    workbook_path = os.path.join(out_dir, "Workshop_TAT_Output.xlsx")
    main_csv_path = os.path.join(out_dir, "Main_Output.csv")
    summary_csv_path = os.path.join(out_dir, "Summary.csv")
    suggestions_csv_path = os.path.join(out_dir, "Suggestions.csv")
    manual_csv_path = os.path.join(out_dir, "Manual_Review.csv")
    stats_csv_path = os.path.join(out_dir, "Statistics.csv")

    with pd.ExcelWriter(workbook_path, engine="xlsxwriter") as writer:
        main_df.to_excel(writer, sheet_name="Main_Output", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    main_df.to_csv(main_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    suggestions_df.to_csv(suggestions_csv_path, index=False)
    manual_df.to_csv(manual_csv_path, index=False)
    summary_df.to_csv(stats_csv_path, index=False)

    # charts
    chart_match_distribution = os.path.join(out_dir, "chart_match_distribution.png")
    chart_tat_buckets = os.path.join(out_dir, "chart_tat_buckets.png")
    chart_waterfall = os.path.join(out_dir, "chart_waterfall.png")

    _save_bar_chart(main_df["Match_Type"].fillna("No Match"), "Match Distribution", chart_match_distribution)
    tat_buckets = _bucketize_tat(main_df["Corrected_TAT_Hours"])
    _save_bar_chart(tat_buckets, "Corrected TAT Buckets", chart_tat_buckets)
    _save_waterfall(
        [
            ("Total Gate Rows", float(len(main_df))),
            ("Timeline Rows", float(summary_df.loc[summary_df["Metric"] == "Gate Register records in timeline", "Value"].iloc[0])),
            ("Matched JCs", float(summary_df.loc[summary_df["Metric"] == "Client KPI - Number of JCs matched from system data", "Value"].iloc[0])),
            ("Corrected TAT", float(summary_df.loc[summary_df["Metric"] == "Records with Corrected TAT calculated", "Value"].iloc[0])),
            ("Final Count", float(summary_df.loc[summary_df["Metric"] == "Final count", "Value"].iloc[0])),
        ],
        "Pipeline Overview",
        chart_waterfall,
    )

    zip_path = os.path.join(out_dir, "Workshop_TAT_Outputs.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in [
            workbook_path, main_csv_path, summary_csv_path, suggestions_csv_path, manual_csv_path, stats_csv_path,
            chart_match_distribution, chart_tat_buckets, chart_waterfall,
        ]:
            zf.write(p, arcname=Path(p).name)

    # client-facing KPIs for the app
    summary_map = dict(summary_df.values.tolist())
    summary_metrics = {
        "total_gate_vehicles_nov1_dec15": summary_map.get("Client KPI - Total gate register vehicles Nov 1 to Dec 15"),
        "valid_gate_entries": summary_map.get("Client KPI - Valid entries from gate register data"),
        "matched_jcs": summary_map.get("Client KPI - Number of JCs matched from system data"),
        "avg_gigo_tat_valid": summary_map.get("Client KPI - Average GIGO TAT for all valid registration numbers"),
        "avg_gigo_delay_matched": summary_map.get("Client KPI - Average GIGO TAT delay for all matched job cards"),
    }

    return {
        "main_csv": main_csv_path,
        "summary_csv": summary_csv_path,
        "suggestions_csv": suggestions_csv_path,
        "manual_csv": manual_csv_path,
        "stats_csv": stats_csv_path,
        "workbook": workbook_path,
        "zip": zip_path,
        "chart_match_distribution": chart_match_distribution,
        "chart_tat_buckets": chart_tat_buckets,
        "chart_waterfall": chart_waterfall,
        "summary_metrics": summary_metrics,
        "detected_gate_columns": gate_mapping,
        "detected_system_columns": sys_mapping,
    }


# ===================== PARITY3 OVERRIDES =====================
# Final knowledge-file alignment:
# - exact registration matching is uncapped by 48h
# - fuzzy matching is considered only when a gate row has no exact candidate
# - split date/time columns are preferred over combined datetime columns
# - system-based registration correction is applied only on non-exact fuzzy matches
# - summary KPI definitions follow the final Workshop TAT knowledge file
# - AM/PM toggle correction moves the full datetime by +/- 12 hours

ENGINE_VERSION = "2026-03-23-parity3"

FUZZY_MATCH_THRESHOLD = 85.0


def _prefer_split_date_time_mapping(mapping: Dict[str, Optional[str]], bases: Iterable[str]) -> Dict[str, Optional[str]]:
    mapping = dict(mapping)
    for base in bases:
        dt_key = f"{base}_datetime"
        d_key = f"{base}_date"
        t_key = f"{base}_time"
        if mapping.get(d_key) is not None and mapping.get(t_key) is not None:
            mapping[dt_key] = None
    return mapping


def build_candidate_table(gate_df: pd.DataFrame, sys_df: pd.DataFrame, purpose: str = "suggest") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, g in gate_df.iterrows():
        g_reg = g["Gate_Reg_No_Final"] or g["Gate_Reg_No_Cleaned"]
        if not g_reg:
            continue
        g_anchor = _candidate_time_anchor(g)

        for _, s in sys_df.iterrows():
            if s.get("__system_matching_blocked__", False):
                continue
            s_reg = s["__system_reg_final__"] or s["__system_reg_cleaned__"]
            if not s_reg:
                continue

            sim = _registration_similarity(g_reg, s_reg)
            exact = (g_reg == s_reg and g_reg != "")
            time_diff = _time_diff_hours(g_anchor, _system_time_anchor(s))
            bill_diff = _bill_diff_hours(g, s)

            if purpose == "suggest" and sim <= 0 and not exact:
                continue

            rows.append(
                {
                    "gate_idx": int(g["__gate_idx__"]),
                    "sys_idx": int(s["__sys_idx__"]),
                    "gate_reg": g_reg,
                    "sys_reg": s_reg,
                    "sim_score": float(sim),
                    "exact": bool(exact),
                    "time_diff_hours": time_diff,
                    "bill_diff_hours": bill_diff,
                    "sys_order": int(s["__system_order__"]),
                }
            )
    return pd.DataFrame(rows)


def _sort_key_exact_row(row: pd.Series) -> Tuple[float, float, int]:
    t = float(row["time_diff_hours"]) if row["time_diff_hours"] is not None and not pd.isna(row["time_diff_hours"]) else 1e12
    b = float(row["bill_diff_hours"]) if row["bill_diff_hours"] is not None and not pd.isna(row["bill_diff_hours"]) else 1e12
    so = int(row["sys_order"]) if row["sys_order"] is not None and not pd.isna(row["sys_order"]) else 10**9
    return (t, b, so)


def _sort_key_suggestion_row(row: pd.Series) -> Tuple[float, float, float, int]:
    score = float(row["sim_score"]) if row["sim_score"] is not None and not pd.isna(row["sim_score"]) else -1.0
    t = float(row["time_diff_hours"]) if row["time_diff_hours"] is not None and not pd.isna(row["time_diff_hours"]) else 1e12
    b = float(row["bill_diff_hours"]) if row["bill_diff_hours"] is not None and not pd.isna(row["bill_diff_hours"]) else 1e12
    so = int(row["sys_order"]) if row["sys_order"] is not None and not pd.isna(row["sys_order"]) else 10**9
    return (-score, t, b, so)


def top_suggestions(candidate_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    suggestions: Dict[int, Dict[str, Any]] = {}
    if candidate_df.empty:
        return suggestions

    for gate_idx, grp in candidate_df.groupby("gate_idx", sort=False):
        exact_grp = grp[grp["exact"] == True]
        pool = exact_grp if not exact_grp.empty else grp
        if not exact_grp.empty:
            ranked = pool.sort_values(
                ["time_diff_hours", "bill_diff_hours", "sys_order"],
                ascending=[True, True, True],
                na_position="last",
            )
        else:
            ranked = pool.sort_values(
                ["sim_score", "time_diff_hours", "bill_diff_hours", "sys_order"],
                ascending=[False, True, True, True],
                na_position="last",
            )
        suggestions[int(gate_idx)] = ranked.iloc[0].to_dict()
    return suggestions


def _build_exact_reason(
    gate_idx: int,
    top_exact: Dict[str, Any],
    sys_lookup: Dict[int, Dict[str, Any]],
    owner_gate: Optional[int],
) -> str:
    sys_row = sys_lookup.get(int(top_exact["sys_idx"]))
    reg = sys_row["Matched_System_Reg_No"] if sys_row else top_exact.get("sys_reg")
    jc = sys_row["Matched_Job_Card_No"] if sys_row else None
    score = float(top_exact.get("sim_score") or 100.0)
    tdiff = top_exact.get("time_diff_hours")
    ttxt = f"{float(tdiff):.2f}h" if tdiff is not None and not pd.isna(tdiff) else "time diff unavailable"
    if owner_gate is not None:
        return (
            f"Exact registration candidate {reg} / JC {jc} (score {score:.2f}, {ttxt}) "
            f"was not used because it was assigned to closer same-registration visit S_No {owner_gate + 1} "
            f"under repeated-vehicle one-to-one rules."
        )
    return (
        f"Exact registration candidate {reg} / JC {jc} (score {score:.2f}, {ttxt}) "
        f"could not be retained after one-to-one repeated-vehicle conflict resolution."
    )


def _exact_assignment_cost(row: pd.Series) -> float:
    t = float(row["time_diff_hours"]) if row["time_diff_hours"] is not None and not pd.isna(row["time_diff_hours"]) else 1e7
    b = float(row["bill_diff_hours"]) if row["bill_diff_hours"] is not None and not pd.isna(row["bill_diff_hours"]) else 1e6
    so = float(row["sys_order"]) if row["sys_order"] is not None and not pd.isna(row["sys_order"]) else 1e5
    return t * 1000.0 + b + so * 1e-3


def _assign_exact_matches(
    gate_df: pd.DataFrame,
    sys_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, str], set]:
    assignments: Dict[int, Dict[str, Any]] = {}
    reasons: Dict[int, str] = {}
    if candidate_df.empty:
        return assignments, reasons, set()

    exact_df = candidate_df[candidate_df["exact"] == True].copy()
    if exact_df.empty:
        return assignments, reasons, set()

    exact_gate_set = set(int(v) for v in exact_df["gate_idx"].unique().tolist())
    sys_lookup = sys_df.set_index("__sys_idx__", drop=False).to_dict(orient="index")

    for reg, grp in exact_df.groupby("gate_reg", sort=False):
        gate_ids = sorted(int(v) for v in grp["gate_idx"].unique().tolist())
        sys_ids = sorted(int(v) for v in grp["sys_idx"].unique().tolist())
        if not gate_ids or not sys_ids:
            continue

        gpos = {g: i for i, g in enumerate(gate_ids)}
        spos = {s: i for i, s in enumerate(sys_ids)}
        dummy_count = len(gate_ids)
        big = 1e9
        dummy_cost = 1e8
        cost = np.full((len(gate_ids), len(sys_ids) + dummy_count), dummy_cost, dtype=float)
        cost[:, :len(sys_ids)] = big

        for _, row in grp.iterrows():
            gi = int(row["gate_idx"])
            si = int(row["sys_idx"])
            cost[gpos[gi], spos[si]] = min(cost[gpos[gi], spos[si]], _exact_assignment_cost(row))

        row_ind, col_ind = linear_sum_assignment(cost)
        pair_lookup = grp.set_index(["gate_idx", "sys_idx"]).sort_index()

        for r, c in zip(row_ind, col_ind):
            gi = gate_ids[int(r)]
            if c >= len(sys_ids):
                continue
            si = sys_ids[int(c)]
            if (gi, si) not in pair_lookup.index:
                continue
            rec = pair_lookup.loc[(gi, si)]
            if isinstance(rec, pd.DataFrame):
                rec = rec.sort_values(["time_diff_hours", "bill_diff_hours", "sys_order"], na_position="last").iloc[0]
            rec_dict = rec.to_dict()
            rec_dict["gate_idx"] = gi
            rec_dict["sys_idx"] = si
            rec_dict["exact"] = True
            assignments[gi] = rec_dict

        owner_by_sys = {int(v["sys_idx"]): gi for gi, v in assignments.items()}
        for gi in gate_ids:
            if gi in assignments:
                continue
            gi_grp = grp[grp["gate_idx"] == gi].sort_values(["time_diff_hours", "bill_diff_hours", "sys_order"], na_position="last")
            if gi_grp.empty:
                continue
            top_exact = gi_grp.iloc[0].to_dict()
            owner_gate = owner_by_sys.get(int(top_exact["sys_idx"]))
            reasons[gi] = _build_exact_reason(gi, top_exact, sys_lookup, owner_gate)

    return assignments, reasons, exact_gate_set


def _fuzzy_reward(row: pd.Series) -> float:
    score = float(row["sim_score"]) if row["sim_score"] is not None and not pd.isna(row["sim_score"]) else 0.0
    t = float(row["time_diff_hours"]) if row["time_diff_hours"] is not None and not pd.isna(row["time_diff_hours"]) else 1e6
    b = float(row["bill_diff_hours"]) if row["bill_diff_hours"] is not None and not pd.isna(row["bill_diff_hours"]) else 1e6
    so = float(row["sys_order"]) if row["sys_order"] is not None and not pd.isna(row["sys_order"]) else 1e6
    return score * 10000.0 - t * 100.0 - b - so * 1e-3


def _assign_fuzzy_matches(
    gate_df: pd.DataFrame,
    sys_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    exact_gate_set: set,
    preassignments: Dict[int, Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    assignments = dict(preassignments)
    if candidate_df.empty:
        return assignments

    used_sys = {int(v["sys_idx"]) for v in assignments.values()}
    fuzzy_df = candidate_df[
        (candidate_df["exact"] == False)
        & (~candidate_df["gate_idx"].isin(list(exact_gate_set)))
        & (candidate_df["sim_score"] > FUZZY_MATCH_THRESHOLD)
        & (candidate_df["time_diff_hours"].notna())
        & (candidate_df["time_diff_hours"] <= 48)
        & (~candidate_df["sys_idx"].isin(list(used_sys)))
    ].copy()

    if fuzzy_df.empty:
        return assignments

    gate_ids = sorted(int(v) for v in fuzzy_df["gate_idx"].unique().tolist())
    sys_ids = sorted(int(v) for v in fuzzy_df["sys_idx"].unique().tolist())
    if not gate_ids or not sys_ids:
        return assignments

    gpos = {g: i for i, g in enumerate(gate_ids)}
    spos = {s: i for i, s in enumerate(sys_ids)}
    dummy_count = len(gate_ids)
    cost = np.zeros((len(gate_ids), len(sys_ids) + dummy_count), dtype=float)

    lookup = fuzzy_df.set_index(["gate_idx", "sys_idx"]).sort_index()
    for _, row in fuzzy_df.iterrows():
        gi = int(row["gate_idx"])
        si = int(row["sys_idx"])
        cost[gpos[gi], spos[si]] = -_fuzzy_reward(row)

    row_ind, col_ind = linear_sum_assignment(cost)
    for r, c in zip(row_ind, col_ind):
        gi = gate_ids[int(r)]
        if c >= len(sys_ids):
            continue
        si = sys_ids[int(c)]
        if (gi, si) not in lookup.index:
            continue
        rec = lookup.loc[(gi, si)]
        if isinstance(rec, pd.DataFrame):
            rec = rec.sort_values(["sim_score", "time_diff_hours", "bill_diff_hours", "sys_order"], ascending=[False, True, True, True], na_position="last").iloc[0]
        rec_dict = rec.to_dict()
        rec_dict["gate_idx"] = gi
        rec_dict["sys_idx"] = si
        rec_dict["exact"] = False
        assignments[gi] = rec_dict

    return assignments


def _match_priority_tuple(cand: Dict[str, Any], gate_idx: int) -> Tuple[float, float, float, float, float]:
    score = float(cand.get("sim_score") or 0.0)
    t = float(cand.get("time_diff_hours")) if cand.get("time_diff_hours") is not None and not pd.isna(cand.get("time_diff_hours")) else 1e12
    b = float(cand.get("bill_diff_hours")) if cand.get("bill_diff_hours") is not None and not pd.isna(cand.get("bill_diff_hours")) else 1e12
    so = float(cand.get("sys_order")) if cand.get("sys_order") is not None and not pd.isna(cand.get("sys_order")) else 1e12
    return (score, -t, -b, -so, -float(gate_idx))


def second_pass_promote(
    assignments: Dict[int, Dict[str, Any]],
    suggestions: Dict[int, Dict[str, Any]],
    gate_df: pd.DataFrame,
    exact_gate_set: Optional[set] = None,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, str]]:
    if exact_gate_set is None:
        exact_gate_set = set()

    reasons: Dict[int, str] = {}
    used_sys = {int(v["sys_idx"]) for v in assignments.values()}
    sys_to_gate = {int(v["sys_idx"]): g for g, v in assignments.items()}

    for gi in range(len(gate_df)):
        if gi in assignments:
            continue

        sugg = suggestions.get(gi)
        if not sugg:
            continue

        score = float(sugg.get("sim_score") or 0.0)
        tdiff = sugg.get("time_diff_hours")
        reg = sugg.get("sys_reg")
        jc = None
        if "sys_idx" in sugg:
            # resolve JC from current suggestion
            pass

        if gi in exact_gate_set:
            # reason for exact conflicts is populated elsewhere and should win over fuzzy logic
            continue

        if sugg.get("exact"):
            continue

        sys_idx = int(sugg["sys_idx"])
        score_txt = f"{score:.2f}"
        time_txt = f"{float(tdiff):.2f}h" if tdiff is not None and not pd.isna(tdiff) else "time diff unavailable"

        if score <= FUZZY_MATCH_THRESHOLD or tdiff is None or pd.isna(tdiff) or float(tdiff) > 48:
            continue

        if sys_idx not in used_sys:
            promoted = dict(sugg)
            promoted["promoted"] = True
            assignments[gi] = promoted
            used_sys.add(sys_idx)
            sys_to_gate[sys_idx] = gi
            reasons[gi] = "High-confidence suggestion reviewed and promoted."
            continue

        owner_gate = sys_to_gate[sys_idx]
        current = assignments[owner_gate]
        if current.get("exact"):
            reasons[gi] = f"High-confidence suggestion reviewed but not promoted due to stronger competing row S_No {owner_gate + 1}."
            continue

        new_pri = _match_priority_tuple(sugg, gi)
        cur_pri = _match_priority_tuple(current, owner_gate)

        if new_pri > cur_pri:
            assignments.pop(owner_gate, None)
            promoted = dict(sugg)
            promoted["promoted"] = True
            assignments[gi] = promoted
            sys_to_gate[sys_idx] = gi
            reasons[gi] = "High-confidence suggestion reviewed and promoted."
            reasons[owner_gate] = f"High-confidence suggestion reviewed but not promoted due to stronger competing row S_No {gi + 1}."
        else:
            reasons[gi] = f"High-confidence suggestion reviewed but not promoted due to stronger competing row S_No {owner_gate + 1}."

    return assignments, reasons


def classify_match_type(score: Optional[float], exact: bool, promoted: bool = False) -> str:
    if score is None:
        return "No Match"
    if exact:
        return "Exact"
    if float(score) > FUZZY_MATCH_THRESHOLD:
        return "Fuzzy_High"
    if float(score) >= 80:
        return "Fuzzy_Medium"
    return "No Match"


def _build_row_remark(
    row: Dict[str, Any],
    second_pass_reason: Optional[str],
    high_conf_rejected_reason: Optional[str],
) -> str:
    remarks: List[str] = []

    cleaned_reg = row.get("Gate_Reg_No_Cleaned")
    final_reg = row.get("Gate_Reg_No_Final")
    match_type = row.get("Match_Type")

    if row.get("Reg_Corrected_via_System_Flag") == "Yes" and cleaned_reg and final_reg and cleaned_reg != final_reg:
        remarks.append(f"Registration corrected via system validation from {cleaned_reg} to {final_reg}.")
    else:
        remarks.append("Registration kept as recorded after standard cleaning.")

    if match_type == "Exact":
        jc = row.get("Matched_Job_Card_No")
        reg = row.get("Matched_System_Reg_No")
        tdiff = row.get("Match_Time_Diff_Hours")
        if tdiff not in (None, "") and not pd.isna(tdiff):
            remarks.append(f"Matched Exact to JC {jc} ({reg}); gate-in/reporting time diff {float(tdiff):.2f}h.")
        else:
            remarks.append(f"Matched Exact to JC {jc} ({reg}).")
    elif match_type in ("Fuzzy_High", "Fuzzy_Medium"):
        jc = row.get("Matched_Job_Card_No")
        reg = row.get("Matched_System_Reg_No")
        score = row.get("Match_Score")
        tdiff = row.get("Match_Time_Diff_Hours")
        promoted_txt = " second-pass promoted." if second_pass_reason and "promoted" in second_pass_reason.lower() else ""
        remarks.append(
            f"{match_type} match to JC {jc} / reg {reg} (score {float(score):.2f}, time diff {float(tdiff):.2f}h).{promoted_txt}".strip()
        )
    else:
        reason = high_conf_rejected_reason or second_pass_reason
        if reason:
            remarks.append(reason)
        else:
            remarks.append("No confident one-to-one system match after exact/fuzzy review.")

    corr_source = row.get("TAT_Correction_Source")
    if corr_source == "Gate Out AM/PM Toggle":
        remarks.append("Negative Original TAT was corrected by gate-out AM/PM toggle.")
    elif corr_source == "System Bill Date/Time":
        remarks.append("Negative Original TAT was corrected using matched system bill date/time.")
    elif corr_source == "Unresolved_Negative_TAT":
        remarks.append("Negative TAT could not be resolved with allowed corrections.")
    else:
        remarks.append("No corrections were required.")

    status = row.get("Business_Validation_Status")
    if status in ("OK", "", None):
        if row.get("Final_Considered_Flag") == "Yes":
            remarks.append("Included in final count.")
        else:
            remarks.append("Excluded from final count because Corrected TAT could not be computed.")
    else:
        if row.get("Final_Considered_Flag") == "Yes":
            remarks.append(f"Business flags: {status}. Row remains included in final count.")
        else:
            remarks.append(f"Business flags: {status}. Row is excluded from final count.")

    return " ".join(remarks)


def build_main_output(
    gate_df: pd.DataFrame,
    sys_df: pd.DataFrame,
    assignments: Dict[int, Dict[str, Any]],
    suggestions: Dict[int, Dict[str, Any]],
    second_pass_reasons: Dict[int, str],
) -> pd.DataFrame:
    sys_lookup = sys_df.set_index("__sys_idx__", drop=False).to_dict(orient="index")
    rows: List[Dict[str, Any]] = []

    for i in range(len(gate_df)):
        g = gate_df.iloc[i].to_dict()
        out = {col: None for col in MAIN_OUTPUT_COLUMNS}
        out["S_No"] = i + 1
        for key in [
            "Gate_Reg_No_Raw", "Gate_Reg_No_Cleaned", "Gate_Reg_No_Final",
            "Gate_Reg_Strict_Format_Flag", "Final_Reg_Strict_Format_Flag",
            "Reg_Format_Cleaned_Flag", "Reg_Corrected_via_System_Flag",
            "Reporting_Date_Clean", "Reporting_Time_Clean",
            "Workshop_In_Date_Clean", "Workshop_In_Time_Clean",
            "Workshop_Out_Date_Clean", "Workshop_Out_Time_Clean",
        ]:
            out[key] = g.get(key)

        out["Gate_In_Source"] = g.get("__gate_in_source__")
        out["Base_Gate_In_Date"] = _format_date(_safe_dt_date(g.get("__base_gate_in_dt__")))
        out["Base_Gate_In_Time"] = _format_time(_safe_dt_time(g.get("__base_gate_in_dt__")))

        sugg = suggestions.get(i)
        if sugg:
            srow = sys_lookup.get(int(sugg["sys_idx"]))
            out["Top_Suggested_Job_Card_No"] = srow["Matched_Job_Card_No"] if srow else None
            out["Top_Suggested_System_Reg_No"] = srow["Matched_System_Reg_No"] if srow else None
            out["Top_Suggestion_Score"] = _raw_num(float(sugg["sim_score"])) if sugg.get("sim_score") is not None else None
            out["Top_Suggestion_Time_Diff_Hours"] = _raw_num(float(sugg["time_diff_hours"])) if sugg.get("time_diff_hours") is not None and not pd.isna(sugg.get("time_diff_hours")) else None

        assn = assignments.get(i)
        matched_sys = None
        match_score = None
        match_time = None
        match_bill = None
        exact = False
        promoted = False

        if assn:
            matched_sys = sys_lookup.get(int(assn["sys_idx"]))
            match_score = float(assn["sim_score"]) if assn.get("sim_score") is not None else None
            match_time = assn.get("time_diff_hours")
            match_bill = assn.get("bill_diff_hours")
            exact = bool(assn.get("exact", False))
            promoted = bool(assn.get("promoted", False))

        if matched_sys and not exact and matched_sys["Matched_System_Reg_No"]:
            current_final = out["Gate_Reg_No_Final"] or out["Gate_Reg_No_Cleaned"] or ""
            system_final = matched_sys["Matched_System_Reg_No"] or ""
            if system_final and current_final and system_final != current_final:
                out["Gate_Reg_No_Final"] = system_final
                out["Final_Reg_Strict_Format_Flag"] = "Yes" if bool(matched_sys.get("__system_reg_strict__", False)) or is_strict_reg(system_final) else out["Final_Reg_Strict_Format_Flag"]
                out["Reg_Corrected_via_System_Flag"] = "Yes"

        if matched_sys:
            out["Matched_Job_Card_No"] = matched_sys["Matched_Job_Card_No"]
            out["Matched_System_Reg_No"] = matched_sys["Matched_System_Reg_No"]
            out["Match_Type"] = classify_match_type(match_score, exact=exact, promoted=promoted)
            out["Match_Score"] = _raw_num(match_score)
            out["Match_Time_Diff_Hours"] = _raw_num(match_time)
            out["Match_Bill_Diff_Hours"] = _raw_num(match_bill) if match_bill is not None and not pd.isna(match_bill) else None
            out["Matched_System_Gate_In_Date"] = matched_sys["Matched_System_Gate_In_Date"]
            out["Matched_System_Gate_In_Time"] = matched_sys["Matched_System_Gate_In_Time"]
            out["Matched_System_Bill_Date"] = matched_sys["Matched_System_Bill_Date"]
            out["Matched_System_Bill_Time"] = matched_sys["Matched_System_Bill_Time"]
            out["ROT_Hours"] = _raw_num(matched_sys["ROT_Hours"]) if matched_sys["ROT_Hours"] is not None and not pd.isna(matched_sys["ROT_Hours"]) else None
            out["System_Validation_Available_Flag"] = "Yes"
        else:
            out["Match_Type"] = "No Match"
            out["System_Validation_Available_Flag"] = "No"

        base_gate_in_dt = _normalize_datetime_like(g.get("__base_gate_in_dt__"))
        gate_out_dt = _normalize_datetime_like(g.get("__workshop_out_dt__"))

        original_tat = None
        if base_gate_in_dt is not None and gate_out_dt is not None:
            original_tat = _raw_num(_safe_hours(gate_out_dt - base_gate_in_dt))

        out["Gate_Register_GIGO_TAT_Hours"] = original_tat
        out["Original_TAT_Hours"] = original_tat

        corrected_gate_out_dt = gate_out_dt
        tat_corrected = False
        gate_time_corrected = False
        correction_source = None
        correction_flags: List[str] = []

        if out["Reg_Format_Cleaned_Flag"] == "Yes":
            correction_flags.append("Reg Format Cleaned")
        if out["Reg_Corrected_via_System_Flag"] == "Yes":
            correction_flags.append("Reg Corrected via System")

        if original_tat is not None and original_tat < 0:
            toggle_candidates: List[Tuple[float, datetime]] = []
            if gate_out_dt is not None and base_gate_in_dt is not None:
                for delta in (12, -12):
                    cand_dt = gate_out_dt + timedelta(hours=delta)
                    cand_tat = _safe_hours(cand_dt - base_gate_in_dt)
                    if cand_tat is not None and cand_tat >= 0:
                        toggle_candidates.append((float(cand_tat), cand_dt))
            if toggle_candidates:
                toggle_candidates.sort(key=lambda x: (x[0], x[1]))
                corrected_gate_out_dt = toggle_candidates[0][1]
                tat_corrected = True
                gate_time_corrected = True
                correction_source = "Gate Out AM/PM Toggle"
                correction_flags.append("Gate Out AM/PM Toggle")
            elif matched_sys and _normalize_datetime_like(matched_sys.get("__system_bill_dt__")) is not None:
                corrected_gate_out_dt = _normalize_datetime_like(matched_sys.get("__system_bill_dt__"))
                tat_corrected = True
                correction_source = "System Bill Date/Time"
                correction_flags.append("System Bill Date/Time")
            else:
                correction_source = "Unresolved_Negative_TAT"
                correction_flags.append("Negative TAT Unresolved")

        corrected_gate_out_dt = _normalize_datetime_like(corrected_gate_out_dt)
        corrected_tat = None
        if base_gate_in_dt is not None and corrected_gate_out_dt is not None:
            corrected_tat = _raw_num(_safe_hours(corrected_gate_out_dt - base_gate_in_dt))

        out["Corrected_Gate_Out_Date"] = _format_date(_safe_dt_date(corrected_gate_out_dt))
        out["Corrected_Gate_Out_Time"] = _format_time(_safe_dt_time(corrected_gate_out_dt))
        out["Corrected_TAT_Hours"] = corrected_tat
        out["TAT_Corrected_Flag"] = _bool_to_yes_no(tat_corrected)
        out["Gate_Time_Corrected_Flag"] = _bool_to_yes_no(gate_time_corrected)
        out["TAT_Correction_Source"] = correction_source

        rot_hours = _to_float(out["ROT_Hours"])
        if original_tat is not None and rot_hours is not None:
            out["GIGO_TAT_Delay_Hours"] = _raw_num(original_tat - rot_hours - 4)
        else:
            out["GIGO_TAT_Delay_Hours"] = None

        outlier = False
        rot_mismatch = False
        business_msgs: List[str] = []

        if corrected_tat is None:
            out["Gate_In_Before_Gate_Out_Flag"] = None
            out["Outlier_Flag"] = None
            out["ROT_Work_Mismatch_Flag"] = None
            business_msgs.append("Corrected TAT unavailable")
        else:
            gate_before_out = corrected_tat >= 0
            out["Gate_In_Before_Gate_Out_Flag"] = _bool_to_yes_no(gate_before_out)
            if corrected_tat < 0:
                business_msgs.append("Gate In after Gate Out")
            if corrected_tat < 0.25:
                outlier = True
                business_msgs.append("TAT below 0.25 hours")
                if rot_hours is not None and rot_hours > corrected_tat:
                    rot_mismatch = True
                    business_msgs.append("ROT exceeds TAT for low-duration job")
            if corrected_tat > 336:
                outlier = True
                business_msgs.append("TAT above 336 hours")
            out["Outlier_Flag"] = _bool_to_yes_no(outlier)
            out["ROT_Work_Mismatch_Flag"] = _bool_to_yes_no(rot_mismatch)

        out["Business_Validation_Status"] = "OK" if business_msgs == [] else "; ".join(business_msgs)
        out["Final_Considered_Flag"] = "Yes" if corrected_tat is not None else "No"

        if g.get("__gate_parse_fail__"):
            correction_flags.append("Gate Date/Time Parse Issue")
        if outlier:
            correction_flags.append("Outlier Flag")
        if rot_mismatch:
            correction_flags.append("ROT Work Mismatch")

        out["Any_Correction_Flag"] = _bool_to_yes_no(bool(correction_flags))
        out["Correction_Flags"] = "; ".join(correction_flags) if correction_flags else None

        reason_text = second_pass_reasons.get(i)
        out["Final_Remarks"] = _build_row_remark(
            out,
            second_pass_reason=reason_text if matched_sys else None,
            high_conf_rejected_reason=reason_text if not matched_sys else None,
        )

        rows.append(out)

    main_df = pd.DataFrame(rows, columns=MAIN_OUTPUT_COLUMNS)
    return main_df


def build_summary(gate_df: pd.DataFrame, sys_df: pd.DataFrame, main_df: pd.DataFrame) -> pd.DataFrame:
    gate_timeline_mask = gate_df["__timeline_anchor_date__"].apply(lambda d: _within_operating_window(d) if isinstance(d, date) else False)
    sys_timeline_mask = sys_df["__timeline_anchor_date__"].apply(lambda d: _within_operating_window(d) if isinstance(d, date) else False)

    match_dist = main_df["Match_Type"].fillna("No Match").value_counts()
    matched_mask = main_df["Matched_Job_Card_No"].fillna("").astype(str).str.strip().replace("nan", "") != ""
    final_mask = main_df["Final_Considered_Flag"] == "Yes"

    matched_scores = pd.to_numeric(main_df.loc[matched_mask, "Match_Score"], errors="coerce")
    score_100_share = None
    if matched_scores.notna().any():
        score_100_share = round(float((matched_scores == 100).mean()) * 100, 2)

    gigo_series = pd.to_numeric(main_df["Gate_Register_GIGO_TAT_Hours"], errors="coerce")
    delay_series = pd.to_numeric(main_df["GIGO_TAT_Delay_Hours"], errors="coerce")
    rot_series = pd.to_numeric(main_df["ROT_Hours"], errors="coerce")

    valid_delay_mask = matched_mask & rot_series.notna() & delay_series.notna()
    minor_mask = valid_delay_mask & (rot_series <= 2)

    client_valid_mask = final_mask & gate_timeline_mask.values
    client_delay_mask = valid_delay_mask & gate_timeline_mask.values
    client_matched_mask = matched_mask & gate_timeline_mask.values

    summary_rows = [
        ("Total records processed", len(main_df)),
        ("Gate Register records in timeline", int(gate_timeline_mask.sum())),
        ("System Extract records in timeline", int(sys_timeline_mask.sum())),
        ("Delta between Gate Register and System Extract", int(gate_timeline_mask.sum()) - int(sys_timeline_mask.sum())),
        ("Unique registrations in Gate Register after correction", int(main_df["Gate_Reg_No_Final"].fillna("").replace("", np.nan).dropna().nunique())),
        ("Unique registrations in System Extract", int(sys_df["Matched_System_Reg_No"].fillna("").replace("", np.nan).dropna().nunique())),
        ("System date/time parse failures", int(sys_df["__system_parse_fail__"].sum())),
        ("Gate Register date parse failures", int(gate_df["__gate_parse_fail__"].sum())),
        ("Records with Original TAT calculated", int(pd.to_numeric(main_df["Original_TAT_Hours"], errors="coerce").notna().sum())),
        ("Records with Corrected TAT calculated", int(pd.to_numeric(main_df["Corrected_TAT_Hours"], errors="coerce").notna().sum())),
        ("Records with system validation available", int((main_df["System_Validation_Available_Flag"] == "Yes").sum())),
        ("Final count", int(final_mask.sum())),
        ("Average calculated TAT from Final count", _hours_value(pd.to_numeric(main_df.loc[final_mask, "Corrected_TAT_Hours"], errors="coerce").mean()) if final_mask.any() else None),
        ("Average GIGO TAT", _hours_value(gigo_series.dropna().mean()) if gigo_series.notna().any() else None),
        ("Average GIGO TAT Delay", _hours_value(delay_series.loc[valid_delay_mask].mean()) if valid_delay_mask.any() else None),
        ("Number of minor jobs", int(minor_mask.sum())),
        ("Average GIGO TAT for minor jobs", _hours_value(gigo_series.loc[minor_mask].mean()) if minor_mask.any() else None),
        ("Average GIGO TAT Delay for minor jobs", _hours_value(delay_series.loc[minor_mask].mean()) if minor_mask.any() else None),
        ("Registration numbers corrected", int((main_df["Reg_Corrected_via_System_Flag"] == "Yes").sum())),
        ("Formatting-only registration cleanup", int((main_df["Reg_Format_Cleaned_Flag"] == "Yes").sum())),
        ("Entries with gate in or gate out details corrected", int((main_df["TAT_Corrected_Flag"] == "Yes").sum())),
        ("Correction impact", int((main_df["Any_Correction_Flag"] == "Yes").sum())),
        ("No confident job card match", int((main_df["Match_Type"] == "No Match").sum())),
        ("Outliers flagged", int((main_df["Outlier_Flag"] == "Yes").sum())),
        ("ROT hour work mismatch", int((main_df["ROT_Work_Mismatch_Flag"] == "Yes").sum())),
        ("Negative Original TAT detected", int((pd.to_numeric(main_df["Original_TAT_Hours"], errors="coerce") < 0).sum())),
        ("Gate out AM/PM toggles applied", int((main_df["TAT_Correction_Source"] == "Gate Out AM/PM Toggle").sum())),
        ("System bill date/time used as corrected gate out", int((main_df["TAT_Correction_Source"] == "System Bill Date/Time").sum())),
        ("Match distribution - Exact", int(match_dist.get("Exact", 0))),
        ("Match distribution - Fuzzy_High", int(match_dist.get("Fuzzy_High", 0))),
        ("Match distribution - Fuzzy_Medium", int(match_dist.get("Fuzzy_Medium", 0))),
        ("Match distribution - No Match", int(match_dist.get("No Match", 0))),
        ("Share of matched records with score 100 percent", score_100_share),
        ("Client KPI - Total gate register vehicles Nov 1 to Dec 15", int(gate_timeline_mask.sum())),
        ("Client KPI - Valid entries from gate register data", int(client_valid_mask.sum())),
        ("Client KPI - Number of JCs matched from system data", int(client_matched_mask.sum())),
        ("Client KPI - Average GIGO TAT for all valid registration numbers", _hours_value(gigo_series.loc[client_valid_mask].mean()) if client_valid_mask.any() else None),
        ("Client KPI - Average GIGO TAT delay for all matched job cards", _hours_value(delay_series.loc[client_delay_mask].mean()) if client_delay_mask.any() else None),
    ]
    return pd.DataFrame(summary_rows, columns=["Metric", "Value"])


def run_tat_pipeline(
    gate_path: str,
    sys_path: str,
    out_dir: str,
    gate_sheet: Optional[str] = None,
    sys_sheet: Optional[str] = None,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)

    gate_raw = read_table(gate_path, sheet_name=gate_sheet, kind="gate")
    sys_raw = read_table(sys_path, sheet_name=sys_sheet, kind="system")

    gate_df, gate_mapping = standardize_gate(gate_raw)
    sys_df, sys_mapping = standardize_system(sys_raw)

    all_candidates = build_candidate_table(gate_df, sys_df, purpose="suggest")
    suggestions = top_suggestions(all_candidates)

    exact_assignments, exact_reasons, exact_gate_set = _assign_exact_matches(gate_df, sys_df, all_candidates)
    assignments = _assign_fuzzy_matches(gate_df, sys_df, all_candidates, exact_gate_set, exact_assignments)

    assignments, second_pass_reasons = second_pass_promote(assignments, suggestions, gate_df, exact_gate_set=exact_gate_set)

    # Merge exact conflict reasons with fuzzy review reasons, preserving specific exact explanations.
    merged_reasons = dict(second_pass_reasons)
    for gi, txt in exact_reasons.items():
        merged_reasons.setdefault(gi, txt)

    # Fill weak fuzzy "no promotion" reasons for unmatched non-exact rows
    sys_lookup = sys_df.set_index("__sys_idx__", drop=False).to_dict(orient="index")
    for gi in range(len(gate_df)):
        if gi in assignments or gi in exact_gate_set:
            continue
        sugg = suggestions.get(gi)
        if not sugg:
            continue
        srow = sys_lookup.get(int(sugg["sys_idx"])) if "sys_idx" in sugg else None
        reg = srow["Matched_System_Reg_No"] if srow else sugg.get("sys_reg")
        jc = srow["Matched_Job_Card_No"] if srow else None
        score = sugg.get("sim_score")
        tdiff = sugg.get("time_diff_hours")
        if score is None:
            continue
        if tdiff is not None and not pd.isna(tdiff):
            merged_reasons.setdefault(
                gi,
                f"No exact candidate. Best fuzzy suggestion {reg} / JC {jc} scored {float(score):.2f} with {float(tdiff):.2f}h proximity, but it was not strong enough for promotion after one-to-one review.",
            )
        else:
            merged_reasons.setdefault(
                gi,
                f"No exact candidate. Best fuzzy suggestion {reg} / JC {jc} scored {float(score):.2f}, but it was not strong enough for promotion after one-to-one review.",
            )

    main_df = build_main_output(gate_df, sys_df, assignments, suggestions, merged_reasons)
    summary_df = build_summary(gate_df, sys_df, main_df)
    suggestions_df = build_suggestions_export(main_df)
    manual_df = build_manual_review_export(main_df)

    main_df = _sanitize_dataframe_missing_values(main_df)
    summary_df = _sanitize_dataframe_missing_values(summary_df)
    suggestions_df = _sanitize_dataframe_missing_values(suggestions_df)
    manual_df = _sanitize_dataframe_missing_values(manual_df)

    _validate_output(main_df)

    workbook_path = os.path.join(out_dir, "Workshop_TAT_Output.xlsx")
    main_csv_path = os.path.join(out_dir, "Main_Output.csv")
    summary_csv_path = os.path.join(out_dir, "Summary.csv")
    suggestions_csv_path = os.path.join(out_dir, "Suggestions.csv")
    manual_csv_path = os.path.join(out_dir, "Manual_Review.csv")
    stats_csv_path = os.path.join(out_dir, "Statistics.csv")
    correction_summary_csv_path = os.path.join(out_dir, "Correction_Summary.csv")

    with pd.ExcelWriter(workbook_path, engine="xlsxwriter") as writer:
        main_df.to_excel(writer, sheet_name="Main_Output", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    main_df.to_csv(main_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    suggestions_df.to_csv(suggestions_csv_path, index=False)
    manual_df.to_csv(manual_csv_path, index=False)
    summary_df.to_csv(stats_csv_path, index=False)
    main_df.loc[main_df["Any_Correction_Flag"] == "Yes"].to_csv(correction_summary_csv_path, index=False)

    chart_match_distribution = os.path.join(out_dir, "chart_match_distribution.png")
    chart_tat_buckets = os.path.join(out_dir, "chart_tat_buckets.png")
    chart_waterfall = os.path.join(out_dir, "chart_waterfall.png")

    _save_bar_chart(main_df["Match_Type"].fillna("No Match"), "Match Distribution", chart_match_distribution)
    tat_buckets = _bucketize_tat(main_df["Corrected_TAT_Hours"])
    _save_bar_chart(tat_buckets, "Corrected TAT Buckets", chart_tat_buckets)
    _save_waterfall(
        [
            ("Total Gate Rows", float(len(main_df))),
            ("Timeline Rows", float(summary_df.loc[summary_df["Metric"] == "Gate Register records in timeline", "Value"].iloc[0])),
            ("Matched JCs", float(summary_df.loc[summary_df["Metric"] == "Client KPI - Number of JCs matched from system data", "Value"].iloc[0])),
            ("Corrected TAT", float(summary_df.loc[summary_df["Metric"] == "Records with Corrected TAT calculated", "Value"].iloc[0])),
            ("Final Count", float(summary_df.loc[summary_df["Metric"] == "Final count", "Value"].iloc[0])),
        ],
        "Pipeline Overview",
        chart_waterfall,
    )

    zip_path = os.path.join(out_dir, "Workshop_TAT_Outputs.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in [
            workbook_path, main_csv_path, summary_csv_path, suggestions_csv_path, manual_csv_path,
            stats_csv_path, correction_summary_csv_path, chart_match_distribution, chart_tat_buckets, chart_waterfall,
        ]:
            zf.write(p, arcname=Path(p).name)

    summary_map = dict(summary_df.values.tolist())
    summary_metrics = {
        "total_gate_vehicles_nov1_dec15": summary_map.get("Client KPI - Total gate register vehicles Nov 1 to Dec 15"),
        "valid_gate_entries": summary_map.get("Client KPI - Valid entries from gate register data"),
        "matched_jcs": summary_map.get("Client KPI - Number of JCs matched from system data"),
        "avg_gigo_tat_valid": summary_map.get("Client KPI - Average GIGO TAT for all valid registration numbers"),
        "avg_gigo_delay_matched": summary_map.get("Client KPI - Average GIGO TAT delay for all matched job cards"),
        "minor_jobs": summary_map.get("Number of minor jobs"),
        "avg_minor_gigo_tat": summary_map.get("Average GIGO TAT for minor jobs"),
        "avg_minor_gigo_delay": summary_map.get("Average GIGO TAT Delay for minor jobs"),
    }

    return {
        "main_csv": main_csv_path,
        "summary_csv": summary_csv_path,
        "suggestions_csv": suggestions_csv_path,
        "manual_csv": manual_csv_path,
        "correction_summary_csv": correction_summary_csv_path,
        "stats_csv": stats_csv_path,
        "workbook": workbook_path,
        "zip": zip_path,
        "chart_match_distribution": chart_match_distribution,
        "chart_tat_buckets": chart_tat_buckets,
        "chart_waterfall": chart_waterfall,
        "summary_metrics": summary_metrics,
        "detected_gate_columns": gate_mapping,
        "detected_system_columns": sys_mapping,
    }

# =================== END PARITY3 OVERRIDES ===================


# ===================== RAW-CHECK VALIDATED PARITY4 OVERRIDES =====================
# Validated against the uploaded Nagaur raw Gate Register + Tableau Extract pair.
# Key goals:
# - preserve current matching parity (260 matched JCs / 155 minor jobs on that raw pair)
# - align client KPI cards to the agent-facing summary definitions
# - align exported Main_Output formatting/flags more closely to the agent style

ENGINE_VERSION = "2026-03-23-validated-parity4"


def _export_round_2(val: Any) -> Optional[float]:
    num = _to_float(val)
    if num is None:
        return None
    rounded = round(float(num), 2)
    if abs(rounded) < 1e-9:
        rounded = 0.0
    return rounded


def _export_rot_value(val: Any) -> Optional[float]:
    num = _to_float(val)
    if num is None:
        return None
    if abs(num - round(num)) < 1e-9:
        return float(int(round(num)))
    return float(num)


def _coerce_reason_to_agent_style(reason: Optional[str]) -> Optional[str]:
    if not reason:
        return reason
    txt = str(reason).strip()
    txt = txt.replace("Best fuzzy suggestion", "Best available suggestion")
    txt = txt.replace("scored ", "scored only ", 1) if "Best available suggestion" in txt and "scored only" not in txt else txt
    txt = txt.replace(", but it was not strong enough for promotion after one-to-one review.", "; row remains No Match.")
    txt = txt.replace("could not be retained after one-to-one repeated-vehicle conflict resolution.", "was not retained after one-to-one repeated-vehicle conflict resolution.")
    return txt


def _business_status_agent_style(corrected_tat: Optional[float], rot_hours: Optional[float]) -> str:
    if corrected_tat is None:
        return "Corrected TAT unavailable"
    msgs: List[str] = []
    if corrected_tat < 0:
        msgs.append("Gate In after Gate Out")
    if corrected_tat < 0.25:
        if rot_hours is not None and rot_hours > corrected_tat:
            msgs.append("Outlier: TAT < 0.25 hours; ROT > TAT where TAT < 0.25 hours")
        else:
            msgs.append("Outlier: TAT < 0.25 hours")
    if corrected_tat > 336:
        msgs.append("Outlier: TAT > 336 hours")
    return "OK" if not msgs else "; ".join(msgs)


def _build_row_remark(
    row: Dict[str, Any],
    second_pass_reason: Optional[str],
    high_conf_rejected_reason: Optional[str],
    repeated_exact: bool = False,
) -> str:
    remarks: List[str] = []

    raw_reg = row.get("Gate_Reg_No_Raw")
    cleaned_reg = row.get("Gate_Reg_No_Cleaned")
    final_reg = row.get("Gate_Reg_No_Final")
    match_type = row.get("Match_Type")
    business_status = row.get("Business_Validation_Status")
    any_correction = str(row.get("Any_Correction_Flag") or "").strip().upper() == "YES"

    if row.get("Reg_Corrected_via_System_Flag") == "Yes" and cleaned_reg and final_reg and cleaned_reg != final_reg:
        remarks.append(f"Registration corrected via system validation from {cleaned_reg} to {final_reg}.")
    elif row.get("Reg_Format_Cleaned_Flag") == "Yes" and raw_reg and cleaned_reg and str(raw_reg) != str(cleaned_reg):
        remarks.append(f"Registration cleaned from {raw_reg} to {cleaned_reg}.")
    else:
        remarks.append("Registration kept as recorded after standard cleaning.")

    if match_type == "Exact":
        jc = row.get("Matched_Job_Card_No")
        reg = row.get("Matched_System_Reg_No")
        tdiff = _to_float(row.get("Match_Time_Diff_Hours"))
        phrase = f"Matched Exact to JC {jc} ({reg})"
        if repeated_exact:
            phrase += " under repeated-vehicle one-to-one allocation"
        if tdiff is not None:
            phrase += f"; gate-in/reporting time diff {tdiff:.2f}h"
            if tdiff > 48:
                phrase += ", accepted despite >48h gap per exact-match rule"
        phrase += "."
        remarks.append(phrase)
    elif match_type in ("Fuzzy_High", "Fuzzy_Medium"):
        jc = row.get("Matched_Job_Card_No")
        reg = row.get("Matched_System_Reg_No")
        score = _to_float(row.get("Match_Score"))
        tdiff = _to_float(row.get("Match_Time_Diff_Hours"))
        if score is not None and tdiff is not None:
            remarks.append(
                f"No exact candidate. High-confidence suggestion {reg} / JC {jc} (score {score:.2f}, {tdiff:.2f}h) "
                f"was reviewed in mandatory second pass and promoted under one-to-one conflict resolution."
            )
        elif score is not None:
            remarks.append(
                f"No exact candidate. High-confidence suggestion {reg} / JC {jc} (score {score:.2f}) "
                f"was reviewed in mandatory second pass and promoted under one-to-one conflict resolution."
            )
        else:
            remarks.append(
                f"No exact candidate. High-confidence suggestion {reg} / JC {jc} was reviewed in mandatory second pass and "
                f"promoted under one-to-one conflict resolution."
            )
    else:
        reason = _coerce_reason_to_agent_style(high_conf_rejected_reason or second_pass_reason)
        if reason:
            remarks.append(reason)
        else:
            sugg_reg = row.get("Top_Suggested_System_Reg_No")
            sugg_jc = row.get("Top_Suggested_Job_Card_No")
            sugg_score = _to_float(row.get("Top_Suggestion_Score"))
            sugg_tdiff = _to_float(row.get("Top_Suggestion_Time_Diff_Hours"))
            if sugg_reg or sugg_jc or sugg_score is not None:
                if sugg_score is not None and sugg_tdiff is not None:
                    remarks.append(
                        f"No exact candidate. Best available suggestion {sugg_reg} / JC {sugg_jc} "
                        f"scored only {sugg_score:.2f} with {sugg_tdiff:.2f}h proximity; row remains No Match."
                    )
                elif sugg_score is not None:
                    remarks.append(
                        f"No exact candidate. Best available suggestion {sugg_reg} / JC {sugg_jc} "
                        f"scored only {sugg_score:.2f}; row remains No Match."
                    )
                else:
                    remarks.append("No confident one-to-one system match after exact/fuzzy review; row remains No Match.")
            else:
                remarks.append("No confident one-to-one system match after exact/fuzzy review; row remains No Match.")

    corr_source = row.get("TAT_Correction_Source")
    if corr_source == "Gate Out AM/PM Toggle":
        remarks.append("Negative Original TAT was corrected by gate-out AM/PM toggle.")
    elif corr_source == "System Bill Date/Time":
        remarks.append("Negative Original TAT was corrected using matched system bill date/time.")
    elif corr_source == "Unresolved_Negative_TAT":
        remarks.append("Negative TAT could not be resolved with allowed corrections.")

    if business_status not in ("OK", "", None):
        if row.get("Final_Considered_Flag") == "Yes":
            remarks.append(f"Business flags: {business_status}; row remains included in final count because Corrected TAT is available.")
        else:
            remarks.append(f"Business flags: {business_status}; row is excluded from final count because Corrected TAT could not be computed.")
    else:
        if match_type == "Exact" and not any_correction:
            if row.get("Final_Considered_Flag") == "Yes":
                remarks.append("No corrections were required; included in final count.")
            else:
                remarks.append("No corrections were required, but row is excluded from final count because Corrected TAT could not be computed.")
        else:
            if row.get("Final_Considered_Flag") == "Yes":
                remarks.append("Included in final count.")
            else:
                remarks.append("Excluded from final count because Corrected TAT could not be computed.")

    return " ".join(remarks)


def build_main_output(
    gate_df: pd.DataFrame,
    sys_df: pd.DataFrame,
    assignments: Dict[int, Dict[str, Any]],
    suggestions: Dict[int, Dict[str, Any]],
    second_pass_reasons: Dict[int, str],
) -> pd.DataFrame:
    sys_lookup = sys_df.set_index("__sys_idx__", drop=False).to_dict(orient="index")
    gate_reg_series = gate_df["Gate_Reg_No_Final"].replace("", np.nan).fillna(gate_df["Gate_Reg_No_Cleaned"]).fillna("")
    gate_reg_counts = gate_reg_series.value_counts(dropna=False).to_dict()
    sys_reg_series = sys_df["Matched_System_Reg_No"].fillna("")
    sys_reg_counts = sys_reg_series.value_counts(dropna=False).to_dict()

    rows: List[Dict[str, Any]] = []

    for i in range(len(gate_df)):
        g = gate_df.iloc[i].to_dict()
        out = {col: None for col in MAIN_OUTPUT_COLUMNS}
        out["S_No"] = i + 1

        for key in [
            "Gate_Reg_No_Raw", "Gate_Reg_No_Cleaned", "Gate_Reg_No_Final",
            "Gate_Reg_Strict_Format_Flag", "Final_Reg_Strict_Format_Flag",
            "Reg_Format_Cleaned_Flag", "Reg_Corrected_via_System_Flag",
            "Reporting_Date_Clean", "Reporting_Time_Clean",
            "Workshop_In_Date_Clean", "Workshop_In_Time_Clean",
            "Workshop_Out_Date_Clean", "Workshop_Out_Time_Clean",
        ]:
            out[key] = g.get(key)

        out["Gate_In_Source"] = g.get("__gate_in_source__")
        out["Base_Gate_In_Date"] = _format_date(_safe_dt_date(g.get("__base_gate_in_dt__")))
        out["Base_Gate_In_Time"] = _format_time(_safe_dt_time(g.get("__base_gate_in_dt__")))

        assn = assignments.get(i)
        matched_sys = None
        match_score = None
        match_time = None
        match_bill = None
        exact = False
        promoted = False

        if assn:
            matched_sys = sys_lookup.get(int(assn["sys_idx"]))
            match_score = float(assn["sim_score"]) if assn.get("sim_score") is not None else None
            match_time = assn.get("time_diff_hours")
            match_bill = assn.get("bill_diff_hours")
            exact = bool(assn.get("exact", False))
            promoted = bool(assn.get("promoted", False))

        # Suggestions: leave blank for final Exact matches to align with the agent style.
        sugg = suggestions.get(i)
        if sugg and (not matched_sys or not exact):
            srow = sys_lookup.get(int(sugg["sys_idx"]))
            out["Top_Suggested_Job_Card_No"] = srow["Matched_Job_Card_No"] if srow else None
            out["Top_Suggested_System_Reg_No"] = srow["Matched_System_Reg_No"] if srow else None
            out["Top_Suggestion_Score"] = _export_round_2(sugg.get("sim_score")) if sugg.get("sim_score") is not None else None
            tdh = sugg.get("time_diff_hours")
            out["Top_Suggestion_Time_Diff_Hours"] = _export_round_2(tdh) if tdh is not None and not pd.isna(tdh) else None

        # System-driven reg correction only on non-exact matched fuzzy rows.
        if matched_sys and not exact and matched_sys["Matched_System_Reg_No"]:
            current_final = out["Gate_Reg_No_Final"] or out["Gate_Reg_No_Cleaned"] or ""
            system_final = matched_sys["Matched_System_Reg_No"] or ""
            if system_final and current_final and system_final != current_final:
                out["Gate_Reg_No_Final"] = system_final
                out["Final_Reg_Strict_Format_Flag"] = "Yes" if bool(matched_sys.get("__system_reg_strict__", False)) or is_strict_reg(system_final) else out["Final_Reg_Strict_Format_Flag"]
                out["Reg_Corrected_via_System_Flag"] = "Yes"

        if matched_sys:
            out["Matched_Job_Card_No"] = matched_sys["Matched_Job_Card_No"]
            out["Matched_System_Reg_No"] = matched_sys["Matched_System_Reg_No"]
            out["Match_Type"] = classify_match_type(match_score, exact=exact, promoted=promoted)
            out["Match_Score"] = _export_round_2(match_score)
            out["Match_Time_Diff_Hours"] = _export_round_2(match_time)
            out["Match_Bill_Diff_Hours"] = _export_round_2(match_bill) if match_bill is not None and not pd.isna(match_bill) else None
            out["Matched_System_Gate_In_Date"] = matched_sys["Matched_System_Gate_In_Date"]
            out["Matched_System_Gate_In_Time"] = matched_sys["Matched_System_Gate_In_Time"]
            out["Matched_System_Bill_Date"] = matched_sys["Matched_System_Bill_Date"]
            out["Matched_System_Bill_Time"] = matched_sys["Matched_System_Bill_Time"]
            out["ROT_Hours"] = _export_rot_value(matched_sys["ROT_Hours"]) if matched_sys["ROT_Hours"] is not None and not pd.isna(matched_sys["ROT_Hours"]) else None
            out["System_Validation_Available_Flag"] = "Yes"
        else:
            out["Match_Type"] = "No Match"
            out["System_Validation_Available_Flag"] = "No"

        base_gate_in_dt = _normalize_datetime_like(g.get("__base_gate_in_dt__"))
        gate_out_dt = _normalize_datetime_like(g.get("__workshop_out_dt__"))

        original_tat = None
        if base_gate_in_dt is not None and gate_out_dt is not None:
            original_tat = _export_round_2(_safe_hours(gate_out_dt - base_gate_in_dt))

        out["Gate_Register_GIGO_TAT_Hours"] = original_tat
        out["Original_TAT_Hours"] = original_tat

        corrected_gate_out_dt = gate_out_dt
        tat_corrected = False
        gate_time_corrected = False
        correction_source = None

        correction_flags: List[str] = []
        if out["Reg_Format_Cleaned_Flag"] == "Yes":
            correction_flags.append("Registration cleaned")
        if out["Reg_Corrected_via_System_Flag"] == "Yes":
            correction_flags.append("Registration corrected via system")

        if original_tat is not None and original_tat < 0:
            toggle_candidates: List[Tuple[float, datetime]] = []
            if gate_out_dt is not None and base_gate_in_dt is not None:
                for delta in (12, -12):
                    cand_dt = gate_out_dt + timedelta(hours=delta)
                    cand_tat = _safe_hours(cand_dt - base_gate_in_dt)
                    if cand_tat is not None and cand_tat >= 0:
                        toggle_candidates.append((float(cand_tat), cand_dt))
            if toggle_candidates:
                toggle_candidates.sort(key=lambda x: (x[0], x[1]))
                corrected_gate_out_dt = toggle_candidates[0][1]
                tat_corrected = True
                gate_time_corrected = True
                correction_source = "Gate Out AM/PM Toggle"
                correction_flags.append("Gate Out AM/PM Toggle")
            elif matched_sys and _normalize_datetime_like(matched_sys.get("__system_bill_dt__")) is not None:
                corrected_gate_out_dt = _normalize_datetime_like(matched_sys.get("__system_bill_dt__"))
                tat_corrected = True
                correction_source = "System Bill Date/Time"
                correction_flags.append("System Bill Date/Time")
            else:
                correction_source = "Unresolved_Negative_TAT"

        corrected_gate_out_dt = _normalize_datetime_like(corrected_gate_out_dt)
        corrected_tat = None
        if base_gate_in_dt is not None and corrected_gate_out_dt is not None:
            corrected_tat = _export_round_2(_safe_hours(corrected_gate_out_dt - base_gate_in_dt))

        out["Corrected_Gate_Out_Date"] = _format_date(_safe_dt_date(corrected_gate_out_dt))
        out["Corrected_Gate_Out_Time"] = _format_time(_safe_dt_time(corrected_gate_out_dt))
        out["Corrected_TAT_Hours"] = corrected_tat
        out["TAT_Corrected_Flag"] = _bool_to_yes_no(tat_corrected)
        out["Gate_Time_Corrected_Flag"] = _bool_to_yes_no(gate_time_corrected)
        out["TAT_Correction_Source"] = correction_source

        rot_hours = _to_float(out["ROT_Hours"])
        if original_tat is not None and rot_hours is not None:
            out["GIGO_TAT_Delay_Hours"] = _export_round_2(original_tat - rot_hours - 4)
        else:
            out["GIGO_TAT_Delay_Hours"] = None

        if corrected_tat is None:
            out["Gate_In_Before_Gate_Out_Flag"] = None
            out["Outlier_Flag"] = None
            out["ROT_Work_Mismatch_Flag"] = None
        else:
            gate_before_out = corrected_tat >= 0
            out["Gate_In_Before_Gate_Out_Flag"] = _bool_to_yes_no(gate_before_out)

            outlier = corrected_tat < 0.25 or corrected_tat > 336
            rot_mismatch = bool(corrected_tat < 0.25 and rot_hours is not None and rot_hours > corrected_tat)
            out["Outlier_Flag"] = _bool_to_yes_no(outlier)
            out["ROT_Work_Mismatch_Flag"] = _bool_to_yes_no(rot_mismatch)

        out["Business_Validation_Status"] = _business_status_agent_style(corrected_tat, rot_hours)
        out["Final_Considered_Flag"] = "Yes" if corrected_tat is not None else "No"
        out["Any_Correction_Flag"] = _bool_to_yes_no(bool(correction_flags))
        out["Correction_Flags"] = "; ".join(correction_flags) if correction_flags else None

        reg_key = out["Gate_Reg_No_Final"] or out["Gate_Reg_No_Cleaned"] or ""
        sys_reg = out.get("Matched_System_Reg_No") or ""
        repeated_exact = bool(
            out["Match_Type"] == "Exact"
            and (
                gate_reg_counts.get(reg_key, 0) > 1
                or sys_reg_counts.get(sys_reg, 0) > 1
            )
        )

        reason_text = second_pass_reasons.get(i)
        out["Final_Remarks"] = _build_row_remark(
            out,
            second_pass_reason=reason_text if matched_sys else None,
            high_conf_rejected_reason=reason_text if not matched_sys else None,
            repeated_exact=repeated_exact,
        )

        rows.append(out)

    main_df = pd.DataFrame(rows, columns=MAIN_OUTPUT_COLUMNS)
    return main_df


def build_summary(gate_df: pd.DataFrame, sys_df: pd.DataFrame, main_df: pd.DataFrame) -> pd.DataFrame:
    gate_timeline_mask = gate_df["__timeline_anchor_date__"].apply(lambda d: _within_operating_window(d) if isinstance(d, date) else False)
    sys_timeline_mask = sys_df["__timeline_anchor_date__"].apply(lambda d: _within_operating_window(d) if isinstance(d, date) else False)

    match_dist = main_df["Match_Type"].fillna("No Match").value_counts()
    matched_mask = main_df["Matched_Job_Card_No"].fillna("").astype(str).str.strip().replace("nan", "") != ""
    final_mask = main_df["Final_Considered_Flag"] == "Yes"

    matched_scores = pd.to_numeric(main_df.loc[matched_mask, "Match_Score"], errors="coerce")
    score_100_share = None
    if matched_scores.notna().any():
        score_100_share = round(float((matched_scores == 100).mean()) * 100, 2)

    gigo_series = pd.to_numeric(main_df["Gate_Register_GIGO_TAT_Hours"], errors="coerce")
    delay_series = pd.to_numeric(main_df["GIGO_TAT_Delay_Hours"], errors="coerce")
    rot_series = pd.to_numeric(main_df["ROT_Hours"], errors="coerce")

    valid_delay_mask = matched_mask & rot_series.notna() & delay_series.notna()
    minor_mask = valid_delay_mask & (rot_series <= 2)

    # Client KPI cards should align to the agent-facing interpretation:
    # timeline count for the first card, then overall valid / matched / averages on the analyzed output.
    client_total_gate = int(gate_timeline_mask.sum())
    client_valid_entries = int(final_mask.sum())
    client_matched = int(matched_mask.sum())
    client_avg_gigo = _hours_value(gigo_series.loc[final_mask].mean()) if final_mask.any() else None
    client_avg_delay = _hours_value(delay_series.loc[valid_delay_mask].mean()) if valid_delay_mask.any() else None

    summary_rows = [
        ("Total records processed", len(main_df)),
        ("Gate Register records in timeline", int(gate_timeline_mask.sum())),
        ("System Extract records in timeline", int(sys_timeline_mask.sum())),
        ("Delta between Gate Register and System Extract", int(gate_timeline_mask.sum()) - int(sys_timeline_mask.sum())),
        ("Unique registrations in Gate Register after correction", int(main_df["Gate_Reg_No_Final"].fillna("").replace("", np.nan).dropna().nunique())),
        ("Unique registrations in System Extract", int(sys_df["Matched_System_Reg_No"].fillna("").replace("", np.nan).dropna().nunique())),
        ("System date/time parse failures", int(sys_df["__system_parse_fail__"].sum())),
        ("Gate Register date parse failures", int(gate_df["__gate_parse_fail__"].sum())),
        ("Records with Original TAT calculated", int(pd.to_numeric(main_df["Original_TAT_Hours"], errors="coerce").notna().sum())),
        ("Records with Corrected TAT calculated", int(pd.to_numeric(main_df["Corrected_TAT_Hours"], errors="coerce").notna().sum())),
        ("Records with system validation available", int((main_df["System_Validation_Available_Flag"] == "Yes").sum())),
        ("Final count", int(final_mask.sum())),
        ("Average calculated TAT from Final count", _hours_value(pd.to_numeric(main_df.loc[final_mask, "Corrected_TAT_Hours"], errors="coerce").mean()) if final_mask.any() else None),
        ("Average GIGO TAT", _hours_value(gigo_series.dropna().mean()) if gigo_series.notna().any() else None),
        ("Average GIGO TAT Delay", _hours_value(delay_series.loc[valid_delay_mask].mean()) if valid_delay_mask.any() else None),
        ("Number of minor jobs", int(minor_mask.sum())),
        ("Average GIGO TAT for minor jobs", _hours_value(gigo_series.loc[minor_mask].mean()) if minor_mask.any() else None),
        ("Average GIGO TAT Delay for minor jobs", _hours_value(delay_series.loc[minor_mask].mean()) if minor_mask.any() else None),
        ("Registration numbers corrected", int((main_df["Reg_Corrected_via_System_Flag"] == "Yes").sum())),
        ("Formatting-only registration cleanup", int((main_df["Reg_Format_Cleaned_Flag"] == "Yes").sum())),
        ("Entries with gate in or gate out details corrected", int((main_df["TAT_Corrected_Flag"] == "Yes").sum())),
        ("Correction impact", int((main_df["Any_Correction_Flag"] == "Yes").sum())),
        ("No confident job card match", int((main_df["Match_Type"] == "No Match").sum())),
        ("Outliers flagged", int((main_df["Outlier_Flag"] == "Yes").sum())),
        ("ROT hour work mismatch", int((main_df["ROT_Work_Mismatch_Flag"] == "Yes").sum())),
        ("Negative Original TAT detected", int((pd.to_numeric(main_df["Original_TAT_Hours"], errors="coerce") < 0).sum())),
        ("Gate out AM/PM toggles applied", int((main_df["TAT_Correction_Source"] == "Gate Out AM/PM Toggle").sum())),
        ("System bill date/time used as corrected gate out", int((main_df["TAT_Correction_Source"] == "System Bill Date/Time").sum())),
        ("Match distribution - Exact", int(match_dist.get("Exact", 0))),
        ("Match distribution - Fuzzy_High", int(match_dist.get("Fuzzy_High", 0))),
        ("Match distribution - Fuzzy_Medium", int(match_dist.get("Fuzzy_Medium", 0))),
        ("Match distribution - No Match", int(match_dist.get("No Match", 0))),
        ("Share of matched records with score 100 percent", score_100_share),
        ("Client KPI - Total gate register vehicles Nov 1 to Dec 15", client_total_gate),
        ("Client KPI - Valid entries from gate register data", client_valid_entries),
        ("Client KPI - Number of JCs matched from system data", client_matched),
        ("Client KPI - Average GIGO TAT for all valid registration numbers", client_avg_gigo),
        ("Client KPI - Average GIGO TAT delay for all matched job cards", client_avg_delay),
    ]
    return pd.DataFrame(summary_rows, columns=["Metric", "Value"])


# Top suggestion selection aligned to the agent:
# - if any exact candidate exists, keep the best exact candidate regardless of time gap
# - otherwise, prefer non-exact suggestions within the 48h / 2-day proximity window;
#   only if none exist, fall back to the full non-exact pool.
def top_suggestions(candidate_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    suggestions: Dict[int, Dict[str, Any]] = {}
    if candidate_df.empty:
        return suggestions

    for gate_idx, grp in candidate_df.groupby("gate_idx", sort=False):
        exact_grp = grp[grp["exact"] == True]
        if not exact_grp.empty:
            ranked = exact_grp.sort_values(
                ["time_diff_hours", "bill_diff_hours", "sys_order"],
                ascending=[True, True, True],
                na_position="last",
            )
            suggestions[int(gate_idx)] = ranked.iloc[0].to_dict()
            continue

        fuzzy_grp = grp.copy()
        near_mask = fuzzy_grp["time_diff_hours"].apply(lambda x: (x is not None and not pd.isna(x) and float(x) <= 48.0))
        pool = fuzzy_grp.loc[near_mask].copy() if near_mask.any() else fuzzy_grp
        ranked = pool.sort_values(
            ["sim_score", "time_diff_hours", "bill_diff_hours", "sys_order"],
            ascending=[False, True, True, True],
            na_position="last",
        )
        if not ranked.empty:
            suggestions[int(gate_idx)] = ranked.iloc[0].to_dict()
    return suggestions
