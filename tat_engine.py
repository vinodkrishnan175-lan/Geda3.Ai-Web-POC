
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

ENGINE_VERSION = "2026-03-19-hotfix3"

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
        ["reg", "no"],
        ["vehicle", "no"],
        ["vehicle", "number"],
        ["vehicle", "reg"],
        ["truck", "no"],
        ["plate"],
    ],
    "reporting_datetime": [["reporting", "datetime"], ["reporting", "date", "time"]],
    "reporting_date": [["reporting", "date"], ["report", "date"]],
    "reporting_time": [["reporting", "time"], ["report", "time"]],
    "workshop_in_datetime": [["workshop", "in", "datetime"], ["gate", "in", "datetime"], ["in", "datetime"]],
    "workshop_in_date": [["workshop", "in", "date"], ["gate", "in", "date"], ["in", "date"]],
    "workshop_in_time": [["workshop", "in", "time"], ["gate", "in", "time"], ["in", "time"]],
    "workshop_out_datetime": [["workshop", "out", "datetime"], ["gate", "out", "datetime"], ["out", "datetime"]],
    "workshop_out_date": [["workshop", "out", "date"], ["gate", "out", "date"], ["out", "date"]],
    "workshop_out_time": [["workshop", "out", "time"], ["gate", "out", "time"], ["out", "time"]],
}

SYSTEM_HINTS = {
    "reg": [
        ["registration"],
        ["reg", "no"],
        ["vehicle", "no"],
        ["vehicle", "number"],
        ["plate"],
        ["chassis", "reg"],  # defensive, low match
    ],
    "job_card": [["job", "card"], ["jc"], ["repair", "order"], ["ro", "no"], ["invoice", "no"]],
    "gate_in_datetime": [["gate", "in", "datetime"], ["vehicle", "in", "datetime"], ["check", "in", "datetime"]],
    "gate_in_date": [["gate", "in", "date"], ["vehicle", "in", "date"]],
    "gate_in_time": [["gate", "in", "time"], ["vehicle", "in", "time"]],
    "reporting_datetime": [["reporting", "datetime"], ["job", "open", "datetime"], ["opening", "datetime"]],
    "reporting_date": [["reporting", "date"], ["job", "open", "date"]],
    "reporting_time": [["reporting", "time"], ["job", "open", "time"]],
    "bill_datetime": [["bill", "datetime"], ["billing", "datetime"], ["invoice", "datetime"], ["close", "datetime"]],
    "bill_date": [["bill", "date"], ["billing", "date"], ["invoice", "date"], ["close", "date"]],
    "bill_time": [["bill", "time"], ["billing", "time"], ["invoice", "time"], ["close", "time"]],
    "rot_hours": [["rot"], ["repair", "time"], ["labour", "hours"], ["work", "hours"]],
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


def read_table(file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    path = Path(file_path)
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        chosen_sheet = 0 if sheet_name in (None, "") else sheet_name
        return pd.read_excel(file_path, sheet_name=chosen_sheet, engine=None)
    return pd.read_csv(file_path)


def _norm_header(col: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(col).strip().lower()).strip()


def _best_header_match(headers: List[str], pattern_groups: List[List[str]]) -> Optional[str]:
    best_col = None
    best_score = -1
    for col in headers:
        norm = _norm_header(col)
        score = 0
        for group in pattern_groups:
            if all(word in norm for word in group):
                score = max(score, len(group) * 10)
                # prefer exact-ish header
                joined = " ".join(group)
                if norm == joined:
                    score += 5
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


def _fallback_reg_column(df: pd.DataFrame) -> Optional[str]:
    best_col = None
    best_score = 0.0
    for col in df.columns:
        share = _registration_like_share(df[col])
        if share > best_score:
            best_col = col
            best_score = share
    return best_col if best_score >= 0.3 else None


def _fallback_numeric_hours_column(df: pd.DataFrame) -> Optional[str]:
    best_col = None
    best_score = 0.0
    for col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        share = float(vals.notna().mean())
        if share > best_score:
            best_col = col
            best_score = share
    return best_col if best_score >= 0.5 else None


def detect_columns(df: pd.DataFrame, hints: Dict[str, List[List[str]]], kind: str) -> Dict[str, Optional[str]]:
    headers = list(df.columns)
    mapping: Dict[str, Optional[str]] = {}
    for target, pattern_groups in hints.items():
        mapping[target] = _best_header_match(headers, pattern_groups)

    if not mapping.get("reg"):
        mapping["reg"] = _fallback_reg_column(df)

    if kind == "system" and not mapping.get("rot_hours"):
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
    alnum = raw_alnum
    if len(alnum) > 10:
        alnum = _best_reg_substring(alnum)
    cleaned = alnum
    final = _positional_reg_fix(cleaned)
    strict_before = is_strict_reg(cleaned)
    strict_after = is_strict_reg(final)
    format_cleaned = (final != raw_alnum) or (raw_text != raw_text.upper()) or bool(re.search(r"[^A-Za-z0-9]", raw_text))
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
        gate_in_source = "Workshop_In" if win_dt_i is not None else ("Reporting" if rep_dt_i is not None else None)
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
        parse_fail_any = gate_fail[i] or rep_fail[i] or bill_fail[i]
        gate_dt_i = _normalize_datetime_like(gate_dt[i])
        rep_dt_i = _normalize_datetime_like(rep_dt[i])
        bill_dt_i = _normalize_datetime_like(bill_dt[i])
        timeline_anchor = _safe_dt_date(gate_dt_i) or _safe_dt_date(rep_dt_i) or _safe_dt_date(bill_dt_i)
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


def _bill_diff_hours(gate_row: pd.Series, sys_row: pd.Series) -> Optional[float]:
    gate_out = gate_row["__workshop_out_dt__"]
    sys_bill = sys_row["__system_bill_dt__"]
    return _time_diff_hours(gate_out, sys_bill)


def classify_match_type(score: Optional[float], exact: bool, promoted: bool = False) -> str:
    if score is None:
        return "No Match"
    if promoted:
        return "Second_Pass_Promoted"
    if exact:
        return "Exact"
    if score >= 90:
        return "Fuzzy_High"
    if score >= 80:
        return "Fuzzy_Medium"
    return "No Match"


def build_candidate_table(gate_df: pd.DataFrame, sys_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, g in gate_df.iterrows():
        g_reg = g["Gate_Reg_No_Final"] or g["Gate_Reg_No_Cleaned"]
        if not g_reg:
            continue
        g_anchor = _candidate_time_anchor(g)
        for _, s in sys_df.iterrows():
            if s["__system_parse_fail__"]:
                continue
            s_reg = s["__system_reg_final__"] or s["__system_reg_cleaned__"]
            if not s_reg:
                continue
            sim = float(fuzz.ratio(g_reg, s_reg))
            exact = (g_reg == s_reg and g_reg != "")
            time_diff = _time_diff_hours(g_anchor, _system_time_anchor(s))
            bill_diff = _bill_diff_hours(g, s)

            valid = False
            if exact:
                # allow wider exact window, but still reject wildly implausible visit pairings
                if time_diff is None or time_diff <= 14 * 24:
                    valid = True
            else:
                if sim >= 80 and time_diff is not None and time_diff <= 48:
                    valid = True

            if not valid:
                continue

            # composite reward:
            # registration score dominates, but materially better time alignment can beat a tiny similarity edge
            reward = sim * 100.0
            if time_diff is not None:
                reward -= time_diff * 3.0
            if bill_diff is not None:
                reward -= bill_diff * 0.25
            if exact:
                reward += 25.0

            rows.append(
                {
                    "gate_idx": int(g["__gate_idx__"]),
                    "sys_idx": int(s["__sys_idx__"]),
                    "sim_score": round(sim, 2),
                    "exact": exact,
                    "time_diff_hours": time_diff,
                    "bill_diff_hours": bill_diff,
                    "reward": round(reward, 4),
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
    if n_gate == 0:
        return assignments
    if candidate_df.empty or n_sys == 0:
        return assignments

    # add one dummy column per gate row to allow unmatched assignment
    cost = np.full((n_gate, n_sys + n_gate), 0.0)
    cost[:, :n_sys] = 10_000.0  # large default for invalid real edges
    for _, row in candidate_df.iterrows():
        gi = int(row["gate_idx"])
        si = int(row["sys_idx"])
        # lower cost is better
        cost[gi, si] = -float(row["reward"])

    # dummy columns cost 0 => chosen unless a valid match improves total reward
    row_ind, col_ind = linear_sum_assignment(cost)
    chosen = {(int(r), int(c)) for r, c in zip(row_ind, col_ind)}

    # map back only real system matches
    indexed = candidate_df.set_index(["gate_idx", "sys_idx"]).sort_index()
    for gi, ci in chosen:
        if ci >= n_sys:
            continue
        if (gi, ci) not in indexed.index:
            continue
        rec = indexed.loc[(gi, ci)]
        if isinstance(rec, pd.DataFrame):
            # impossible but safe
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
    gate_to_sys = {g: int(v["sys_idx"]) for g, v in assignments.items()}
    sys_to_gate = {int(v["sys_idx"]): g for g, v in assignments.items()}

    def _priority_tuple(cand: Dict[str, Any]) -> Tuple[float, float, float, float]:
        return (
            float(cand.get("sim_score") or 0),
            -float(cand.get("time_diff_hours") or 999999),
            -float(cand.get("bill_diff_hours") or 999999),
            -float(cand.get("sys_order") or 999999),
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
            gate_to_sys[gi] = si
            sys_to_gate[si] = gi
            reasons[gi] = "Promoted in second-pass because it was a valid free one-to-one high-confidence suggestion."
            continue
        current_gate = sys_to_gate[si]
        current = assignments[current_gate]
        if _priority_tuple(sugg) > _priority_tuple(current):
            reasons[gi] = "Promoted in second-pass by replacing a weaker competing row for the same system record."
            reasons[current_gate] = "High-confidence suggestion was displaced by a stronger competing row during second-pass conflict resolution."
            demoted = assignments.pop(current_gate)
            promoted = dict(sugg)
            promoted["promoted"] = True
            assignments[gi] = promoted
            sys_to_gate[si] = gi
        else:
            reasons[gi] = "Not promoted in second-pass because a stronger competing row already owned the same system record."
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
    return round(num, 4) if num is not None else None


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
    if row["Reg_Corrected_via_System_Flag"] == "Yes":
        remarks.append("Registration cleaned and corrected via matched system record.")
    elif row["Reg_Format_Cleaned_Flag"] == "Yes":
        remarks.append("Registration cleaned for formatting and OCR noise.")
    else:
        remarks.append("Registration required no cleanup beyond standardization.")

    mt = row["Match_Type"]
    if mt == "No Match":
        remarks.append("No confident job card match.")
    else:
        remarks.append(f"Match type: {mt} with score {row['Match_Score']}.")

    top_score = row["Top_Suggestion_Score"]
    top_tdiff = row["Top_Suggestion_Time_Diff_Hours"]
    if top_score not in (None, ""):
        try:
            if float(top_score) > 85 and top_tdiff not in (None, "") and float(top_tdiff) <= 48:
                if row["Match_Type"] == "Second_Pass_Promoted":
                    remarks.append(second_pass_reason or "High-confidence suggestion was promoted in second-pass review.")
                elif mt == "No Match":
                    remarks.append(high_conf_rejected_reason or "High-confidence suggestion was reviewed but not promoted because of a stronger competing one-to-one claim.")
        except Exception:
            pass

    if row["Gate_Time_Corrected_Flag"] == "Yes":
        remarks.append("Gate-out time was corrected by AM/PM toggle.")
    if row["TAT_Correction_Source"] == "System_BillDate_As_GateOut":
        remarks.append("Matched system bill date/time was used as corrected gate-out.")
    elif row["TAT_Correction_Source"] == "Gate_Out_AMPM_Toggled":
        remarks.append("Negative TAT was resolved using gate-out AM/PM toggle.")
    elif row["TAT_Correction_Source"] == "No_Correction_Needed":
        remarks.append("No gate-time correction was required.")
    elif row["TAT_Correction_Source"] == "Unresolved_Negative_TAT":
        remarks.append("Negative TAT could not be resolved with allowed corrections.")

    if row["Business_Validation_Status"] not in ("OK", "", None):
        remarks.append(f"Business flags: {row['Business_Validation_Status']}.")
    if row["Final_Considered_Flag"] == "Yes":
        remarks.append("Row remains included in final count.")
    else:
        remarks.append("Row is excluded from final count because corrected TAT could not be computed.")
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
    used_match_count = 0

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

        # populate suggestion fields first
        sugg = suggestions.get(i)
        high_conf_rejected_reason = None
        if sugg:
            srow = sys_lookup.get(int(sugg["sys_idx"]))
            out["Top_Suggested_Job_Card_No"] = srow["Matched_Job_Card_No"] if srow else None
            out["Top_Suggested_System_Reg_No"] = srow["Matched_System_Reg_No"] if srow else None
            out["Top_Suggestion_Score"] = _hours_value(float(sugg["sim_score"]))
            out["Top_Suggestion_Time_Diff_Hours"] = _hours_value(float(sugg["time_diff_hours"])) if sugg["time_diff_hours"] is not None else None

        assn = assignments.get(i)
        matched_sys = None
        match_score = None
        match_time = None
        match_bill = None
        exact = False
        promoted = False

        if assn:
            matched_sys = sys_lookup.get(int(assn["sys_idx"]))
            used_match_count += 1
            match_score = float(assn["sim_score"])
            match_time = assn["time_diff_hours"]
            match_bill = assn["bill_diff_hours"]
            exact = bool(assn["exact"])
            promoted = bool(assn.get("promoted", False))

        # optional registration correction via matched system record
        if matched_sys and out["Final_Reg_Strict_Format_Flag"] == "No" and matched_sys["Matched_System_Reg_No"] and matched_sys["__system_reg_strict__"] and match_score and match_score >= 85:
            out["Gate_Reg_No_Final"] = matched_sys["Matched_System_Reg_No"]
            out["Final_Reg_Strict_Format_Flag"] = "Yes"
            out["Reg_Corrected_via_System_Flag"] = "Yes"

        # matched system details
        if matched_sys:
            out["Matched_Job_Card_No"] = matched_sys["Matched_Job_Card_No"]
            out["Matched_System_Reg_No"] = matched_sys["Matched_System_Reg_No"]
            out["Match_Type"] = classify_match_type(match_score, exact=exact, promoted=promoted)
            out["Match_Score"] = round(match_score, 2) if match_score is not None else None
            out["Match_Time_Diff_Hours"] = _hours_value(match_time)
            out["Match_Bill_Diff_Hours"] = _hours_value(match_bill) if match_bill is not None else None
            out["Matched_System_Gate_In_Date"] = matched_sys["Matched_System_Gate_In_Date"]
            out["Matched_System_Gate_In_Time"] = matched_sys["Matched_System_Gate_In_Time"]
            out["Matched_System_Bill_Date"] = matched_sys["Matched_System_Bill_Date"]
            out["Matched_System_Bill_Time"] = matched_sys["Matched_System_Bill_Time"]
            out["ROT_Hours"] = _hours_value(matched_sys["ROT_Hours"]) if matched_sys["ROT_Hours"] is not None else None
            out["System_Validation_Available_Flag"] = "Yes"
        else:
            out["Match_Type"] = "No Match"
            out["System_Validation_Available_Flag"] = "No"
            if sugg and float(sugg["sim_score"]) > 85 and sugg["time_diff_hours"] is not None and float(sugg["time_diff_hours"]) <= 48:
                high_conf_rejected_reason = second_pass_reasons.get(i) or "High-confidence suggestion was reviewed but not promoted because of a stronger competing row or one-to-one conflict."

        base_gate_in_dt = _normalize_datetime_like(g["__base_gate_in_dt__"])
        gate_out_dt = _normalize_datetime_like(g["__workshop_out_dt__"])

        original_tat = None
        if base_gate_in_dt is not None and gate_out_dt is not None:
            original_tat = _safe_hours(gate_out_dt - base_gate_in_dt)

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
        else:
            correction_source = "No_Correction_Needed"

        corrected_gate_out_dt = _normalize_datetime_like(corrected_gate_out_dt)
        corrected_tat = None
        if base_gate_in_dt is not None and corrected_gate_out_dt is not None:
            corrected_tat = _safe_hours(corrected_gate_out_dt - base_gate_in_dt)

        out["Corrected_Gate_Out_Date"] = _format_date(_safe_dt_date(corrected_gate_out_dt))
        out["Corrected_Gate_Out_Time"] = _format_time(_safe_dt_time(corrected_gate_out_dt))
        out["Corrected_TAT_Hours"] = corrected_tat
        out["TAT_Corrected_Flag"] = _bool_to_yes_no(tat_corrected)
        out["Gate_Time_Corrected_Flag"] = _bool_to_yes_no(gate_time_corrected)
        out["TAT_Correction_Source"] = correction_source

        rot_hours = _to_float(out["ROT_Hours"])
        if original_tat is not None and rot_hours is not None:
            out["GIGO_TAT_Delay_Hours"] = _hours_value(original_tat - rot_hours - 4)
        else:
            out["GIGO_TAT_Delay_Hours"] = None

        gate_before_out = corrected_tat is not None and corrected_tat >= 0
        out["Gate_In_Before_Gate_Out_Flag"] = _bool_to_yes_no(gate_before_out)

        outlier = False
        rot_mismatch = False
        business_msgs: List[str] = []
        if corrected_tat is None:
            business_msgs.append("Corrected TAT unavailable")
        else:
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
        out["Business_Validation_Status"] = "OK" if not business_msgs else "; ".join(business_msgs)
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
        ("Final count", int((main_df["Final_Considered_Flag"] == "Yes").sum())),
        ("Average calculated TAT from Final count", round(pd.to_numeric(main_df.loc[main_df["Final_Considered_Flag"] == "Yes", "Corrected_TAT_Hours"], errors="coerce").mean(), 4) if (main_df["Final_Considered_Flag"] == "Yes").any() else None),
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
        ("Match distribution - Second_Pass_Promoted", int(match_dist.get("Second_Pass_Promoted", 0))),
        ("Match distribution - No Match", int(match_dist.get("No Match", 0))),
        ("Share of matched records with score 100 percent", score_100_share),
        # client-facing KPIs
        ("Client KPI - Total gate register vehicles Nov 1 to Dec 15", int(gate_timeline_mask.sum())),
        ("Client KPI - Valid entries from gate register data", int((main_df["Final_Considered_Flag"] == "Yes").sum())),
        ("Client KPI - Number of JCs matched from system data", int(main_df["Matched_Job_Card_No"].fillna("").replace("", np.nan).notna().sum())),
        ("Client KPI - Average GIGO TAT for all valid registration numbers", round(pd.to_numeric(main_df.loc[main_df["Final_Considered_Flag"] == "Yes", "Corrected_TAT_Hours"], errors="coerce").mean(), 4) if (main_df["Final_Considered_Flag"] == "Yes").any() else None),
        ("Client KPI - Average GIGO TAT delay for all matched job cards", round(pd.to_numeric(main_df.loc[main_df["Matched_Job_Card_No"].fillna('').replace('', np.nan).notna(), "GIGO_TAT_Delay_Hours"], errors="coerce").mean(), 4) if main_df["Matched_Job_Card_No"].fillna("").replace("", np.nan).notna().any() else None),
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

    gate_raw = read_table(gate_path, sheet_name=gate_sheet)
    sys_raw = read_table(sys_path, sheet_name=sys_sheet)

    gate_df, gate_mapping = standardize_gate(gate_raw)
    sys_df, sys_mapping = standardize_system(sys_raw)

    candidate_df = build_candidate_table(gate_df, sys_df)
    suggestions = top_suggestions(candidate_df)
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
