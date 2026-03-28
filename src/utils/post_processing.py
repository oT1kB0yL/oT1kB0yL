import re

from dateutil.parser import parse
from typing import Any, Dict, List, Optional, Tuple

from src.const.keywords_map import REVERSE_LANG_KEY_MAP, MONTH_DICT

# ---------------------------------------------------------------------------
# Precompiled regex patterns
# ---------------------------------------------------------------------------
_RE_MONTH_STRIP    = re.compile(r'[^0-9a-z]')
_RE_RANGE_SPLIT    = re.compile(r'[\-/~--to]+')
_RE_DIGIT_2_4      = re.compile(r'\d{2,4}')
_RE_CLEAN_NUM      = re.compile(r'[,_\s]')
_RE_NUMERIC_KEY    = re.compile(r'[0-9]')
_RE_NON_DIGITS     = re.compile(r'[^0-9]')
_RE_WHITESPACE     = re.compile(r'\s+')
_RE_MILLION_SUFFIX = re.compile(r'\dm(illion)?$', re.IGNORECASE)
_RE_NON_NUM_CHARS  = re.compile(r'[^\-0-9.]')
_RE_ZERO_PREFIX    = re.compile(r'^0')
_RE_CURRENCY_SPLIT = re.compile(r"'|’|\s|\.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALUE_COLS = {
    "net_pay", "commission", "dividend", "revenue", "net_profit",
    "tax_payable", "employment", "total_income", "total_equity",
    "total_liabilities", "deductions", "total_deduction",
}

ABS_COLS = {"dividend"}
RECORD_LIMIT = 6

# Currency aliases: ISO code → set of raw strings the VLM may produce.
# To add a new currency: add one entry here.
_CURRENCY_ALIASES: Dict[str, set] = {
    "CNY": {"元", "人民币", "人民幣", "人民币元", "RMB", "万元", "亿元"},
    "SGD": {"S$", "SS", "S", "DR", "Singapore"},
    "USD": {"US$", "US"},
    "TWD": {"NT$", "NTD", "新臺幣", "新台幣"},
    "IDR": {"rp", "rupiah", "indonesian rupiah", "indonesian"},
    "MYR": {"rm"},
}

# Case-insensitive reverse lookup derived automatically — do not edit directly.
_CURRENCY_CODE_MAP: Dict[str, str] = {
    alias.lower(): code
    for code, aliases in _CURRENCY_ALIASES.items()
    for alias in (aliases | {code})
}

# Built-in scale for certain CNY denominations
_CNY_SCALE: Dict[str, int] = {"万元": 10_000, "亿元": 100_000_000}

# Scale for the unit suffix portion (e.g. "million", "ribu").
# To add a new unit: add one entry to _UNIT_ALIASES; _UNIT_SCALE is derived automatically.
_UNIT_ALIASES: Dict[int, set] = {
    1_000:         {"thousand", "ribu", "ribuan"},
    1_000_000:     {"million", "m"},
    1_000_000_000: {"billion"},
}
_UNIT_SCALE: Dict[str, int] = {
    alias: value
    for value, aliases in _UNIT_ALIASES.items()
    for alias in aliases
}


# ── Number / date utilities ──────────────────────────────────────────────────

def _to_number(x: Any) -> int:
    """Convert a value to an integer, returning 0 on failure."""
    s = str(x).strip()
    if not s or s.lower() in {"none", "nan", ""}:
        return 0
    s = _RE_CLEAN_NUM.sub("", s)
    try:
        return int(s)
    except (TypeError, ValueError):
        return 0


def _coerce_year(value: Any) -> Optional[int]:
    """
    Coerce a value to an integer year.
    - Handles ints and strings with surrounding characters (e.g. '2020年', '年2020').
    - Range-like strings (e.g. '2020-2022', '2020/2022') return the FIRST year.
    - Returns None if no valid digit sequence is found.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        first_token = _RE_RANGE_SPLIT.split(s, maxsplit=1)[0].strip()
        m = _RE_DIGIT_2_4.search(first_token)
        if not m:
            return None
        try:
            return int(m.group(0))
        except ValueError:
            return None
    return None


def _normalize_month(v: str) -> str:
    """Normalize a raw month string to a Jan/Feb/… abbreviation."""
    month_str = _RE_MONTH_STRIP.sub('', str(v).lower()).lstrip("0")
    if len(month_str) > 2 and len(month_str) <= 4:
        try:
            month_str = str(parse(month_str.split("-")[0]).month)
        except (ValueError, OverflowError) as e:
            pass
    return MONTH_DICT.get(month_str, str(v))


# ── Field-level reformatting ─────────────────────────────────────────────────

def _normalize_currency_unit(raw: str) -> Tuple[str, int]:
    """Parse a raw currency_unit string into an ISO code and a scale multiplier.

    Args:
        raw (str): raw currency string from VLM output.

    Returns:
        Tuple[str, int]: (ISO currency code or '', integer scale multiplier).
    """
    parts = _RE_CURRENCY_SPLIT.split(str(raw), maxsplit=1)
    v, unit = parts[0], "".join(parts[1:])

    # If v is itself a scale word or zero-prefix string (e.g. "000", "million"),
    # promote it to unit and clear v BEFORE the code lookup
    if v.lower() in _UNIT_SCALE or _RE_ZERO_PREFIX.match(v):
        unit = v
        v = ""

    # Resolve currency code; pick up any built-in scale (e.g. 万元 → ×10,000)
    code = _CURRENCY_CODE_MAP.get(v.lower(), v)
    scale = _CNY_SCALE.get(v, 1)

    if not code:
        code = ""

    # Resolve scale from the unit suffix
    unit_lower = unit.lower()
    if unit_lower in _UNIT_SCALE:
        scale = _UNIT_SCALE[unit_lower]
    elif _RE_ZERO_PREFIX.match(unit_lower):
        scale = 10 ** unit.count("0")

    return code, scale


def _reformat_value(y_data: dict, key_map: dict) -> dict:
    """Translate keys and normalise all field values in one VLM record.

    Args:
        y_data (dict): raw per-record dict from VLM output.
        key_map (dict): foreign-language → English key mapping.

    Returns:
        dict: normalised record with English keys.
    """
    year_res: dict = {}
    mul_unit, mul_val = 1, 1

    for k, v in y_data.items():
        k = _RE_NUMERIC_KEY.sub("", k)
        eng_key = key_map.get(k, k).lower()

        if eng_key == "date":
            continue

        if not v:
            v = ""
        elif eng_key == "year":
            v = _RE_NON_DIGITS.sub('', str(v))
            if v and int(v) < 1000:
                v = str(int(v) + 1911)  # Minguo → Gregorian
            v = v[:4]
        elif eng_key == "month":
            v = _normalize_month(str(v))
        elif eng_key == "currency_unit":
            v, mul_unit = _normalize_currency_unit(v)
        elif eng_key in VALUE_COLS:
            v = _RE_WHITESPACE.sub("", str(v))
            mul_val = (
                10_000 if "万" in v
                else 1_000_000 if _RE_MILLION_SUFFIX.search(v)
                else 100_000_000 if "亿" in v
                else 1
            )
            if v:
                v_str = _RE_NON_NUM_CHARS.sub("", v)
                try:
                    val = round(float(v_str) * mul_val)
                    val = abs(val) if eng_key in ABS_COLS else val
                except (ValueError, TypeError):
                    val = 0
                v = f"{val:,}"

        year_res[eng_key] = v

    # Apply currency-unit scale to all value columns (e.g. amounts stated in millions)
    if mul_val == 1 and mul_unit != 1:
        for k in VALUE_COLS:
            if k in year_res:
                raw = str(year_res[k]).replace(",", "")
                if raw:
                    year_res[k] = f"{mul_unit * int(raw):,}"

    return year_res


# ── Doc-type heuristics ──────────────────────────────────────────────────────

def _correct_noa_total(vlm_res: List[dict]) -> List[dict]:
    """Correct NOA records where total_income < employment (common mis-extraction)."""
    result = []
    for res in vlm_res:
        nums = {
            col: int(res.get(col, "").replace(",", "") or 0)
            for col in ("total_income", "employment", "deductions")
        }
        if nums["total_income"] < nums["employment"] <= nums["total_income"] + nums["deductions"]:
            res["total_income"] = f'{nums["total_income"] + nums["deductions"]:,}'
        result.append(res)
    return result


def _correct_financial_equity(vlm_res: List[dict], bboxes: dict) -> List[dict]:
    """Correct total_equity using neighbouring OCR bounding-box values (ZHS only)."""
    if "equity" not in bboxes:
        return vlm_res
    equity_boxes = bboxes["equity"][0].get("equity", {})
    result = []
    for row in vlm_res:
        if "total_equity" not in row or "total_liabilities" not in row:
            result.append(row)
            continue
        equity_val = int(row["total_equity"].replace(",", "") or "0") or 0
        liab_val = int(row["total_liabilities"].replace(",", "") or "0") or 0
        if equity_val in equity_boxes:
            direction = "up" if equity_val == liab_val else "cur"
            equity_val = equity_boxes[equity_val].get(direction, equity_val)
            row["total_equity"] = f"{equity_val:,}"
        result.append(row)
    return result


def _normalize_year_range(
    combined_record: List[Dict[str, Any]],
    logger: Any = None,
) -> List[dict]:
    """Expand a start_year/end_year range into one record per year (TW payslip).

    Returns the original list unchanged if start_year is missing or invalid.
    """
    if not combined_record:
        return combined_record

    rec = combined_record[0] if isinstance(combined_record[0], dict) else {}
    start_raw, end_raw = rec.get("start_year"), rec.get("end_year")

    start = _coerce_year(start_raw)
    end = _coerce_year(end_raw) if end_raw not in (None, "") else None

    if start is None:
        return combined_record

    if end is None:
        end = start

    if start > end:
        return []

    payload = {k: v for k, v in rec.items() if k not in ("start_year", "end_year", "year", "month")}
    return [{"year": y, "month": "", **payload} for y in range(end, start - 1, -1)][:RECORD_LIMIT]


def _normalize_net_pay(vlm_res: List[dict]) -> List[dict]:
    """Convert TW employment-proof monthly salary to annualised net pay."""
    out = []
    for r in vlm_res:
        monthly_salary = r.get("monthly_salary")
        net_pay = r.get("net_pay")
        if monthly_salary:
            net_pay = _to_number(monthly_salary) * 12 + _to_number(r.get("commission"))
        out.append({
            "year": r.get("year"),
            "month": r.get("month"),
            "net_pay": str(_to_number(net_pay)),
            "currency_unit": r.get("currency_unit"),
        })
    return out


def _correct_tw_payslip(vlm_res: List[dict], _bboxes: dict) -> List[dict]:
    """Heuristic pipeline for ZHT payslips: year-range expansion + net-pay normalisation."""
    return _normalize_net_pay(_normalize_year_range(vlm_res))


# Registry: (doc_type, lang) → handler(vlm_res, bboxes) -> List[dict]
# lang=None acts as a wildcard matching any language for that doc_type.
# To add a new correction: define a handler function and add one entry here.
_HEURISTIC_REGISTRY: Dict[tuple, Any] = {
    ("noa",                  None):  lambda r, b: _correct_noa_total(r),
    ("financial_statement", "ZHS"):  _correct_financial_equity,
    ("payslip",             "ZHT"):  _correct_tw_payslip,
}


def _heuristics(vlm_res: List[dict], lang: str, doc_type: str, bboxes: dict) -> List[dict]:
    """Dispatch to the appropriate doc-type/language heuristic correction."""
    handler = (
        _HEURISTIC_REGISTRY.get((doc_type, lang))
        or _HEURISTIC_REGISTRY.get((doc_type, None))
    )
    return handler(vlm_res, bboxes) if handler else vlm_res


# ── Public API ───────────────────────────────────────────────────────────────

def _sort_key(r: dict) -> tuple:
    """Sort key: valid years first (descending), records without a year last."""
    y = _coerce_year(r.get("year"))
    return (0, y) if y is not None else (1, 0)


def normalize_fields(vlm_json: dict, origin_lang: str, doc_type: str, bboxes: dict) -> List[dict]:
    """Normalize VLM extraction results into a list of year-keyed English records.

    Steps:
      1. Translate foreign-language field names to English.
      2. Clean and normalise fields (dates, currencies, numeric values).
      3. Merge supplementary page extractions into the main records.
      4. Apply doc-type/language heuristic corrections.

    Args:
        vlm_json (dict): raw VLM results, keyed by page_type ('main', 'dividend', …).
        origin_lang (str): detected document language code.
        doc_type (str): document type ('payslip', 'financial_statement', 'noa', …).
        bboxes (dict): OCR bounding-box data for numeric value verification.

    Returns:
        List[dict]: normalised per-year records in English.
    """
    if not isinstance(vlm_json, dict):
        return [{"error": str(vlm_json)}]

    key_map = REVERSE_LANG_KEY_MAP.get(origin_lang, {})

    # Process main fields
    res: Dict[str, dict] = {}
    for y_data in sorted(vlm_json.get("main", {}), key=_sort_key, reverse=True)[:RECORD_LIMIT]:
        year_res = _reformat_value(y_data, key_map)
        year = str(year_res.get("year", "")) + str(year_res.get("month", ""))
        if year:
            res[year] = year_res

    # Merge supplementary page types into the main records
    for page_type, records in vlm_json.items():
        if page_type == "main":
            continue
        page_res: Dict[str, dict] = {}
        for y_data in sorted(records, key=_sort_key, reverse=True)[:RECORD_LIMIT]:
            year_res = _reformat_value(y_data, key_map)
            page_res[str(year_res.get("year", ""))] = year_res

        for year, main_record in res.items():
            additional_field = page_res.get(year, page_res.get("", {}))
            for k, v in additional_field.items():
                if k not in main_record:
                    res[year][k] = v

    return _heuristics(list(res.values()), origin_lang, doc_type, bboxes)
