"""
shelf_life.py  —  Shelf-Life Query Module
==========================================
Reads shelf life data from shelf_life_data.csv.

CSV format:
    Ingredient, Refrigerator_Shelf_Life_Days, Pantry_Shelf_Life_Days

Lookup:
  1. Exact match   — normalised name in CSV
  2. Returns None  — caller must ask user for expiry date
     Once user supplies it, call save_shelf_life() to store it
     permanently in the CSV so future lookups hit step 1.

Environment variables:
    SHELF_LIFE_CSV   Path to CSV (default: shelf_life_data.csv)

Public API:
    get_shelf_life_days(item_name) -> int | None
    save_shelf_life(item_name, fridge_days, pantry_days) -> None
    calculate_expiry(item_name) -> (purchase_date_str, expiry_date_str, days_int) | None
"""

from __future__ import annotations

import csv
import logging
import os
import re
import threading
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional, Dict
from rapidfuzz import process, fuzz

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

_CSV_PATH       = os.environ.get("SHELF_LIFE_CSV", "shelf_life_data.csv")
_FRIDGE_WEIGHT  = 0.7
_PANTRY_WEIGHT  = 0.3

_COL_NAME   = "Ingredient"
_COL_FRIDGE = "Refrigerator_Shelf_Life_Days"
_COL_PANTRY = "Pantry_Shelf_Life_Days"

# Thread-safe write lock for CSV appends
_csv_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# NORMALISATION
# ---------------------------------------------------------------------------

def _normalise(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


# ---------------------------------------------------------------------------
# CSV LOADER — in-memory cache, reloads when file changes on disk
# ---------------------------------------------------------------------------

_csv_cache: list[tuple[str, str, Optional[int], Optional[int]]] = []
_csv_mtime: float = 0.0


def _int_or_none(val: str) -> Optional[int]:
    try:
        v = int(float(str(val).strip()))
        return v if v > 0 else None
    except (ValueError, AttributeError):
        return None


def _load_csv() -> list[tuple[str, str, Optional[int], Optional[int]]]:
    """Load CSV into memory. Deduplicates by normalised name (keeps first row)."""
    if not os.path.exists(_CSV_PATH):
        logger.warning("Shelf-life CSV not found at '%s'.", _CSV_PATH)
        return []

    rows: list[tuple[str, str, Optional[int], Optional[int]]] = []
    seen: set[str] = set()

    try:
        with open(_CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                original = row.get(_COL_NAME, "").strip()
                if not original:
                    continue
                norm = _normalise(original)
                if norm in seen:
                    continue
                seen.add(norm)
                fridge = _int_or_none(row.get(_COL_FRIDGE, ""))
                pantry = _int_or_none(row.get(_COL_PANTRY, ""))
                rows.append((norm, original, fridge, pantry))
    except Exception as e:
        logger.error("Failed to read CSV '%s': %s", _CSV_PATH, e)

    return rows


def _get_csv_data() -> list[tuple[str, str, Optional[int], Optional[int]]]:
    """Return cached CSV data, reloading if the file has changed on disk."""
    global _csv_cache, _csv_mtime
    try:
        mtime = os.path.getmtime(_CSV_PATH)
    except OSError:
        mtime = 0.0

    if not _csv_cache or mtime != _csv_mtime:
        _csv_cache = _load_csv()
        _csv_mtime = mtime
        _lookup_exact.cache_clear()
        logger.debug("CSV reloaded: %d items from '%s'", len(_csv_cache), _CSV_PATH)

    return _csv_cache


# ---------------------------------------------------------------------------
# EXACT MATCH LOOKUP
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2048)
def _lookup_exact(name_lower: str) -> Optional[tuple[Optional[int], Optional[int]]]:
    # Try exact match first for speed
    for norm, _, fridge, pantry in _get_csv_data():
        if norm == name_lower:
            return (fridge, pantry)

    # Try singular/plural matching
    escaped_name = re.escape(name_lower)
    pattern = r'^' + escaped_name + r'(s|es)?$'
    
    for norm, _, fridge, pantry in _get_csv_data():
        # Check if CSV norm is a plural/singular variation of user query
        if re.search(pattern, norm, re.IGNORECASE):
            return (fridge, pantry)
            
        # Check if user query is a plural/singular variation of CSV norm
        rev_pattern = r'^' + re.escape(norm) + r'(s|es)?$'
        if re.search(rev_pattern, name_lower, re.IGNORECASE):
            return (fridge, pantry)

    return None


# ---------------------------------------------------------------------------
# SHELF-LIFE COMPUTATION
# ---------------------------------------------------------------------------

def _compute_days(fridge: Optional[int], pantry: Optional[int]) -> int:
    """
    Weighted average: round(0.7 * fridge + 0.3 * pantry).
    Uses whichever value is non-zero if only one exists.
    """
    f = fridge if (fridge is not None and fridge > 0) else None
    p = pantry if (pantry is not None and pantry > 0) else None

    if f is not None and p is not None:
        return round(_FRIDGE_WEIGHT * f + _PANTRY_WEIGHT * p)
    if f is not None:
        return f
    if p is not None:
        return p
    return 0


# ---------------------------------------------------------------------------
# CATEGORY DEFAULTS (for AI prediction fallback)
# ---------------------------------------------------------------------------
_CATEGORY_DEFAULTS = {
    "dairy": (7, 2),        # milk, yogurt, curd
    "veg": (10, 5),         # general vegetable
    "fruit": (7, 4),        # general fruit
    "meat": (3, 0),         # chicken, beef, etc.
    "grain": (180, 365),    # rice, flour, pulses
    "spice": (365, 730),    # powders, seeds
    "oil": (0, 365),        # sunflower, olive
    "sauce": (180, 365),    # ketchup, jam
    "snack": (0, 180),      # biscuits, chips
    "nonfood": (0, 730),    # soap, toothpaste, detergent
}

_NAME_TO_CATEGORY = {
    "milk": "dairy", "curd": "dairy", "yogurt": "dairy", "paneer": "dairy", "cheese": "dairy", "butter": "dairy",
    "chicken": "meat", "mutton": "meat", "beef": "meat", "pork": "meat", "fish": "meat", "prawn": "meat", "egg": "meat",
    "rice": "grain", "flour": "grain", "dal": "grain", "pulse": "grain", "pasta": "grain", "noodle": "grain", "wheat": "grain",
    "powder": "spice", "masala": "spice", "seed": "spice", "chili": "spice", "turmeric": "spice", "cumin": "spice",
    "oil": "oil", "ghee": "oil",
    "ketchup": "sauce", "sauce": "sauce", "jam": "sauce", "honey": "sauce", "vinegar": "sauce",
    "biscuit": "snack", "chip": "snack", "snack": "snack", "namkeen": "snack", "cookie": "snack",
    "soap": "nonfood", "paste": "nonfood", "detergent": "nonfood", "shampoo": "nonfood", "cleaner": "nonfood",
}


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def get_shelf_life_days(item_name: str) -> Optional[int]:
    """
    Return estimated shelf life in days for the given grocery item.

    Returns:
        int   — days from CSV if found
        None  — item not in CSV; caller should prompt user for expiry date
    """
    normalised = _normalise(item_name)

    # Trigger mtime check / reload
    _get_csv_data()

    exact = _lookup_exact(normalised)
    if exact is not None:
        days = _compute_days(exact[0], exact[1])
        if days > 0:
            logger.debug("CSV exact match '%s' -> %d days", normalised, days)
            return days

    # Not found — return None so caller can ask user
    logger.info("'%s' not found in shelf-life CSV — user input required.", item_name)
    return None


def predict_shelf_life(item_name: str) -> Dict[str, Optional[int]]:
    """
    Predict shelf life for an unknown item using:
    1. Fuzzy match against CSV.
    2. Category keyword mapping.
    3. Safe defaults.
    """
    norm = _normalise(item_name)
    data = _get_csv_data()
    
    # 1. Fuzzy match (threshold 80)
    names = [row[0] for row in data]
    match = process.extractOne(norm, names, scorer=fuzz.token_set_ratio)
    if match and match[1] >= 85:
        # Found a close match
        best_norm = match[0]
        for n, orig, fridge, pantry in data:
            if n == best_norm:
                return {
                    "fridge": fridge,
                    "pantry": pantry,
                    "match": orig,
                    "source": "fuzzy_match"
                }

    # 2. Category mapping
    words = norm.split()
    for word in words:
        if word in _NAME_TO_CATEGORY:
            cat = _NAME_TO_CATEGORY[word]
            f, p = _CATEGORY_DEFAULTS[cat]
            return {
                "fridge": f,
                "pantry": p,
                "match": cat.title(),
                "source": "category"
            }
            
    # 3. Last Resort — common defaults based on storage hints in name
    if "cool" in norm or "chill" in norm or "frozen" in norm:
        return {"fridge": 30, "pantry": 0, "match": "Generic Chilled", "source": "heuristic"}
    
    return {"fridge": 7, "pantry": 30, "match": "General Item", "source": "default"}


def save_shelf_life(item_name: str, fridge_days: Optional[int], pantry_days: Optional[int]) -> None:
    """
    Append a new item to the CSV so future lookups find it instantly.
    Thread-safe. Resets in-memory cache after write.

    Args:
        item_name   : canonical name as entered by user
        fridge_days : refrigerator shelf life in days (or None)
        pantry_days : pantry/room-temp shelf life in days (or None)
    """
    fridge_val = str(fridge_days) if fridge_days and fridge_days > 0 else "0"
    pantry_val = str(pantry_days) if pantry_days and pantry_days > 0 else "0"

    with _csv_write_lock:
        try:
            write_header = not os.path.exists(_CSV_PATH)
            with open(_CSV_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([_COL_NAME, _COL_FRIDGE, _COL_PANTRY])
                writer.writerow([item_name, fridge_val, pantry_val])

            logger.info(
                "Saved '%s' to CSV (fridge=%s pantry=%s)",
                item_name, fridge_val, pantry_val,
            )

            # Reset cache so the new row is found on the very next call
            global _csv_cache, _csv_mtime
            _csv_cache = []
            _csv_mtime = 0.0
            _lookup_exact.cache_clear()

        except Exception as e:
            logger.error("Failed to save '%s' to CSV: %s", item_name, e)


def calculate_expiry(item_name: str) -> Optional[tuple[str, str, int]]:
    """
    Returns (purchase_date_str, expiry_date_str, days_int) if item is in CSV.
    Returns None if item is unknown — caller must ask the user for the expiry date.
    """
    days = get_shelf_life_days(item_name)
    if days is None or days <= 0:
        return None

    today  = datetime.now().date()
    expiry = today + timedelta(days=days)
    return str(today), str(expiry), days


# ---------------------------------------------------------------------------
# CSV HEALTH CHECK
# ---------------------------------------------------------------------------

def csv_stats() -> dict:
    data = _get_csv_data()
    if not data:
        return {"status": "missing_or_empty", "csv_path": _CSV_PATH}
    return {
        "status":      "ok",
        "csv_path":    _CSV_PATH,
        "total_items": len(data),
        "sample":      [row[1] for row in data[:5]],
    }


# ---------------------------------------------------------------------------
# CLI QUICK-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("CSV stats:", csv_stats())
    print()

    test_items = sys.argv[1:] or [
        "Carrots", "Potatoes", "Tomatoes", "Spinach",   # should be found
        "dragon fruit", "kokum", "taro root",            # should return None
    ]

    print(f"{'Item':<30} {'Days':>6}  {'Expiry'}")
    print("-" * 55)
    for item in test_items:
        result = calculate_expiry(item)
        if result:
            purchase, expiry, days = result
            print(f"{item:<30} {days:>6}  {expiry}")
        else:
            print(f"{item:<30}  {'N/A':>6}  (not in CSV — ask user)")