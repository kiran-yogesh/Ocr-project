
from __future__ import annotations

import ast
import sqlite3
import numpy as np
import pandas as pd

from recommender import (
    compute_scores,
    evaluate_recommendation_system,
    load_inventory,
    df as recipe_df,
    PROPOSED_SCORE_THRESHOLD,
    PROPOSED_RELEVANCE_THRESHOLD,
    BASELINE_SCORE_THRESHOLD,
    BASELINE_RELEVANCE_THRESHOLD,
)

# Try to import run_baseline_scores — fall back gracefully if missing
try:
    from baseline_model import run_baseline_scores
    _BASELINE_RETURNS_SCORES = True
except ImportError:
    _BASELINE_RETURNS_SCORES = False


# ---------------------------------------------------------------------------
# FIX-C2: Helper to load raw (un-normalised) inventory for the baseline.
# FIX-CMP-5: tz_localize(None) for naive datetime arithmetic.
# ---------------------------------------------------------------------------

def _load_raw_inventory(username: str) -> tuple[list[str], dict[str, int]]:
    """
    Load inventory without the recommender's normalize_ingredient() pass.
    Returns (raw_items, raw_inventory_dict) for use by the TF-IDF baseline
    which does its own clean_text() lowercasing.
    """
    conn = sqlite3.connect("grocery_ocr.db")
    df_inv = pd.read_sql_query(
        "SELECT item, expiry_date FROM grocery_items WHERE user_id = ?",
        conn, params=(username,)
    )
    conn.close()

    if df_inv.empty:
        return [], {}

    df_inv["expiry_date"] = pd.to_datetime(df_inv["expiry_date"], errors="coerce")
    # FIX-CMP-5: strip timezone info if present
    if hasattr(df_inv["expiry_date"].dtype, "tz") and df_inv["expiry_date"].dtype.tz is not None:
        df_inv["expiry_date"] = df_inv["expiry_date"].dt.tz_localize(None)
    df_inv = df_inv.dropna(subset=["expiry_date"])

    today = pd.Timestamp.now().normalize()
    df_inv["days_left"]  = (df_inv["expiry_date"] - today).dt.days
    df_inv["item_lower"] = df_inv["item"].str.lower().str.strip()
    df_inv = df_inv.groupby("item_lower", as_index=False)["days_left"].min()

    raw_dict  = dict(zip(df_inv["item_lower"], df_inv["days_left"]))
    raw_items = list(raw_dict.keys())
    return raw_items, raw_dict


def run_full_comparison(username: str) -> dict:
    """
    Score all recipes for both models and evaluate each with its own
    calibrated thresholds.
    """
    inventory_items, inventory_dict, inventory_text = load_inventory(username)

    if not inventory_items:
        return _empty_comparison()

    # ── Proposed model ───────────────────────────────────────────────────────
    proposed_scores_df = compute_scores(
        recipe_df.copy(), inventory_items, inventory_dict, inventory_text
    )
    proposed_metrics = evaluate_recommendation_system(
        proposed_scores_df,
        inventory_items=inventory_items,
        inventory_dict=inventory_dict,
        k=50,
        relevance_threshold=PROPOSED_RELEVANCE_THRESHOLD,
        score_threshold=PROPOSED_SCORE_THRESHOLD,
    )

    # ── Baseline model ───────────────────────────────────────────────────────
    # FIX-C2: Use raw (un-normalised) inventory for TF-IDF baseline
    raw_items, raw_inv_dict = _load_raw_inventory(username)

    if _BASELINE_RETURNS_SCORES:
        baseline_scores_df = run_baseline_scores(
            recipe_df.copy(), raw_items, raw_inv_dict
        )
    else:
        baseline_scores_df = _baseline_score_inline(
            recipe_df.copy(), raw_items, raw_inv_dict
        )

    baseline_eval_items = raw_items if raw_items else inventory_items
    baseline_eval_dict  = raw_inv_dict if raw_inv_dict else inventory_dict

    baseline_metrics = evaluate_recommendation_system(
        baseline_scores_df,
        inventory_items=baseline_eval_items,
        inventory_dict=baseline_eval_dict,
        k=50,
        relevance_threshold=BASELINE_RELEVANCE_THRESHOLD,
        score_threshold=BASELINE_SCORE_THRESHOLD,
    )

    # ── Exact Match model ────────────────────────────────────────────────────
    exact_scores_df = _exact_match_score_inline(
        recipe_df.copy(), raw_items, raw_inv_dict
    )
    exact_metrics = evaluate_recommendation_system(
        exact_scores_df,
        inventory_items=baseline_eval_items,
        inventory_dict=baseline_eval_dict,
        k=50,
        relevance_threshold=BASELINE_RELEVANCE_THRESHOLD,
        score_threshold=BASELINE_SCORE_THRESHOLD,
    )

    # ── Delta table ──────────────────────────────────────────────────────────
    compared_keys = [
        "precision", "recall", "f1", "accuracy",
        "ndcg_8", "ndcg_k", "precision_k", "recall_k", "f1_k",
        "ap_k", "mrr", "hit_rate_k",
        "catalog_coverage", "fwri", "weighted_expiry",
    ]
    deltas = {}
    for key in compared_keys:
        b_val = baseline_metrics.get(key)
        p_val = proposed_metrics.get(key)
        # FIX-CMP-4: only compute delta when both values are numeric
        if b_val is not None and p_val is not None:
            try:
                deltas[key] = round(float(p_val) - float(b_val), 3)
            except (TypeError, ValueError):
                deltas[key] = None
        else:
            deltas[key] = None

    return {
        "baseline": baseline_metrics,
        "proposed": proposed_metrics,
        "exact":    exact_metrics,
        "deltas":   deltas,
        "baseline_reach": f"{baseline_metrics.get('catalog_coverage', 0) * 100:.1f}%",
        "proposed_reach": f"{proposed_metrics.get('catalog_coverage', 0) * 100:.1f}%",
    }


def _baseline_score_inline(
    df_raw: pd.DataFrame,
    inventory_items: list[str],
    inventory_dict: dict[str, int],
) -> pd.DataFrame:
    """
    Minimal TF-IDF scoring used when baseline_model.py cannot be imported.
    Mirrors the logic in baseline_model._score_recipes().
    FIX-C1: eval() wrapped in _safe_eval().
    FIX-CMP-2: zero-vector cosine suppressed with numpy errstate.
    """
    import re as _re
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cos

    IGNORE = {"water", "lukewarm water", "sunflower oil"}

    def _clean(t):
        return _re.sub(r"\s+", " ", str(t).lower()).strip()

    # FIX-C1: safe eval for malformed rows
    def _safe_eval(val):
        try:
            return ast.literal_eval(val)
        except Exception:
            return []

    df = df_raw.copy()
    if "Cleaned_Ingredients" in df.columns and not df.empty:
        sample = df["Cleaned_Ingredients"].iloc[0]
        if isinstance(sample, str):
            df["Cleaned_Ingredients"] = df["Cleaned_Ingredients"].apply(_safe_eval)

    df["Ingredient_Text"] = df["Cleaned_Ingredients"].apply(
        lambda x: " ".join(
            _clean(i) for i in x
            if isinstance(x, list) and _clean(i) not in IGNORE
        ) if isinstance(x, list) else ""
    )

    inv_filtered = [i for i in inventory_items if i not in IGNORE]
    inv_set      = set(inv_filtered)

    # FIX-BL-4 equivalent: guard empty inventory
    if not inv_filtered:
        df["Semantic_Score"] = 0.0
        df["Matched"] = df["Missed"] = [[] for _ in range(len(df))]
        df["Coverage"] = df["Expiry"] = df["Final_Score"] = 0.0
        return df

    inv_text  = " ".join(inv_filtered)
    all_texts = df["Ingredient_Text"].tolist() + [inv_text]

    if not any(t.strip() for t in all_texts):
        df["Semantic_Score"] = 0.0
        df["Matched"] = df["Missed"] = [[] for _ in range(len(df))]
        df["Coverage"] = df["Expiry"] = df["Final_Score"] = 0.0
        return df

    vec   = TfidfVectorizer()
    tfidf = vec.fit_transform(all_texts)

    # FIX-CMP-2: suppress UndefinedMetricWarning for zero-vector cosine
    with np.errstate(invalid="ignore"):
        sem_scores = _cos(tfidf[:-1], tfidf[-1:]).flatten()
    sem_scores = np.nan_to_num(sem_scores, nan=0.0)

    matched_list, missed_list, cov_list, exp_list = [], [], [], []
    for _, row in df.iterrows():
        if not isinstance(row["Cleaned_Ingredients"], list):
            matched_list.append([]); missed_list.append([])
            cov_list.append(0.0);    exp_list.append(0.0)
            continue
        ings     = [_clean(i) for i in row["Cleaned_Ingredients"] if _clean(i) not in IGNORE]
        matched  = list(set(i for i in ings if i in inv_set))
        missed   = list(set(i for i in ings if i not in inv_set))
        coverage = len(matched) / max(len(ings), 1)
        # FIX-B5 applied here too: clamp expiry score to [0, 1]
        exp_sc   = [max(0.0, min(1.0, (30 - inventory_dict.get(m, 30)) / 30)) for m in matched]
        expiry   = float(np.mean(exp_sc)) if exp_sc else 0.0
        matched_list.append(matched); missed_list.append(missed)
        cov_list.append(round(coverage, 3)); exp_list.append(round(expiry, 3))

    df["Matched"]        = matched_list
    df["Missed"]         = missed_list
    df["Coverage"]       = cov_list
    df["Expiry"]         = exp_list
    df["Semantic_Score"] = sem_scores
    # CALIBRATION FIX: aligned weights with proposed model for fair comparison
    df["Final_Score"]    = (
        0.25 * df["Semantic_Score"]
        + 0.35 * df["Coverage"]
        + 0.40 * df["Expiry"]
    )
    return df


def _exact_match_score_inline(
    df_raw: pd.DataFrame,
    inventory_items: list[str],
    inventory_dict: dict[str, int],
) -> pd.DataFrame:
    """
    Pure Exact Match scoring — no TF-IDF, no cosine similarity.
    An ingredient matches ONLY if the exact inventory item string appears
    in the recipe's ingredient list (case-insensitive, full-word).
    """
    import re as _re

    IGNORE = {"water", "lukewarm water", "sunflower oil"}

    def _clean(t):
        return _re.sub(r"\s+", " ", str(t).lower()).strip()

    def _safe_eval(val):
        try:
            return ast.literal_eval(val)
        except Exception:
            return []

    df = df_raw.copy()
    if "Cleaned_Ingredients" in df.columns and not df.empty:
        sample = df["Cleaned_Ingredients"].iloc[0]
        if isinstance(sample, str):
            df["Cleaned_Ingredients"] = df["Cleaned_Ingredients"].apply(_safe_eval)

    inv_filtered = [i for i in inventory_items if i not in IGNORE]
    inv_set = set(inv_filtered)

    if not inv_filtered:
        df["Semantic_Score"] = 0.0
        df["Matched"] = df["Missed"] = [[] for _ in range(len(df))]
        df["Coverage"] = df["Expiry"] = df["Final_Score"] = 0.0
        return df

    matched_list, missed_list, cov_list, exp_list, sem_list = [], [], [], [], []
    for _, row in df.iterrows():
        if not isinstance(row["Cleaned_Ingredients"], list):
            matched_list.append([]); missed_list.append([])
            cov_list.append(0.0); exp_list.append(0.0); sem_list.append(0.0)
            continue
        ings = [_clean(i) for i in row["Cleaned_Ingredients"] if _clean(i) not in IGNORE]
        # EXACT match only: ingredient must exactly equal an inventory item
        matched = list(set(i for i in ings if i in inv_set))
        missed = list(set(i for i in ings if i not in inv_set))
        coverage = len(matched) / max(len(ings), 1)
        # Semantic score = coverage itself (no TF-IDF vectorization)
        semantic = coverage
        exp_sc = [max(0.0, min(1.0, (30 - inventory_dict.get(m, 30)) / 30)) for m in matched]
        expiry = float(np.mean(exp_sc)) if exp_sc else 0.0
        matched_list.append(matched); missed_list.append(missed)
        cov_list.append(round(coverage, 3)); exp_list.append(round(expiry, 3))
        sem_list.append(round(semantic, 3))

    df["Matched"] = matched_list
    df["Missed"] = missed_list
    df["Coverage"] = cov_list
    df["Expiry"] = exp_list
    df["Semantic_Score"] = sem_list
    # Same weights as TF-IDF for fair comparison
    df["Final_Score"] = (
        0.25 * df["Semantic_Score"]
        + 0.35 * df["Coverage"]
        + 0.40 * df["Expiry"]
    )
    return df


# FIX-C3: "deltas" key always present.
def _empty_comparison() -> dict:
    empty_metrics = {
        "ndcg_8": 0.0, "f1_overall": 0.0, "k": 0,
        "precision_k": 0.0, "recall_k": 0.0, "f1_k": 0.0,
        "ap_k": 0.0, "hit_rate_k": 0.0, "mrr": 0.0, "ndcg_k": 0.0,
        "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "tn": 0, "fp": 0, "fn": 0, "tp": 0,
        "confusion_matrix": [[0, 0], [0, 0]],
        "total_recipes": 0, "total_relevant": 0, "positive_rate": 0.0,
        "warnings": ["No inventory found — cannot run comparison."],
        "catalog_coverage": 0.0, "weighted_expiry": 0.0,
        "expiry_utilisation": 0.0, "fwri": 0.0,
    }
    return {
        "baseline": empty_metrics,
        "proposed": empty_metrics,
        "exact":    empty_metrics,
        "deltas":   {},           # FIX-C3: always present
        "baseline_reach": "0.0%",
        "proposed_reach": "0.0%",
    }


# ---------------------------------------------------------------------------
# Standalone execution
# FIX-C4: None guard in delta_str formatter.
# FIX-CMP-6: print per-model warnings.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Model Comparison — Baseline (TF-IDF) vs Proposed (Semantic)")
    print("=" * 60)

    USERNAME = "default"
    data = run_full_comparison(USERNAME)

    headers = [
        ("Classification", ["precision", "recall", "f1", "accuracy"]),
        ("Ranking @50",    ["ndcg_8", "ndcg_k", "precision_k", "recall_k",
                            "f1_k", "ap_k", "mrr", "hit_rate_k"]),
        ("Sustainability", ["catalog_coverage", "fwri", "weighted_expiry"]),
    ]

    print(f"\n  {'Metric':<26} {'Baseline':>10} {'Proposed':>10} {'Delta':>10}")
    print(f"  {'─'*26} {'─'*10} {'─'*10} {'─'*10}")

    for section, keys in headers:
        print(f"\n  [{section}]")
        for key in keys:
            b = data["baseline"].get(key, "n/a")
            p = data["proposed"].get(key, "n/a")
            d = data["deltas"].get(key)
            # FIX-C4: None guard
            if d is not None:
                delta_str = f"{'+' if d >= 0 else ''}{d:.3f}"
            else:
                delta_str = "n/a"
            print(f"  {key:<26} {str(b):>10} {str(p):>10} {delta_str:>10}")

    print(f"\n  Baseline Active Reach : {data['baseline_reach']}")
    print(f"  Proposed Active Reach : {data['proposed_reach']}")

    # FIX-CMP-6: print per-model warnings
    for model in ("baseline", "proposed"):
        warnings = data[model].get("warnings", [])
        if warnings:
            print(f"\n  [{model.title()} warnings]")
            for w in warnings:
                print(f"    ⚠  {w}")
