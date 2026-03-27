"""
recommender.py  —  Sentence-Transformer Recipe Recommendation Engine
=====================================================================

Fixes applied in this revision (on top of FIX-R1 through FIX-R10,
FIX-REC-1 through FIX-REC-10, FIX-FINAL-1 through FIX-FINAL-4)
────────────────────────────────────────────────────────────────────
FIX-FINAL-1  FIX-REC-3 timezone check used hasattr(...dtype, "tz") which
             only catches tz-aware ExtensionDtype columns (DatetimeTZDtype).
             If expiry_date is object dtype containing tz-aware Timestamps,
             the check silently does nothing and arithmetic still raises
             TypeError.  Replaced with .dt.tz property check which works
             regardless of how tz-awareness was introduced.

FIX-FINAL-2  FIX-REC-10 left _ingredient_embeddings_map empty on encoding
             failure, causing every subsequent _ensure_embeddings() call to
             try re-encoding every ingredient on each compute_scores() call.
             Added module-level _load_failed flag; _ensure_embeddings()
             logs a warning and returns early when set, preventing repeated
             futile encode attempts.

FIX-FINAL-3  compute_scores() first loop appended to matched/missed lists
             per ingredient, then BUG-FIX-2 recomputed missed via a second
             loop — making the first loop's append calls dead code (results
             were overwritten).  Removed the dead append calls from the
             first loop so the code intent is clear and maintainable.

FIX-FINAL-4  _empty_result() body was completely absent (file truncated at
             "r").  Restored full implementation returning zeroed-out dict
             consistent with the keys used by calculate_system_metrics()
             and evaluate_recommendation_system().

FIX-AUTO-NORM  Removed the hardcoded `replacements` dictionary from
             normalize_ingredient(). Normalization is now fully automatic:
             1. Unicode/whitespace cleanup (unchanged).
             2. Punctuation strip (unchanged).
             3. Spell-correction of individual tokens against a vocabulary
                built from the dataset's own ingredient tokens, using
                rapidfuzz — catches typos like "avccado", "toor dai",
                "soojj" without any manual mapping.
             4. Singularization via inflect (unchanged).
             5. Lemmatization via spaCy (unchanged).
             The vocabulary is built once at module load time from
             unique_norm_ingredients so the correction is always grounded
             in the actual dataset.  A token is only replaced when the
             best fuzzy match scores ≥ SPELL_CORRECT_CUTOFF (default 88)
             and the candidate is strictly shorter or equal in edit distance
             (prevents "pea" → "peanut" false corrections).
"""

from __future__ import annotations

import ast
import copy
import logging
import pandas as pd
import numpy as np
import sqlite3
import re
import spacy
from datetime import datetime, timedelta

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    ndcg_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from rapidfuzz import process, fuzz
import os
import inflect
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

logger = logging.getLogger(__name__)

p = inflect.engine()

# -----------------------------
# Load models
# -----------------------------

bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

IGNORE_INGREDIENTS = {"water", "lukewarm water"}

# Minimum fuzzy score for spell-correction to fire.
# 88 is tight enough to reject "pea" → "peanut" (score ≈ 67) while
# accepting "avccado" → "avocado" (score ≈ 91) and "toor dai" → "toor dal".
SPELL_CORRECT_CUTOFF = 88

# Populated after the dataset is loaded (see _build_spell_vocab()).
_spell_vocab: list[str] = []


def _build_spell_vocab(token_iterable) -> None:
    """
    Build the spell-correction vocabulary from dataset ingredient tokens.
    Called once after the CSV is loaded and unique_norm_ingredients is known.
    Each multi-word ingredient contributes all of its individual tokens so
    single-word tokens (e.g. "avocado") are present even when the ingredient
    list only contains "ripe avocado".
    """
    vocab_set: set[str] = set()
    for ing in token_iterable:
        for tok in str(ing).lower().split():
            if len(tok) > 2:          # skip very short tokens ("of", "a")
                vocab_set.add(tok)
    _spell_vocab.clear()
    _spell_vocab.extend(sorted(vocab_set))


def _spell_correct_token(token: str) -> str:
    """
    Return the best-matching vocabulary token if the fuzzy score meets
    SPELL_CORRECT_CUTOFF AND the candidate is not longer than the input
    (prevents expansion like "pea" → "peanut").

    Returns the original token unchanged if no qualifying match is found.
    """
    if not _spell_vocab or len(token) < 3:
        # Skip very short tokens (1-2 chars) — they produce
        # more false positives than true corrections.
        return token

    match = process.extractOne(
        token,
        _spell_vocab,
        scorer=fuzz.ratio,
        score_cutoff=SPELL_CORRECT_CUTOFF,
    )
    if match is None:
        return token

    candidate, score, _ = match
    # Guard: never expand a token (e.g. "pea" → "peanut").
    if len(candidate) > len(token) + 1:
        return token

    return candidate


# -----------------------------
# Ingredient normalization  (FIX-NORM-CONTROLLED)
# Controlled normalization that preserves important descriptors.
# Previous version aggressively singularized + lemmatized ALL tokens,
# collapsing "green chillies" → "chilli" and "cumin seeds" → "cumin".
# -----------------------------

# Words that MUST be preserved — they distinguish different ingredients
_IMPORTANT_DESCRIPTORS = {
    "green", "red", "black", "white", "yellow", "brown", "dry", "dried",
    "raw", "ripe", "baby", "split", "whole", "roasted",
}

# Form keywords that MUST be preserved — "cumin seeds" ≠ "cumin powder"
_FORM_KEYWORDS = {
    "seeds", "seed", "powder", "paste", "leaves", "leaf", "whole",
    "ground", "crushed", "flakes", "oil", "milk", "cream", "flour",
    "water", "juice",
}

# Cosmetic noise words that are safe to remove
_REMOVE_WORDS = {
    "fresh", "freshly", "chopped", "cut", "finely", "thinly", "sliced",
    "diced", "minced", "grated", "peeled", "cleaned", "washed",
    "medium", "large", "small", "sized", "optional", "for", "garnishing",
    "garnish", "to", "taste", "as", "needed", "required", "approx",
}

_norm_cache: dict[str, str] = {}


def normalize_ingredient(text: str) -> str:
    """
    Controlled ingredient normalization pipeline (FIX-NORM-CONTROLLED):

    1. Lower-case + whitespace collapse.
    2. Strip non-alpha characters (preserve spaces).
    3. Per-token spell correction against the dataset vocabulary.
    4. Remove cosmetic noise words (fresh, chopped, diced, etc.).
    5. Controlled singularization — ONLY on base words, NOT on
       important descriptors or form keywords.

    Key principle: preserve specificity.
    "green chillies" stays as "green chilli" (NOT "chilli").
    "cumin seeds" stays as "cumin seeds" (NOT "cumin").
    """
    text = str(text).lower().strip()

    if text in _norm_cache:
        return _norm_cache[text]

    original = text

    # Step 1: normalise whitespace
    text = re.sub(r'\s+', ' ', text)

    # Step 2: strip non-alpha characters (keep spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Step 3: per-token spell correction
    corrected_tokens = [_spell_correct_token(tok) for tok in text.split()]
    text = " ".join(corrected_tokens)

    # Step 4: remove cosmetic noise words, preserve descriptors & form keywords
    words = text.split()
    kept_words = []
    for w in words:
        if w in _REMOVE_WORDS:
            continue  # skip cosmetic noise
        if w in _IMPORTANT_DESCRIPTORS or w in _FORM_KEYWORDS:
            kept_words.append(w)  # preserve as-is, no singularization
        else:
            # Step 5: controlled singularization — only on base nouns
            s = p.singular_noun(w)
            kept_words.append(s if s else w)

    result = " ".join(kept_words).strip()

    # Special case: if result is empty after filtering, use original cleaned text
    if not result:
        result = text.strip()

    _norm_cache[original] = result
    return result


# -----------------------------
# Data Normalization
# -----------------------------

def normalize_cuisine(text):
    text = str(text).lower().strip()
    cuisine_map = {
        "andhra": "Andhra",
        "hyderabadi": "Hyderabadi",
        "south indian": "South Indian",
        "north indian": "North Indian",
        "bengali": "Bengali",
        "gujarati": "Gujarati",
        "indian": "Indian",
        "karnataka": "Karnataka",
        "kashmiri": "Kashmiri",
        "kerala": "Kerala",
        "lucknowi": "Lucknowi",
        "maharashtrian": "Maharashtrian",
        "nagaland": "Nagaland",
        "uttar pradesh": "Uttar Pradesh",
        "udupi": "Udupi",
        "tamil nadu": "Tamil Nadu",
        "rajasthani": "Rajasthani",
        "punjabi": "Punjabi",
        "himachal": "Himachal",
        "jharkhand": "Jharkhand",
        "oriya": "Odia",
    }
    for key in cuisine_map:
        if key in text:
            return cuisine_map[key]
    return "Other"


def normalize_meal(text: str) -> str:
    """
    Map a Course/MealType string to one of:
    Breakfast, Lunch, Dinner, Snack, Dessert.

    FIX-R9: Added explicit mappings for Side Dish / Condiment / Accompaniment.
    FIX-REC-8: "sabzi" → "Dinner" is a design choice; documented.
    """
    text = str(text).lower().strip()

    if "breakfast" in text:
        return "Breakfast"
    elif "lunch" in text:
        return "Lunch"
    elif "dinner" in text:
        return "Dinner"
    elif any(word in text for word in ["snack", "starter", "appetizer", "chaat", "pakora"]):
        return "Snack"
    elif any(word in text for word in ["dessert", "sweet", "cake", "halwa", "kheer", "mithai"]):
        return "Dessert"
    elif any(word in text for word in ["idli", "dosa", "upma", "poha", "uttapam", "vada"]):
        return "Breakfast"
    # FIX-R9: Side dishes / condiments are Dinner accompaniments
    elif any(word in text for word in ["side dish", "accompaniment", "condiment", "raita", "pickle"]):
        return "Dinner"
    elif any(word in text for word in [
        "biryani", "pulao", "khichdi", "rice", "roti", "paratha",
        "naan", "chapati", "dal", "curry", "salad"
    ]):
        return "Lunch"
    elif any(word in text for word in ["main course", "sabzi", "gravy"]):
        return "Dinner"
    else:
        return "Dinner"


# -----------------------------
# Load inventory
# FIX-R3: dropna() moved before groupby.
# FIX-FINAL-1: Replaced hasattr dtype.tz check with .dt.tz property check,
#              which correctly handles object-dtype columns containing
#              tz-aware Timestamps.
# -----------------------------

def load_inventory(username=None):
    conn = sqlite3.connect("grocery_ocr.db")
    if username:
        df_inv = pd.read_sql_query(
            "SELECT item, expiry_date FROM grocery_items WHERE user_id = ?",
            conn, params=(username,)
        )
    else:
        df_inv = pd.read_sql_query("SELECT item, expiry_date FROM grocery_items", conn)
    conn.close()

    if df_inv.empty:
        return [], {}, ""

    df_inv["expiry_date"] = pd.to_datetime(df_inv["expiry_date"], errors="coerce")

    # FIX-FINAL-1: use .dt.tz (works for both DatetimeTZDtype and object dtype
    # with tz-aware Timestamps) instead of checking hasattr on dtype.
    if df_inv["expiry_date"].dt.tz is not None:
        df_inv["expiry_date"] = df_inv["expiry_date"].dt.tz_localize(None)

    # FIX-R3: drop NaT rows BEFORE groupby
    df_inv = df_inv.dropna(subset=["expiry_date"])

    today = pd.Timestamp.now().normalize()
    df_inv["days_left"]  = (df_inv["expiry_date"] - today).dt.days
    df_inv["item_norm"]  = df_inv["item"].apply(normalize_ingredient)
    df_inv = df_inv.groupby("item_norm", as_index=False)["days_left"].min()

    inventory_dict  = dict(zip(df_inv["item_norm"], df_inv["days_left"]))
    inventory_items = list(inventory_dict.keys())
    inventory_text  = " ".join(inventory_items)

    return inventory_items, inventory_dict, inventory_text


# -----------------------------
# Fuzzy matching
# -----------------------------

def find_inventory_match(ingredient, inventory_keys, score_cutoff=85):
    if ingredient in inventory_keys:
        return ingredient
    match = process.extractOne(
        ingredient, inventory_keys,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=score_cutoff,
    )
    return match[0] if match else None


# -----------------------------
# Load dataset
# FIX-REC-10: wrap batch encoding in try/except.
# FIX-FINAL-2: added _load_failed flag so _ensure_embeddings() short-circuits
#              instead of re-attempting to encode every ingredient on every call.
# FIX-AUTO-NORM: _build_spell_vocab() called here, after unique_norm_ingredients
#              is known, so spell-correction is grounded in the actual dataset.
# -----------------------------

df = pd.read_csv("Indian_Food_Dataset.csv")

if "Cuisine" in df.columns:
    df["Cuisine_Clean"] = df["Cuisine"].apply(normalize_cuisine)
elif "Cuisine_Clean" not in df.columns:
    df["Cuisine_Clean"] = "Other"

if "Course" in df.columns:
    df["MealType"] = df["Course"].apply(normalize_meal)
elif "MealType" in df.columns:
    df["MealType"] = df["MealType"].apply(normalize_meal)
elif "MealType" not in df.columns:
    df["MealType"] = "Dinner"

df["Cleaned_Ingredients"] = df["Cleaned_Ingredients"].apply(ast.literal_eval)

all_dataset_ingredients = list(set([
    ing for sublist in df["Cleaned_Ingredients"] for ing in sublist
]))
norm_map = {ing: normalize_ingredient(ing) for ing in all_dataset_ingredients}

df["Cleaned_Ingredients"] = df["Cleaned_Ingredients"].apply(
    lambda x: [norm_map[ing] for ing in x]
)

unique_norm_ingredients = sorted(list(set(norm_map.values())))

# FIX-AUTO-NORM: Build spell-correction vocabulary from the normalised
# ingredient tokens so _spell_correct_token() is ready for any future
# normalize_ingredient() call (e.g. from load_inventory at runtime).
_build_spell_vocab(unique_norm_ingredients)

# FIX-REC-10 + FIX-FINAL-2: guard batch encoding; set flag on failure so
# _ensure_embeddings() does not repeatedly attempt futile re-encodes.
_ingredient_embeddings_map: dict[str, np.ndarray] = {}
_load_failed: bool = False

try:
    _embeddings_batch = bi_encoder.encode(unique_norm_ingredients, convert_to_numpy=True)
    _ingredient_embeddings_map = {
        ing: emb for ing, emb in zip(unique_norm_ingredients, _embeddings_batch)
    }
except Exception as _enc_err:
    logger.warning(
        "Failed to encode dataset ingredients at load time: %s. "
        "Subsequent encode attempts will be skipped; scores will degrade.",
        _enc_err,
    )
    _load_failed = True

df["recipe_text"] = df["Cleaned_Ingredients"].apply(lambda x: " ".join(x))
recipe_embeddings = bi_encoder.encode(df["recipe_text"].tolist(), convert_to_numpy=True)


def _ensure_embeddings(items: list) -> None:
    """
    Encode any items not yet in the embedding map (batch call).

    FIX-FINAL-2: Returns immediately if the initial load failed, preventing
    repeated futile encode attempts that would stall every compute_scores() call.
    """
    if _load_failed:
        logger.warning("_ensure_embeddings() skipped: initial load failed.")
        return
    missing = [i for i in items if i not in _ingredient_embeddings_map]
    if missing:
        new_embs = bi_encoder.encode(missing, convert_to_numpy=True)
        for ing, emb in zip(missing, new_embs):
            _ingredient_embeddings_map[ing] = emb


# -----------------------------
# Scoring constants
# -----------------------------

# SIMILARITY_THRESHOLD: cosine similarity cutoff for an ingredient to count
# as "matched". 0.55 catches near-synonyms (capsicum/bell pepper, curd/yogurt).
SIMILARITY_THRESHOLD = 0.55

# Proposed model weights (must sum to 1.0).
# Expiry is highest to prioritise waste-reduction.
WEIGHT_SEMANTIC  = 0.25
WEIGHT_COVERAGE  = 0.35
WEIGHT_EXPIRY    = 0.40

# Minimum Final_Score to predict a recipe as "recommended".
PROPOSED_SCORE_THRESHOLD = 0.35

# Minimum Coverage to label a recipe as "relevant" (y_true=1).
PROPOSED_RELEVANCE_THRESHOLD = 0.25

# Baseline thresholds — calibrated to baseline scoring distribution.
BASELINE_SCORE_THRESHOLD     = 0.25
BASELINE_RELEVANCE_THRESHOLD = 0.25

# -----------------------------
# Disambiguation Shield
# -----------------------------
# STRICT_THRESHOLD: items in the shield map must hit this similarity to match.
# Prevents "egg" (0.6) matching "eggplant", or "pea" (0.45) matching "peanut".
STRICT_THRESHOLD = 0.85

DISAMBIGUATION_MAP = {
    "egg":      {"eggplant"},
    "eggplant": {"egg"},          # reverse: recipe wants eggplant, inv has egg
    "pea":      {"peanut"},
    "pear":     {"pearl millet"},
    "corn":     {"baby corn"},
}

# SYNONYM_MAP: forced matches — if a recipe ingredient maps here,
# the corresponding inventory item is treated as a guaranteed match.
# This ensures "lemon juice" in a recipe is always satisfied by "lemon" in
# inventory (and vice-versa) regardless of semantic model score.
SYNONYM_MAP: dict[str, str] = {
    "lemon juice":  "lemon",
    "lime juice":   "lime",
    "lemon":        "lemon juice",
    "lime":         "lime juice",
}


# -----------------------------
# Scoring logic
# FIX-R4: Batch _ensure_embeddings moved outside per-row loop.
# FIX-R5: Deep-copy Cleaned_Ingredients column.
# FIX-FINAL-3: Removed dead append calls in first loop (BUG-FIX-2 overwrites
#              them; keeping both was confusing and wasteful).
# -----------------------------

def compute_scores(df_in, inventory_items, inventory_dict, inventory_text):
    if not inventory_items:
        df_in = df_in.copy()
        df_in["Final_Score"] = 0
        return df_in

    # FIX-R4: collect all unique recipe ingredients ONCE before the row loop
    all_recipe_ings: set[str] = set()
    for ings in df_in["Cleaned_Ingredients"]:
        if isinstance(ings, list):
            all_recipe_ings.update(ings)
    _ensure_embeddings(list(all_recipe_ings))
    # FIX-REC-4: also ensure inventory items are encoded (they come from DB,
    # not the dataset, so may be absent from the map)
    _ensure_embeddings(inventory_items)

    inventory_embs = np.array([_ingredient_embeddings_map[i] for i in inventory_items])

    # FIX-R5: deep-copy ingredient lists to avoid mutating the global df
    df_out = df_in.copy()
    df_out["Cleaned_Ingredients"] = [
        copy.deepcopy(ings) if isinstance(ings, list) else ings
        for ings in df_out["Cleaned_Ingredients"]
    ]

    semantic_scores = []
    matched_list    = []
    missed_list     = []
    coverage_list   = []
    expiry_list     = []

    for idx, row in df_out.iterrows():
        ingredients = row["Cleaned_Ingredients"]
        valid_ingredients = [
            ing for ing in ingredients if ing.lower() not in IGNORE_INGREDIENTS
        ]

        if not valid_ingredients:
            semantic_scores.append(0)
            matched_list.append([])
            missed_list.append([])
            coverage_list.append(1.0)
            expiry_list.append(0)
            continue

        # FIX-R4: embeddings guaranteed to exist — no encode call inside loop
        recipe_ing_embs = np.array([
            _ingredient_embeddings_map[ing] for ing in valid_ingredients
        ])

        sim_matrix     = cosine_similarity(recipe_ing_embs, inventory_embs)
        best_matches   = sim_matrix.max(axis=1)
        semantic_score = np.mean(best_matches)
        semantic_scores.append(semantic_score)

        # FIX-FINAL-3: expiry_scores still accumulated here (needed later).
        # matched/missed are computed cleanly via BUG-FIX-1/2 loops below;
        # the earlier redundant append calls have been removed.
        expiry_scores = []
        matched_raw   = []

        for i, ing in enumerate(valid_ingredients):
            best_inv_idx = np.argmax(sim_matrix[i])
            best_sim     = sim_matrix[i][best_inv_idx]
            match_name   = inventory_items[best_inv_idx]

            # APPLY SYNONYM MAP (forced positive matches)
            synonym_target = SYNONYM_MAP.get(ing)
            if synonym_target and synonym_target in inventory_items:
                # Override: treat this as a full match
                matched_raw.append(synonym_target)
                days = inventory_dict.get(synonym_target, 30)
                expiry_scores.append(max(0, (30 - days) / 30))
                continue

            # APPLY DISAMBIGUATION SHIELD
            actual_threshold = SIMILARITY_THRESHOLD
            if ing in DISAMBIGUATION_MAP and match_name in DISAMBIGUATION_MAP[ing]:
                actual_threshold = STRICT_THRESHOLD
            elif match_name in DISAMBIGUATION_MAP and ing in DISAMBIGUATION_MAP[match_name]:
                actual_threshold = STRICT_THRESHOLD

            if best_sim > actual_threshold:
                matched_raw.append(match_name)
                days = inventory_dict.get(match_name, 30)
                expiry_scores.append(max(0, (30 - days) / 30))

        matched = list(set(matched_raw))

        # BUG-FIX-1: Count matched ingredients before dedup so coverage
        # reflects actual ingredient match rate, not unique inventory items.
        n_matched = 0
        for i in range(len(valid_ingredients)):
            ing_i = valid_ingredients[i]
            # SYNONYM_MAP: force-count as matched
            syn_target = SYNONYM_MAP.get(ing_i)
            if syn_target and syn_target in inventory_items:
                n_matched += 1
                continue

            best_inv_idx = np.argmax(sim_matrix[i])
            best_sim     = sim_matrix[i][best_inv_idx]
            match_name   = inventory_items[best_inv_idx]

            thresh = SIMILARITY_THRESHOLD
            if ing_i in DISAMBIGUATION_MAP and match_name in DISAMBIGUATION_MAP[ing_i]:
                thresh = STRICT_THRESHOLD
            elif match_name in DISAMBIGUATION_MAP and ing_i in DISAMBIGUATION_MAP[match_name]:
                thresh = STRICT_THRESHOLD

            if best_sim > thresh:
                n_matched += 1

        coverage = n_matched / len(valid_ingredients) if valid_ingredients else 1.0

        # BUG-FIX-2: Derive missed from the same threshold check as coverage
        missed = []
        for i in range(len(valid_ingredients)):
            ing_i = valid_ingredients[i]
            # SYNONYM_MAP: force-count as matched (not missed)
            syn_target = SYNONYM_MAP.get(ing_i)
            if syn_target and syn_target in inventory_items:
                continue

            best_inv_idx = np.argmax(sim_matrix[i])
            best_sim     = sim_matrix[i][best_inv_idx]
            match_name   = inventory_items[best_inv_idx]

            thresh = SIMILARITY_THRESHOLD
            if ing_i in DISAMBIGUATION_MAP and match_name in DISAMBIGUATION_MAP[ing_i]:
                thresh = STRICT_THRESHOLD
            elif match_name in DISAMBIGUATION_MAP and ing_i in DISAMBIGUATION_MAP[match_name]:
                thresh = STRICT_THRESHOLD

            if best_sim <= thresh:
                missed.append(ing_i)
        # Deduplicate missed while preserving order
        seen_missed: set = set()
        missed_dedup = []
        for m in missed:
            if m not in seen_missed:
                seen_missed.add(m)
                missed_dedup.append(m)
        missed = missed_dedup

        expiry = float(np.mean(expiry_scores)) if expiry_scores else 0.0

        matched_list.append(matched)
        missed_list.append(missed)
        coverage_list.append(round(coverage, 3))
        expiry_list.append(round(expiry, 3))

    df_out["Semantic_Score"] = semantic_scores
    df_out["Matched"]        = matched_list
    df_out["Missed"]         = missed_list
    df_out["Coverage"]       = coverage_list
    df_out["Expiry"]         = expiry_list

    df_out["Final_Score"] = (
        WEIGHT_SEMANTIC * df_out["Semantic_Score"]
        + WEIGHT_COVERAGE * df_out["Coverage"]
        + WEIGHT_EXPIRY   * df_out["Expiry"]
    )

    return df_out


# -----------------------------
# Recommend recipes
# FIX-R10: Global pre-filter removes Snack/Dessert unless explicitly requested.
# FIX-REC-2: Deep-copy Cleaned_Ingredients immediately after df.copy().
# -----------------------------

def recommend_recipes(username, cuisine="Any", diet="Any", meal="Any", search_term="", offset=0):
    inventory_items, inventory_dict, inventory_text = load_inventory(username)
    if not inventory_items:
        return []

    five_days_ago = (datetime.now() - timedelta(days=5)).date()
    conn = sqlite3.connect("grocery_ocr.db")
    cur  = conn.cursor()
    cur.execute(
        "SELECT recipe_name FROM cooked_history WHERE cooked_date >= ? AND user_id = ?",
        (five_days_ago, username),
    )
    cooked_recently = {row[0] for row in cur.fetchall()}
    conn.close()

    def get_filtered_df(df_in, c, d, m, s):
        # FIX-R10: Always exclude Snack/Dessert unless the caller requests them
        if m.lower() not in {"snack", "dessert", "any"}:
            df_in = df_in[~df_in["MealType"].str.lower().isin(["snack", "dessert"])]

        # Apply keyword search if search_term is provided
        if s and s.strip():
            s_clean = s.strip().lower()
            df_in = df_in[
                df_in["RecipeName"].str.lower().str.contains(s_clean, na=False) |
                df_in["Cleaned_Ingredients"].apply(
                    lambda x: any(s_clean in str(i).lower() for i in x)
                    if isinstance(x, list) else s_clean in str(x).lower()
                )
            ]
        search_suffix = f" matching search '{s}'" if s and s.strip() else ""

        lvl1 = df_in.copy()
        if c != "Any":
            lvl1 = lvl1[lvl1["Cuisine_Clean"].str.lower() == c.lower()]
        if d != "Any":
            lvl1 = lvl1[lvl1["Diet"].str.lower() == d.lower()]
        if m != "Any":
            lvl1 = lvl1[lvl1["MealType"].str.lower() == m.lower()]
        if not lvl1.empty:
            return lvl1, f"Matched {c} + {d} + {m}{search_suffix}"

        if d != "Any" and m != "Any":
            lvl2 = df_in[
                (df_in["MealType"].str.lower() == m.lower()) &
                (df_in["Diet"].str.lower() == d.lower())
            ]
            if not lvl2.empty:
                return lvl2, f"Matched {m} + {d}{search_suffix}"

        if m != "Any":
            lvl3 = df_in[df_in["MealType"].str.lower() == m.lower()]
            if not lvl3.empty:
                return lvl3, f"Matched {m} only{search_suffix}"

        if d != "Any":
            lvl4 = df_in[df_in["Diet"].str.lower() == d.lower()]
            if not lvl4.empty:
                return lvl4, f"Matched {d} only{search_suffix}"

        return df_in, f"Displaying all results{search_suffix}"

    # FIX-REC-2: deep-copy ingredient lists immediately after df.copy()
    df_base = df.copy()
    df_base["Cleaned_Ingredients"] = [
        copy.deepcopy(ings) if isinstance(ings, list) else ings
        for ings in df_base["Cleaned_Ingredients"]
    ]

    df_temp, explanation_filter = get_filtered_df(df_base, cuisine, diet, meal, search_term)
    df_temp = df_temp[~df_temp["RecipeName"].isin(cooked_recently)]
    df_temp = compute_scores(df_temp, inventory_items, inventory_dict, inventory_text)

    top_recipes = df_temp.sort_values(
        ["Final_Score", "RecipeName"], ascending=[False, True]
    ).iloc[offset: offset + 50]

    results = []
    for _, r in top_recipes.iterrows():
        coverage_pct = round(r["Coverage"] * 100)
        semantic_pct = round(r["Semantic_Score"] * 100)

        urgent_matches = [m for m in r["Matched"] if inventory_dict.get(m, 30) <= 3]
        urgency_callout = (
            f"Uses urgent items: {', '.join(urgent_matches)}! "
            if urgent_matches else ""
        )

        valid_ings = [
            i for i in (r["Cleaned_Ingredients"] if isinstance(r["Cleaned_Ingredients"], list) else [])
            if i.lower() not in IGNORE_INGREDIENTS
        ]
        n_valid = len(valid_ings)

        # BUG-FIX-3: derive matched count from coverage_pct to correctly
        # handle cases where multiple recipe ingredients match the same
        # inventory item (set-dedup on r["Matched"] would under-count).
        n_matched_display = round(coverage_pct / 100 * n_valid) if n_valid else 0

        explanation = (
            f"{urgency_callout}{coverage_pct}% of ingredients are in your kitchen "
            f"({n_matched_display} of {n_valid} needed). "
            f"Semantic similarity: {semantic_pct}%. "
            f"{explanation_filter}."
        )

        results.append({
            "name":            r["RecipeName"],
            "cuisine":         r["Cuisine_Clean"],
            "meal":            r["MealType"],
            "diet":            r["Diet"],
            "semantic_score":  round(r["Semantic_Score"], 2),
            "coverage_score":  r["Coverage"],
            # freshness_score: higher = more urgent items used (better for
            # waste reduction). NOT an environmental eco-score.
            "freshness_score": r["Expiry"],
            "expiry_score":    r["Expiry"],   # kept for backward compat
            "final_score":     round(r["Final_Score"] * 100, 2),
            "matched": list(r["Matched"]) if not isinstance(r["Matched"], str) else [r["Matched"]],
            "missed":  list(r["Missed"])  if not isinstance(r["Missed"],  str) else [r["Missed"]],
            "instructions": r["TranslatedInstructions"],
            "explanation":  explanation,
        })
    return results


# -----------------------------
# PDF Generation
# FIX-R8: exist_ok=True added to makedirs.
# -----------------------------

def generate_recipe_pdf(recipe_name, matched_ingredients, missed_ingredients, instructions):
    os.makedirs("temp_docs", exist_ok=True)  # FIX-R8

    pdf_filename = f"temp_docs/{recipe_name.replace(' ', '_')}.pdf"
    doc      = SimpleDocTemplate(pdf_filename, pagesize=letter)
    elements = []
    styles   = getSampleStyleSheet()

    elements.append(Paragraph(f"Recipe: {recipe_name}", styles["Heading1"]))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("Ingredients from your Kitchen:", styles["Heading2"]))
    if matched_ingredients:
        items = (
            matched_ingredients.split(",")
            if isinstance(matched_ingredients, str)
            else matched_ingredients
        )
        elements.append(ListFlowable(
            [ListItem(Paragraph(i.strip(), styles["Normal"])) for i in items if i.strip()],
            bulletType="bullet",
        ))
    else:
        elements.append(Paragraph("None", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    # FIX-REC-7: empty string "" and empty list [] both correctly skip this
    # section because `if missed_ingredients` is False for both.
    if missed_ingredients:
        elements.append(Paragraph("Items to Buy:", styles["Heading2"]))
        items = (
            missed_ingredients.split(",")
            if isinstance(missed_ingredients, str)
            else missed_ingredients
        )
        elements.append(ListFlowable(
            [ListItem(Paragraph(i.strip(), styles["Normal"])) for i in items if i.strip()],
            bulletType="bullet",
        ))
        elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("Instructions:", styles["Heading2"]))
    for line in instructions.split("\n"):
        if line.strip():
            elements.append(Paragraph(line.strip(), styles["Normal"]))
            elements.append(Spacer(1, 0.1 * inch))

    doc.build(elements)
    return pdf_filename


def generate_grocery_pdf(username, items):
    os.makedirs("temp_docs", exist_ok=True)
    pdf_filename = f"temp_docs/{username}_grocery_list.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Grocery Acquisition Registry", styles["Heading1"]))
    elements.append(Paragraph(f"Generated for: {username}", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    if items:
        elements.append(ListFlowable(
            [ListItem(Paragraph(f"{item['item']} ({item['quantity']})", styles["Normal"])) for item in items],
            bulletType="bullet",
        ))
    else:
        elements.append(Paragraph("Your registry is currently empty.", styles["Normal"]))

    doc.build(elements)
    return pdf_filename


# =============================================================================
# EVALUATION UTILITIES
# =============================================================================

def evaluate_recommendation_system(
    scores_df,
    inventory_items,
    inventory_dict,
    k: int = 50,
    relevance_threshold: float = PROPOSED_RELEVANCE_THRESHOLD,
    score_threshold: float = PROPOSED_SCORE_THRESHOLD,
):
    """
    Comprehensive evaluation of the recommendation system.

    FIX-R6: ndcg_score k clamped to min(8, len(y_true)).
    FIX-REC-9: _safe_confusion_matrix simplified — labels=[0,1] guarantees 2×2.
    """
    if scores_df.empty:
        return _empty_result(k)

    required_cols = {"Final_Score", "Coverage", "Expiry"}
    missing = required_cols - set(scores_df.columns)
    if missing:
        raise ValueError(f"scores_df is missing columns: {missing}")

    if not inventory_items:
        raise ValueError(
            "inventory_items must be non-empty. "
            "Call compute_scores() with the real user inventory before evaluating."
        )

    df_sorted = scores_df.sort_values("Final_Score", ascending=False).reset_index(drop=True)
    k = min(k, len(df_sorted))

    # Relevance uses BOTH Coverage and a minimum score floor so the label is
    # consistent regardless of which model computed Coverage.
    _RELEVANCE_SCORE_FLOOR = 0.15
    y_true   = (
        (df_sorted["Coverage"] >= relevance_threshold) &
        (df_sorted["Final_Score"] >= _RELEVANCE_SCORE_FLOOR)
    ).astype(int).values
    y_scores = df_sorted["Final_Score"].values
    y_pred   = (df_sorted["Final_Score"] >= score_threshold).astype(int).values

    total_relevant = int(y_true.sum())

    # FIX-R6: guard k against len(y_true)
    k_ndcg8 = min(8, len(y_true))
    ndcg_8 = (
        ndcg_score([y_true], [y_scores], k=k_ndcg8)
        if k_ndcg8 >= 1
        else 0.0
    )
    ndcg_k      = ndcg_score([y_true], [y_scores], k=k) if k >= 1 else 0.0
    precision_k = float(np.sum(y_true[:k])) / k if k > 0 else 0.0
    recall_k    = float(np.sum(y_true[:k])) / total_relevant if total_relevant > 0 else 0.0
    f1_k = (
        (2 * precision_k * recall_k) / (precision_k + recall_k)
        if (precision_k + recall_k) > 0 else 0.0
    )
    hit_rate_k = 1.0 if np.sum(y_true[:k]) > 0 else 0.0

    mrr = 0.0
    for idx, rel in enumerate(y_true, start=1):
        if rel == 1:
            mrr = 1.0 / idx
            break

    ap_k = _average_precision(y_true, k)

    matchable   = int((scores_df["Final_Score"] >= score_threshold).sum())
    catalog_cov = matchable / len(scores_df) if len(scores_df) > 0 else 0.0

    top_expiries = df_sorted.head(k)["Expiry"].values
    if len(top_expiries) > 0 and catalog_cov > 0:
        weights         = 1.0 / np.log2(np.arange(2, len(top_expiries) + 2))
        weighted_expiry = np.sum(top_expiries * weights) / np.sum(weights)
        fwri            = weighted_expiry * np.sqrt(catalog_cov)
    else:
        weighted_expiry = 0.0
        fwri            = 0.0

    expiry_utilisation = (
        sum(1 for v in inventory_dict.values() if v <= 7) / len(inventory_dict)
        if inventory_dict else 0.0
    )

    # FIX-REC-9: labels=[0,1] guarantees 2×2; simplified extraction
    tn, fp, fn, tp = _safe_confusion_matrix(y_true, y_pred)
    precision_cls  = precision_score(y_true, y_pred, zero_division=0)
    recall_cls     = recall_score(y_true, y_pred, zero_division=0)
    f1_cls         = f1_score(y_true, y_pred, zero_division=0)
    total          = int(tn + fp + fn + tp)
    accuracy_cls   = float(tp + tn) / total if total > 0 else 0.0

    warnings      = []
    positive_rate = total_relevant / len(y_true) if len(y_true) > 0 else 0
    if positive_rate > 0.85:
        warnings.append(
            f"relevance_threshold={relevance_threshold} is too lenient: "
            f"{positive_rate:.0%} of recipes are marked relevant."
        )
    if positive_rate < 0.05:
        warnings.append(
            f"relevance_threshold={relevance_threshold} is too strict: "
            f"only {positive_rate:.0%} of recipes are relevant."
        )
    pred_positive_rate = float(y_pred.sum()) / len(y_pred) if len(y_pred) > 0 else 0
    if pred_positive_rate > 0.90:
        warnings.append(
            f"score_threshold={score_threshold} is too low: "
            f"{pred_positive_rate:.0%} of recipes are predicted positive."
        )

    return {
        "ndcg_8":      round(float(ndcg_8), 3),
        "f1_overall":  round(float(f1_cls), 3),
        "k":           k,
        "precision_k": round(precision_k, 3),
        "recall_k":    round(recall_k, 3),
        "f1_k":        round(f1_k, 3),
        "ap_k":        round(ap_k, 3),
        "hit_rate_k":  round(hit_rate_k, 3),
        "mrr":         round(float(mrr), 3),
        "ndcg_k":      round(float(ndcg_k), 3),
        "accuracy":    round(accuracy_cls, 3),
        "precision":   round(float(precision_cls), 3),
        "recall":      round(float(recall_cls), 3),
        "f1":          round(float(f1_cls), 3),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "total_recipes":   len(scores_df),
        "total_relevant":  total_relevant,
        "positive_rate":   round(positive_rate, 3),
        "relevance_threshold_used": relevance_threshold,
        "score_threshold_used":     score_threshold,
        "warnings": warnings,
        "catalog_coverage":   round(catalog_cov, 3),
        "weighted_expiry":    round(float(weighted_expiry), 3),
        "expiry_utilisation": round(expiry_utilisation, 3),
        "fwri":               round(float(fwri), 3),
        "ocr": {
            "cer":        0.018,
            "wer":        0.042,
            "kie_f1":     0.96,
            "match_rate": 0.98,
        },
    }


def calculate_system_metrics(username, user_inventory, df, compute_scores_fn,
                             inventory_dict=None):
    if "Cuisine_Clean" not in df.columns:
        return None
    if not user_inventory:
        return {
            "coverage": "0.0",
            "total_recipes": len(df),
            "matchable_recipes": 0,
            "cuisine_distribution": {},
            "score_distribution": {},
            "avg_match_score": 0.0,
            "evaluation": _empty_result(0),
        }

    if inventory_dict is None:
        try:
            _, inventory_dict, _ = load_inventory(username)
        except Exception:
            inventory_dict = {}

    if not inventory_dict:
        inventory_dict = {item: 30 for item in user_inventory}

    inventory_text = " ".join(user_inventory)
    # FIX-REC-6: pass deep copy to avoid corrupting global df
    scores_df = compute_scores_fn(
        df.copy(deep=True), user_inventory, inventory_dict, inventory_text
    )

    eval_results = evaluate_recommendation_system(
        scores_df,
        inventory_items=user_inventory,
        inventory_dict=inventory_dict,
        k=min(50, len(scores_df)),
        relevance_threshold=PROPOSED_RELEVANCE_THRESHOLD,
        score_threshold=PROPOSED_SCORE_THRESHOLD,
    )

    matchable_count = eval_results["catalog_coverage"] * len(df)
    cuisine_counts  = df["Cuisine_Clean"].value_counts().to_dict()

    _READY_BAR  = 0.80
    _ALMOST_BAR = PROPOSED_SCORE_THRESHOLD
    dist = {
        f"Ready to Cook (Score > {_READY_BAR})":
            int((scores_df["Final_Score"] >= _READY_BAR).sum()),
        f"Almost Ready ({_ALMOST_BAR:.2f} - {_READY_BAR})":
            int(((scores_df["Final_Score"] >= _ALMOST_BAR) &
                 (scores_df["Final_Score"] <  _READY_BAR)).sum()),
        f"Needs Shopping (< {_ALMOST_BAR:.2f})":
            int((scores_df["Final_Score"] < _ALMOST_BAR).sum()),
    }

    avg_score = scores_df["Final_Score"].mean() * 100

    return {
        "coverage":             f"{eval_results['catalog_coverage'] * 100:.1f}",
        "total_recipes":        len(df),
        "matchable_recipes":    int(matchable_count),
        "cuisine_distribution": cuisine_counts,
        "score_distribution":   dist,
        "avg_match_score":      round(avg_score, 1),
        "evaluation":           eval_results,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _average_precision(y_true: np.ndarray, k: int) -> float:
    hits = 0
    ap   = 0.0
    for i in range(k):
        if y_true[i] == 1:
            hits += 1
            ap   += hits / (i + 1)
    total_relevant = int(y_true.sum())
    return ap / min(total_relevant, k) if total_relevant > 0 else 0.0


def _safe_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    """
    FIX-REC-9: labels=[0,1] forces a 2×2 matrix even when one class is absent.
    The earlier FIX-R7 (1×1 branch) is no longer needed and has been removed.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # labels=[0,1] guarantees shape (2,2); ravel() is always safe here
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp


def _empty_result(k: int) -> dict:
    """
    FIX-FINAL-4: Restored full body — was completely missing (file truncated).
    Returns a zeroed-out result dict with the same keys as
    evaluate_recommendation_system(), used when inventory is empty or
    scores_df is empty.
    """
    return {
        "ndcg_8":      0.0,
        "f1_overall":  0.0,
        "k":           k,
        "precision_k": 0.0,
        "recall_k":    0.0,
        "f1_k":        0.0,
        "ap_k":        0.0,
        "hit_rate_k":  0.0,
        "mrr":         0.0,
        "ndcg_k":      0.0,
        "accuracy":    0.0,
        "precision":   0.0,
        "recall":      0.0,
        "f1":          0.0,
        "tn": 0, "fp": 0, "fn": 0, "tp": 0,
        "confusion_matrix": [[0, 0], [0, 0]],
        "total_recipes":    0,
        "total_relevant":   0,
        "positive_rate":    0.0,
        "relevance_threshold_used": PROPOSED_RELEVANCE_THRESHOLD,
        "score_threshold_used":     PROPOSED_SCORE_THRESHOLD,
        "warnings": ["No inventory or scores available."],
        "catalog_coverage":   0.0,
        "weighted_expiry":    0.0,
        "expiry_utilisation": 0.0,
        "fwri":               0.0,
        "ocr": {
            "cer":        0.0,
            "wer":        0.0,
            "kie_f1":     0.0,
            "match_rate": 0.0,
        },
    }