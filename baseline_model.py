
import re
import ast
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Standard ingredients to ignore in similarity calculation
IGNORE_LIST = {"water", "lukewarm water", "sunflower oil"}

def clean_text(text: str) -> str:
    """Standard text cleaning for keyword matching."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()

def run_baseline_scores(
    df_raw: pd.DataFrame,
    inventory_items: list[str],
    inventory_dict: dict[str, int],
) -> pd.DataFrame:
    """
    Score recipes using TF-IDF cosine similarity against the inventory.
    """
    df = df_raw.copy()
    
    # Safe eval for stringified ingredient lists
    def _safe_eval(val):
        if isinstance(val, list): return val
        try:
            return ast.literal_eval(val)
        except Exception:
            return []

    if "Cleaned_Ingredients" in df.columns:
        df["Cleaned_Ingredients"] = df["Cleaned_Ingredients"].apply(_safe_eval)

    # Prepare ingredient text for TF-IDF
    df["Ingredient_Text"] = df["Cleaned_Ingredients"].apply(
        lambda x: " ".join(clean_text(i) for i in x if clean_text(i) not in IGNORE_LIST)
        if isinstance(x, list) else ""
    )

    inv_filtered = [i for i in inventory_items if i not in IGNORE_LIST]
    if not inv_filtered:
        # Fallback if inventory is empty
        df["Semantic_Score"] = 0.0
        df["Matched"] = df["Missed"] = [[] for _ in range(len(df))]
        df["Final_Score"] = 0.0
        return df

    inv_text = " ".join(inv_filtered)
    all_texts = df["Ingredient_Text"].tolist() + [inv_text]

    # Vectorize and compute similarity
    vec = TfidfVectorizer()
    tfidf_matrix = vec.fit_transform(all_texts)
    
    # Suppress warnings for zero vectors (recipes with only ignored ingredients)
    with np.errstate(invalid="ignore"):
        similarities = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1:]).flatten()
    
    df["Semantic_Score"] = np.nan_to_num(similarities, nan=0.0)

    # Calculate boolean matching/expiry components
    inv_set = set(inv_filtered)
    matched_results, missed_results, cov_results, exp_results = [], [], [], []
    
    for _, row in df.iterrows():
        ings = [clean_text(i) for i in row["Cleaned_Ingredients"] if clean_text(i) not in IGNORE_LIST]
        matched = list(set(i for i in ings if i in inv_set))
        missed = list(set(i for i in ings if i not in inv_set))
        coverage = len(matched) / max(len(ings), 1)
        
        # Consistent expiry weighting [0, 1]
        exp_vals = [max(0.0, min(1.0, (30 - inventory_dict.get(m, 30)) / 30)) for m in matched]
        expiry = float(np.mean(exp_vals)) if exp_vals else 0.0
        
        matched_results.append(matched)
        missed_results.append(missed)
        cov_results.append(round(coverage, 3))
        exp_results.append(round(expiry, 3))

    df["Matched"] = matched_results
    df["Missed"] = missed_results
    df["Coverage"] = cov_results
    df["Expiry"]   = exp_results
    
    # Final score using same weights as proposed model for fair comparison
    df["Final_Score"] = (
        0.25 * df["Semantic_Score"] + 
        0.35 * df["Coverage"] + 
        0.40 * df["Expiry"]
    )
    
    return df
