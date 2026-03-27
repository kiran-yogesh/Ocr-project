"""
meal_plan.py  —  Simple Meal Plan Generator
============================================

Imported by app.py as:
    from meal_plan import generate_meal_plan

Called by app.py as:
    detailed_plan, grocery_list = generate_meal_plan(username, plan)

Where `plan` is the session dict:
    {
        "Breakfast": {"index": 0},
        "Lunch":     {"index": 0},
        "Dinner":    {"index": 0},
    }

Design decisions
----------------
- Only three MealType tags are used: Breakfast, Lunch, Dinner.
  Snack, Dessert, Side Dish, Appetizer are never returned.
- inventory load + compute_scores runs ONCE (not once per meal slot)
  by calling recommend_recipes() with meal="Any" and then partitioning
  the results — avoids 3× redundant embedding computation.
- cooked_recently is fetched once here; recommend_recipes() fetches it
  again internally, which is a redundant DB call but harmless given
  SQLite's speed.
- Index cycling is bounded by len(pool) so it never raises IndexError.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from recommender import recommend_recipes

# The only meal tags this planner will ever return.
PLAN_MEAL_TYPES = ("Breakfast", "Lunch", "Dinner")


def generate_meal_plan(
    username: str,
    current_plan: dict,
) -> tuple[dict, list[str]]:
    """
    Generate one meal per slot (Breakfast / Lunch / Dinner) for today.

    Parameters
    ----------
    username     : session["user"] — used for inventory + history lookup.
    current_plan : session["meal_plan"] dict, e.g.
                   {"Breakfast": {"index": 0}, "Lunch": {"index": 1}, ...}

    Returns
    -------
    detailed_plan : {"Breakfast": recipe_dict | None,
                     "Lunch":     recipe_dict | None,
                     "Dinner":    recipe_dict | None}
    grocery_list  : sorted list of missing ingredients across all slots.
    """

    # ------------------------------------------------------------------
    # 1. Fetch cooked-recently set ONCE (recommend_recipes also fetches
    #    this internally, but we need it here for pre-filtering).
    # ------------------------------------------------------------------
    five_days_ago = (datetime.now() - timedelta(days=5)).date()
    conn = sqlite3.connect("grocery_ocr.db")
    cur  = conn.cursor()
    cur.execute(
        "SELECT recipe_name FROM cooked_history "
        "WHERE cooked_date >= ? AND user_id = ?",
        (five_days_ago, username),
    )
    cooked_recently = {row[0] for row in cur.fetchall()}
    conn.close()

    # ------------------------------------------------------------------
    # 2. Fetch all scored recommendations in ONE call (meal="Any"),
    #    then partition by MealType.
    #    This avoids running load_inventory() + compute_scores() 3 times.
    # ------------------------------------------------------------------
    all_recs = recommend_recipes(username, cuisine="Any", diet="Any", meal="Any")

    # Partition into per-slot pools, applying:
    #   - strict MealType filter (only Breakfast / Lunch / Dinner)
    #   - cooked-recently exclusion
    pools: dict[str, list[dict]] = {slot: [] for slot in PLAN_MEAL_TYPES}
    for rec in all_recs:
        slot = rec.get("meal", "")
        if slot in pools and rec["name"] not in cooked_recently:
            pools[slot].append(rec)

    # ------------------------------------------------------------------
    # 3. Pick one recipe per slot using the session index.
    # ------------------------------------------------------------------
    detailed_plan: dict[str, dict | None] = {}
    grocery_set: set[str] = set()

    for meal_type in PLAN_MEAL_TYPES:
        pool = pools[meal_type]

        if not pool:
            detailed_plan[meal_type] = None
            continue

        # Wrap index within current pool length — safe even if pool shrinks.
        raw_idx = current_plan.get(meal_type, {}).get("index", 0)
        idx     = raw_idx % len(pool)
        recipe  = pool[idx].copy()

        # Guarantee the meal tag is exactly the requested slot name.
        recipe["meal"] = meal_type

        detailed_plan[meal_type] = recipe

        # Collect missing ingredients for the grocery list.
        missed = recipe.get("missed", [])
        if isinstance(missed, list):
            grocery_set.update(i.strip() for i in missed if i.strip())
        elif isinstance(missed, str):
            grocery_set.update(i.strip() for i in missed.split(",") if i.strip())

    return detailed_plan, sorted(grocery_set)