from __future__ import annotations

import itertools
import os
import re
import threading
import time
from datetime import datetime, timedelta

import pandas as pd
from flask import Flask, render_template, request, redirect, session
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from paddleocr import PaddleOCR

from recommender import recommend_recipes, generate_recipe_pdf
from meal_plan import generate_meal_plan

from ocr_utils import (
    extract_grocery_items,
    extract_grocery_items_low_conf,
    clean_item_name,
    normalize_quantity,
    is_qty as is_quantity,
    preprocess_image,
    classify_image_for_grocery,
    detect_image_type,
    extract_product_name_from_image,
)

# Backward-compatible alias
process_grocery_list = extract_grocery_items

from shelf_life import calculate_expiry as _sl_calculate_expiry, save_shelf_life, predict_shelf_life

app = Flask(__name__)
app.secret_key = "kitchen_secret"

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------
# EMAIL CONFIGURATION
# ---------------------------
# IMPORTANT: Provide valid SMTP credentials below to enable notifications.
# For Gmail: Use an "App Password" (https://myaccount.google.com/apppasswords)
SMTP_SERVER   = "smtp.gmail.com"
SMTP_PORT     = 587
SMTP_USER     = "chefaimanager@gmail.com"
SMTP_PASSWORD = "joyqvqcyrnrgisbm"
SENDER_EMAIL  = "chefaimanager@gmail.com"
MAIL_ENABLED  = True


def allowed_file(filename: str) -> bool:
    # FIX-APP-9: strip whitespace from extension
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower().strip()
    return ext in ALLOWED_EXTENSIONS


# ---------------------------
# SHELF LIFE SOURCE
# (Removed old shelf life source)
# ---------------------------


# ---------------------------
# OCR ENGINE
# ---------------------------
ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


# ---------------------------
# DATABASE INIT
# ---------------------------
def init_db():
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        email TEXT UNIQUE,
        password TEXT,
        mail_sent_count INTEGER DEFAULT 0
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS grocery_items(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        serial_no INTEGER,
        item TEXT,
        quantity TEXT,
        purchase_date DATE,
        expiry_date DATE,
        days_left INTEGER,
        user_id TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS inventory(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item TEXT,
        quantity TEXT,
        purchase_date DATE,
        expiry_date DATE,
        user_id TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS grocery_list(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item TEXT,
        quantity TEXT,
        added_date DATE,
        user_id TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS cooked_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recipe_name TEXT NOT NULL,
        cooked_date DATE NOT NULL,
        user_id TEXT
    )
    """)

    for table in ["grocery_items", "inventory", "grocery_list"]:
        cur.execute(f"PRAGMA table_info({table})")
        columns = [column[1] for column in cur.fetchall()]
        if "user_id" not in columns:
            print(f"Adding user_id column to {table}...")
            cur.execute(f"ALTER TABLE {table} ADD COLUMN user_id TEXT")
            cur.execute(f"UPDATE {table} SET user_id = 'default' WHERE user_id IS NULL")

    # Add mail_sent_count to users table if missing
    cur.execute("PRAGMA table_info(users)")
    user_columns = [column[1] for column in cur.fetchall()]
    if "mail_sent_count" not in user_columns:
        print("Adding mail_sent_count column to users...")
        cur.execute("ALTER TABLE users ADD COLUMN mail_sent_count INTEGER DEFAULT 0")
        cur.execute("UPDATE users SET mail_sent_count = 0 WHERE mail_sent_count IS NULL")

    cur.execute("SELECT COUNT(*) FROM grocery_items")
    if cur.fetchone()[0] == 0:
        cur.execute("SELECT COUNT(*) FROM inventory")
        if cur.fetchone()[0] > 0:
            cur.execute("SELECT item, quantity, purchase_date, expiry_date, user_id FROM inventory")
            legacy_items = cur.fetchall()
            sno = 1
            for item, qty, p_date, e_date, u_id in legacy_items:
                try:
                    p_date_dt = datetime.strptime(p_date, '%Y-%m-%d').date() if p_date else datetime.now().date()
                    e_date_dt = datetime.strptime(e_date, '%Y-%m-%d').date() if e_date else (p_date_dt + timedelta(days=10))
                    days_left = (e_date_dt - datetime.now().date()).days
                except Exception:
                    days_left = 10
                cur.execute("""
                INSERT INTO grocery_items (serial_no, item, quantity, purchase_date, expiry_date, days_left, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (sno, item, qty, p_date, e_date, days_left, u_id))
                sno += 1

    conn.commit()
    conn.close()


init_db()


# ---------------------------
# AUTO-DELETE EXPIRED ITEMS
# FIX-APP-6
# ---------------------------
def auto_cleanup_expired():
    while True:
        try:
            conn = sqlite3.connect("grocery_ocr.db")
            cur = conn.cursor()
            cur.execute("DELETE FROM grocery_items WHERE expiry_date < date('now')")
            deleted = cur.rowcount
            conn.commit()
            conn.close()
            if deleted > 0:
                print(f"[Auto-Cleanup] Removed {deleted} expired item{'s' if deleted != 1 else ''} from inventory.")
        except Exception as e:
            print(f"[Auto-Cleanup] Error: {e}")
        time.sleep(86400)


cleanup_thread = threading.Thread(target=auto_cleanup_expired, daemon=True)
cleanup_thread.start()


# ---------------------------
# EMAIL NOTIFICATION ENGINE
# ---------------------------
def send_expiry_notification(user_email, username, expiring_items):
    """Sends an email summary of expiring items using SMTP."""
    if not MAIL_ENABLED:
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Chef AI: Your Kitchen Inventory Expiry Alert"
        msg["From"]    = f"Chef AI Notification <{SENDER_EMAIL}>"
        msg["To"]      = user_email

        # Build email body
        item_rows = ""
        for item, days in expiring_items:
            status = "Expiring Today" if days == 0 else f"Expiring in {days} days"
            item_rows += f"<li><b>{item}</b>: {status}</li>"

        html = f"""
        <html>
          <body style="font-family: sans-serif; color: #334155;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #e2e8f0; border-radius: 12px;">
              <h2 style="color: #10b981;">Kitchen Expiry Alert</h2>
              <p>Hi {username},</p>
              <p>The following items in your inventory are expiring soon. Try to use them in a recipe today!</p>
              <ul style="line-height: 1.6;">
                {item_rows}
              </ul>
              <hr style="border: 0; border-top: 1px solid #e2e8f0; margin: 30px 0;">
              <p style="font-size: 0.8rem; color: #94a3b8;">This is an automated message from your Chef AI Kitchen System.</p>
            </div>
          </body>
        </html>
        """
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SENDER_EMAIL, user_email, msg.as_string())
        
        # Increment mail sent count in DB
        try:
            conn = sqlite3.connect("grocery_ocr.db")
            cur = conn.cursor()
            cur.execute("UPDATE users SET mail_sent_count = mail_sent_count + 1 WHERE email = ?", (user_email,))
            conn.commit()
            conn.close()
        except Exception as db_e:
            print(f"[Email Engine] Error updating mail count for {user_email}: {db_e}")

        return True
    except Exception as e:
        print(f"[Email Engine] Error sending to {user_email}: {e}")
        return False


def daily_email_notifications():
    """Background thread that sends daily expiry summaries to all users."""
    while True:
        if MAIL_ENABLED:
            try:
                print("[Email Engine] Checking for expiring items...")
                conn = sqlite3.connect("grocery_ocr.db")
                cur = conn.cursor()
                
                # Find users who have items expiring in 0-3 days
                cur.execute("SELECT id, username, email FROM users")
                users = cur.fetchall()
                
                for _, username, email in users:
                    cur.execute("""
                        SELECT item, 
                               CAST(julianday(expiry_date) - julianday('now') AS INTEGER) AS days_remaining
                        FROM grocery_items
                        WHERE julianday(expiry_date) - julianday('now') <= 3
                          AND julianday(expiry_date) - julianday('now') >= 0
                          AND user_id = ?
                        ORDER BY days_remaining ASC
                    """, (email,))
                    
                    expiring_items = cur.fetchall()
                    if expiring_items:
                        print(f"[Email Engine] Sending summary to {username} ({email})...")
                        send_expiry_notification(email, username, expiring_items)
                
                conn.close()
            except Exception as e:
                print(f"[Email Engine] Global Error: {e}")
        
        # Sleep for 24 hours
        time.sleep(86400)


email_thread = threading.Thread(target=daily_email_notifications, daemon=True)
email_thread.start()


# ---------------------------
# QUANTITY PARSING UTILS
# ---------------------------
def parse_qty(q):
    match = re.search(r"(\d+\.?\d*)", str(q))
    return float(match.group(1)) if match else 0.0


def get_unit(q):
    unit_patterns = [
        "kg", "g", "gm", "gram", "grams", "ltr", "liter", "liters", "lit", "ml",
        "pcs", "piece", "pieces", "packet", "packets", "pack", "packs", "box", "boxes", "bunch", "bunches", "jar", "jars",
        "units", "unit", "spoon", "spoons",
        "loaf", "loaves", "bottle", "bottles", "can", "cans", "tin", "tins", "sachet", "sachets", "tray", "trays", "back"
    ]
    q_lower = str(q).lower()
    for u in unit_patterns:
        if re.search(r'\b' + u + r'\b', q_lower):
            return u
    return "unit"


def convert_to_base(val, unit, context_unit=None):
    unit = unit.lower()
    if unit in ["kg", "kgs"]:
        return val * 1000, "g"
    if unit in ["gm", "gram", "grams", "g"]:
        return val, "g"
    if unit in ["ltr", "liter", "liters", "l", "lit"]:
        return val * 1000, "ml"
    if unit == "ml":
        return val, "ml"
    if unit in ["spoon", "spoons"]:
        if context_unit == "g":
            return val * 2, "g"
        if context_unit == "ml":
            return val * 2, "ml"
        return val * 2, "g"
    
    # Map plurals to singulars for unified UI grouping
    plural_map = {
        "loaves": "loaf",
        "bottles": "bottle",
        "cans": "can",
        "tins": "tin",
        "packets": "packet",
        "packs": "pack",
        "back": "pack",
        "boxes": "box",
        "pieces": "pc",
        "pcs": "pc",
        "piece": "pc",
        "jars": "jar",
        "units": "unit",
    }
    unit = plural_map.get(unit, unit)

    return val, unit


# ---------------------------------------------------------------------------
# EXPIRY CALCULATION  (replaces the old calculate_expiry + _scraper_shelf_life)
# ---------------------------------------------------------------------------

def calculate_expiry(item_name: str):
    """
    Returns (purchase_date_str, expiry_date_str, days_int) when item is in
    the shelf-life CSV, or None when it is unknown.

    Callers that receive None must redirect to /ask_expiry so the user can
    supply the expiry date.  The answer is saved back to the CSV permanently.
    """
    return _sl_calculate_expiry(item_name)


# ---------------------------------------------------------------------------
# ASK-EXPIRY POPUP ROUTES
# ---------------------------------------------------------------------------

@app.route("/ask_expiry", methods=["GET"])
def ask_expiry():
    """
    Show a modal/form asking the user for shelf-life details.
    Query params passed from the redirecting route are echoed back as hidden
    fields so /save_expiry knows what to insert after saving.

    Expected query params:
        item        : canonical item name
        qty         : quantity string
        source      : 'upload' | 'manual' | 'packet'
        expiry_date : (optional) already-known date from packet scan
    """
    if "user" not in session:
        return redirect("/")

    item        = request.args.get("item", "")
    qty         = request.args.get("qty", "1 pc")
    source      = request.args.get("source", "manual")
    expiry_date = request.args.get("expiry_date", "")   # from packet scan

    return render_template(
        "ask_expiry.html",
        item=item,
        qty=qty,
        source=source,
        expiry_date=expiry_date,
    )


@app.route("/save_expiry", methods=["POST"])
def save_expiry():
    """
    1. Reads the user-supplied shelf-life days (fridge / pantry) OR a direct
       expiry date.
    2. Persists the shelf-life data to shelf_life_data.csv so this item is
       never asked again.
    3. Inserts the grocery item into the DB.
    4. Redirects to /inventory.
    """
    if "user" not in session:
        return redirect("/")

    item        = request.form.get("item", "").strip()
    qty         = request.form.get("qty", "1 pc").strip()
    source      = request.form.get("source", "manual")

    fridge_days_str = request.form.get("fridge_days", "").strip()
    pantry_days_str = request.form.get("pantry_days", "").strip()
    expiry_date_str = request.form.get("expiry_date", "").strip()

    today_str = str(datetime.now().date())

    if expiry_date_str:
        try:
            expiry_dt = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
            days_left = (expiry_dt - datetime.now().date()).days
        except ValueError:
            days_left = 10
            expiry_date_str = str(datetime.now().date() + timedelta(days=10))
        purchase_str = today_str
    else:
        try:
            fridge_days = int(fridge_days_str) if fridge_days_str else None
        except ValueError:
            fridge_days = None
        try:
            pantry_days = int(pantry_days_str) if pantry_days_str else None
        except ValueError:
            pantry_days = None

        if item and (fridge_days or pantry_days):
            save_shelf_life(item, fridge_days, pantry_days)

        f = fridge_days if fridge_days and fridge_days > 0 else None
        p = pantry_days if pantry_days and pantry_days > 0 else None
        if f and p:
            days_left = round(0.7 * f + 0.3 * p)
        elif f:
            days_left = f
        elif p:
            days_left = p
        else:
            days_left = 10

        purchase_str    = today_str
        expiry_date_str = str(datetime.now().date() + timedelta(days=days_left))

    if not item:
        return redirect("/inventory")

    username = session["user"]
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    cur.execute("SELECT MAX(serial_no) FROM grocery_items WHERE user_id = ?", (username,))
    res = cur.fetchone()[0]
    sno = (res if res else 0) + 1
    cur.execute("""
        INSERT INTO grocery_items(serial_no, item, quantity, purchase_date, expiry_date, days_left, user_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (sno, item, qty, purchase_str, expiry_date_str, days_left, username))
    conn.commit()
    conn.close()
    return redirect("/inventory")


@app.route("/api/predict_shelf_life")
def api_predict_shelf_life():
    if "user" not in session:
        return {"error": "Unauthorized"}, 401
    item = request.args.get("item", "").strip()
    if not item:
        return {"error": "No item provided"}, 400
    
    prediction = predict_shelf_life(item)
    return prediction




# ---------------------------
# LOGIN
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        login_id = request.form["login_id"].strip()
        password = request.form["password"]
        conn = sqlite3.connect("grocery_ocr.db")
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM users WHERE (email=? OR username=?) AND password=?",
            (login_id, login_id, password)
        )
        user = cur.fetchone()
        conn.close()
        if user:
            session["user"] = user[2]
            session["username"] = user[1]
            return redirect("/home")
        else:
            return render_template("login.html", error="Invalid credentials")
    success = request.args.get("success")
    return render_template("login.html", success=success)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# ---------------------------
# FORGOT PASSWORD
# ---------------------------
@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")
        if not email or not new_password:
            return render_template("forgot_password.html", error="All fields are required.")
        if new_password != confirm_password:
            return render_template("forgot_password.html", error="Passwords do not match.")
        conn = sqlite3.connect("grocery_ocr.db")
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE email=?", (email,))
        user = cur.fetchone()
        if not user:
            conn.close()
            return render_template("forgot_password.html", error="Email not found.")
        cur.execute("UPDATE users SET password=? WHERE email=?", (new_password, email))
        conn.commit()
        conn.close()
        return render_template("forgot_password.html", success="Password reset successful!")
    return render_template("forgot_password.html")


# ---------------------------
# REGISTER
# ---------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        if not username or not email or not password:
            return render_template("register.html", error="All fields are required")
        allowed_domains = ["gmail.com", "yahoo.com", "outlook.com"]
        if not any(email.lower().endswith("@" + domain) for domain in allowed_domains):
            return render_template("register.html",
                                   error="Only Gmail, Yahoo, or Outlook addresses are allowed")
        conn = sqlite3.connect("grocery_ocr.db")
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE email=?", (email,))
        if cur.fetchone():
            conn.close()
            return render_template("register.html", error="Email already registered")
        
        try:
            cur.execute(
                "INSERT INTO users(username,email,password) VALUES(?,?,?)",
                (username, email, password)
            )
            conn.commit()
            conn.close()
            return redirect("/?success=Registration+successful.+Please+login.")
        except Exception:
            conn.close()
            return render_template("register.html", error="Registration failed. Please try again.")
    return render_template("register.html")


# ---------------------------
# HOME
# ---------------------------
@app.route("/home")
def home():
    if "user" not in session:
        return redirect("/")
    plan = session.get("meal_plan", {})
    return render_template("home.html", plan=plan)


@app.route("/profile")
def profile():
    if "user" not in session:
        return redirect("/")
    
    username = session.get("username")
    email = session.get("user")
    
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    
    # 1. Total Inventory Items
    cur.execute("SELECT COUNT(*) FROM grocery_items WHERE user_id = ?", (email,))
    total_items = cur.fetchone()[0]
    
    # 2. Items Expiring Soon (3 days)
    cur.execute("""
        SELECT COUNT(*) FROM grocery_items 
        WHERE user_id = ? 
        AND julianday(expiry_date) - julianday('now') <= 3 
        AND julianday(expiry_date) - julianday('now') >= 0
    """, (email,))
    expiring_soon = cur.fetchone()[0]
    
    # 3. Total Recipes Cooked
    cur.execute("SELECT COUNT(*) FROM cooked_history WHERE user_id = ?", (email,))
    total_cooked = cur.fetchone()[0]
    
    # 4. Grocery List Items
    cur.execute("SELECT COUNT(*) FROM grocery_list WHERE user_id = ?", (email,))
    grocery_count = cur.fetchone()[0]
    
    # 5. Recent Activity (Latest 5 cooked recipes)
    cur.execute("""
        SELECT recipe_name, cooked_date 
        FROM cooked_history 
        WHERE user_id = ? 
        ORDER BY cooked_date DESC LIMIT 5
    """, (email,))
    recent_activity = cur.fetchall()

    
    conn.close()
    
    stats = {
        "total_items": total_items,
        "expiring_soon": expiring_soon,
        "total_cooked": total_cooked,
        "grocery_count": grocery_count
    }
    
    return render_template(
        "profile.html",
        username=username,
        email=email,
        stats=stats,
        recent_activity=recent_activity
    )


# ---------------------------
# MEAL PLANNER
# ---------------------------
@app.route("/meal_planner")
def meal_planner():
    if "user" not in session:
        return redirect("/")
    if "meal_plan" not in session:
        session["meal_plan"] = {
            "Breakfast": {"index": 0},
            "Lunch":     {"index": 0},
            "Dinner":    {"index": 0},
        }
        session.modified = True
    plan = session["meal_plan"]
    username = session["user"]
    
    detailed_plan, grocery_list = generate_meal_plan(username, plan)
    
    return render_template("meal_planner.html",
                           plan=detailed_plan, grocery=grocery_list)


@app.route("/meal_planner/refresh/<meal_type>", methods=["GET", "POST"])
def refresh_meal(meal_type):
    if "user" not in session:
        return redirect("/")
    if "meal_plan" not in session:
        session["meal_plan"] = {
            "Breakfast": {"index": 0},
            "Lunch":     {"index": 0},
            "Dinner":    {"index": 0},
        }
        session.modified = True
    plan = session["meal_plan"]
    if meal_type in plan:
        plan[meal_type]["index"] = plan[meal_type].get("index", 0) + 1
        session.modified = True
    return redirect("/meal_planner")


@app.route("/meal_planner/cooked/<path:recipe_name>", methods=["GET", "POST"])
def mark_as_cooked(recipe_name):
    if "user" not in session:
        return redirect("/")
    from urllib.parse import quote
    return redirect(f"/cooked_update/{quote(recipe_name)}")


@app.route("/cooked_update/<path:recipe_name>")
def cooked_update(recipe_name):
    if "user" not in session:
        return redirect("/")
    from recommender import df
    recipe_row = df[df["RecipeName"] == recipe_name]
    if recipe_row.empty:
        return "Recipe not found", 404
    ingredients = recipe_row.iloc[0]["Cleaned_Ingredients"]
    username = session["user"]
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    cur.execute("SELECT item, quantity FROM grocery_items WHERE user_id = ?", (username,))
    inventory = cur.fetchall()
    conn.close()
    matched_ingredients = []
    for ing in ingredients:
        ing_clean = clean_item_name(ing) or ing.lower().strip()
        if not ing_clean:
            matched_ingredients.append({"name": ing, "matches": []})
            continue
        matches = []
        for item, qty in inventory:
            item_clean = clean_item_name(item) or item.lower().strip()
            if not item_clean:
                continue
            # FIX-APP-4: guard against empty strings before building regex
            pattern     = r'\b' + re.escape(ing_clean)  + r'(s|es)?\b'
            rev_pattern = r'\b' + re.escape(item_clean) + r'(s|es)?\b'
            if (re.search(pattern, item_clean, re.IGNORECASE) or
                    re.search(rev_pattern, ing_clean, re.IGNORECASE)):
                matches.append({"item": item, "quantity": qty})
        matched_ingredients.append({"name": ing, "matches": matches})
    return render_template("cooked_update.html",
                           recipe_name=recipe_name, ingredients=matched_ingredients)


@app.route("/process_cooked_update", methods=["POST"])
def process_cooked_update():
    if "user" not in session:
        return redirect("/")
    recipe_name = request.form.get("recipe_name")
    items_to_update = request.form.getlist("items[]")
    quantities = request.form.getlist("quantities[]")
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    username = session["user"]
    # FIX-APP-5: zip_longest prevents silent truncation on mismatched list lengths
    for item, used_qty_str in itertools.zip_longest(items_to_update, quantities, fillvalue=""):
        if not used_qty_str or not item:
            continue
        cur.execute(
            "SELECT quantity FROM grocery_items WHERE item = ? AND user_id = ?",
            (item, username)
        )
        row = cur.fetchone()
        if row:
            current_qty_str = row[0]
            current_val  = parse_qty(current_qty_str)
            used_val     = parse_qty(used_qty_str)
            current_unit = get_unit(current_qty_str)
            used_unit    = get_unit(used_qty_str)
            base_current_val, base_current_unit = convert_to_base(current_val, current_unit)
            base_used_val, base_used_unit = convert_to_base(
                used_val, used_unit, context_unit=base_current_unit
            )
            if base_current_unit != base_used_unit:
                continue
            new_val_base = base_current_val - base_used_val
            if new_val_base <= 0:
                cur.execute(
                    "DELETE FROM grocery_items WHERE item = ? AND user_id = ?",
                    (item, username)
                )
            else:
                formatted_val = (
                    int(new_val_base)
                    if new_val_base == int(new_val_base)
                    else round(new_val_base, 2)
                )
                new_qty_str = f"{formatted_val} {base_current_unit}"
                cur.execute(
                    "UPDATE grocery_items SET quantity = ? WHERE item = ? AND user_id = ?",
                    (new_qty_str, item, username)
                )
    today = datetime.now().date()
    cur.execute(
        "INSERT INTO cooked_history (recipe_name, cooked_date, user_id) VALUES (?, ?, ?)",
        (recipe_name, today, username)
    )
    conn.commit()
    conn.close()
    return redirect("/inventory")


# ---------------------------
# METRICS
# ---------------------------
@app.route("/metrics")
def metrics():
    if "user" not in session:
        return redirect("/")
    from recommender import calculate_system_metrics, df, compute_scores
    username = session["user"]
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    cur.execute("SELECT item FROM grocery_items WHERE user_id = ?", (username,))
    inventory_items = [row[0] for row in cur.fetchall()]
    conn.close()
    stats = calculate_system_metrics(username, inventory_items, df, compute_scores)
    return render_template("metrics.html", stats=stats)


@app.route("/research_metrics")
def research_metrics():
    if "user" not in session:
        return redirect("/")
    from recommender import compute_scores, evaluate_recommendation_system, df, load_inventory
    username = session["user"]
    inventory_items, inventory_dict, inventory_text = load_inventory(username)
    if not inventory_items:
        return render_template("research_metrics.html", metrics=None)
    scores_df = compute_scores(df.copy(), inventory_items, inventory_dict, inventory_text)
    metrics = evaluate_recommendation_system(
        scores_df,
        inventory_items=inventory_items,
        inventory_dict=inventory_dict,
    )
    return render_template("research_metrics.html", metrics=metrics)


# ---------------------------
# MODEL COMPARISON
# FIX-APP-11: Pass all 3 models to template (exact, tfidf, proposed).
# ---------------------------
@app.route("/model_comparison")
def model_comparison():
    if "user" not in session:
        return redirect("/")

    from compare_models import run_full_comparison

    # Run the full comparison logic (Baseline TF-IDF vs Proposed BERT)
    username = session.get("user")
    data = run_full_comparison(username)

    # If no data or empty results (e.g., no inventory), return empty state
    if not data or data["proposed"].get("total_recipes", 0) == 0:
        return render_template(
            "model_comparison.html",
            error="No inventory found. Please add items to your kitchen first to run the analysis.",
            exact={}, tfidf={}, proposed={}, baseline={}, deltas={}
        )

    # For the 3-column UI, we now have genuinely separate models:
    # 1. 'exact'    -> Pure string matching (no TF-IDF)
    # 2. 'tfidf'    -> TF-IDF cosine similarity baseline
    # 3. 'proposed' -> Semantic BERT model
    
    return render_template(
        "model_comparison.html",
        exact=data["exact"],
        tfidf=data["baseline"],
        proposed=data["proposed"],
        deltas=data["deltas"],
        error=None,
    )


# ---------------------------------------------------------------------------
# UPLOAD BILL  (replace the existing upload() route body)
# ---------------------------------------------------------------------------

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "bill" not in request.files:
            return render_template("upload.html", error="No file part")
        file = request.files["bill"]
        if file.filename == "":
            return render_template("upload.html", error="No selected file")
        if not (file and allowed_file(file.filename)):
            return render_template(
                "upload.html",
                error="Invalid file type. Please upload an image (PNG, JPG, JPEG, WEBP)."
            )
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        # ── STEP 1: Detect image type (bill / product / other) ──────────────
        image_type = detect_image_type(path)

        if image_type == "other":
            try:
                os.remove(path)
            except OSError:
                pass
            return render_template(
                "upload.html",
                error=(
                    "The uploaded image does not appear to be a grocery bill or "
                    "grocery product photo. Please upload a grocery receipt or a "
                    "food/grocery product image."
                ),
            )

        username = session.get("user", "default")
        conn = sqlite3.connect("grocery_ocr.db")
        cur = conn.cursor()
        cur.execute("SELECT MAX(serial_no) FROM grocery_items WHERE user_id = ?", (username,))
        res = cur.fetchone()[0]
        sno = (res if res else 0) + 1
        unknown_items = []

        # ── STEP 2a: PRODUCT IMAGE PATH ─────────────────────────────────────
        if image_type == "product":
            product_name = extract_product_name_from_image(path)
            try:
                os.remove(path)
            except OSError:
                pass

            if not product_name:
                conn.close()
                return render_template(
                    "upload.html",
                    error=(
                        "Could not identify the grocery product in the image. "
                        "Please make sure the product label is clearly visible, "
                        "or add the item manually."
                    ),
                )

            expiry_result = calculate_expiry(product_name)
            if expiry_result is None:
                # Not in shelf life DB — ask user for expiry date
                unknown_items.append({"item": product_name, "qty": "1 pc"})
            else:
                purchase, expiry, days_left = expiry_result
                cur.execute(
                    """
                    INSERT INTO grocery_items
                        (serial_no, item, quantity, purchase_date, expiry_date, days_left, user_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (sno, product_name, "1 pc", purchase, expiry, days_left, username),
                )
                sno += 1

            conn.commit()
            conn.close()

            if unknown_items:
                session["pending_items"] = unknown_items
                session.modified = True
                return redirect("/ask_expiry_batch")

            return redirect("/inventory")

        # ── STEP 2b: GROCERY BILL PATH ──────────────────────────────────────
        # Run a quick OCR pass first so the fallback classifier can also use
        # the raw text if needed.
        raw_result = ocr.ocr(path)

        # Secondary gate: verify OCR text has grocery signals (fallback safety net)
        ok, reason = classify_image_for_grocery(path, raw_ocr_result=raw_result)
        if not ok:
            try:
                os.remove(path)
            except OSError:
                pass
            conn.close()
            return render_template("upload.html", error=reason)

        grocery_items: list = []
        if raw_result is not None:
            grocery_items = extract_grocery_items(raw_result)
        if not grocery_items:
            proc_path = preprocess_image(path)
            if proc_path != path:
                raw_result2 = ocr.ocr(proc_path)
                if raw_result2 is not None:
                    grocery_items = extract_grocery_items_low_conf(raw_result2)
                try:
                    os.remove(proc_path)
                except OSError:
                    pass
        if not grocery_items:
            conn.close()
            return render_template(
                "upload.html",
                error=(
                    "Could not detect any grocery items. "
                    "Make sure the bill is clear and well-lit, or add items manually."
                ),
            )

        for item, qty in grocery_items:
            result = calculate_expiry(item)
            if result is None:
                unknown_items.append({"item": item, "qty": qty})
            else:
                purchase, expiry, days_left = result
                cur.execute(
                    """
                    INSERT INTO grocery_items
                        (serial_no, item, quantity, purchase_date, expiry_date, days_left, user_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (sno, item, qty, purchase, expiry, days_left, username),
                )
                sno += 1

        conn.commit()
        conn.close()

        if unknown_items:
            session["pending_items"] = unknown_items
            session.modified = True
            return redirect("/ask_expiry_batch")

        return redirect("/inventory")
    return render_template("upload.html")




# ---------------------------------------------------------------------------
# MANUAL ADD  (replace existing manual_add())
# ---------------------------------------------------------------------------

@app.route("/manual_add", methods=["POST"])
def manual_add():
    raw_item = request.form["item"].strip()
    qty      = request.form["qty"]
    cleaned  = clean_item_name(raw_item)
    item     = cleaned if cleaned else raw_item.title()

    result = calculate_expiry(item)
    if result is None:
        # Not in CSV — ask user
        from urllib.parse import urlencode
        params = urlencode({"item": item, "qty": qty, "source": "manual"})
        return redirect(f"/ask_expiry?{params}")

    purchase, expiry, days_left = result
    username = session.get("user", "default")
    conn = sqlite3.connect("grocery_ocr.db")
    cur  = conn.cursor()
    cur.execute("SELECT MAX(serial_no) FROM grocery_items WHERE user_id = ?", (username,))
    res = cur.fetchone()[0]
    sno = (res if res else 0) + 1
    cur.execute("""
        INSERT INTO grocery_items(serial_no, item, quantity, purchase_date, expiry_date, days_left, user_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (sno, item, qty, purchase, expiry, days_left, username))
    conn.commit()
    conn.close()
    return redirect("/inventory")


# ---------------------------------------------------------------------------
# BATCH ASK — for items detected via OCR that are not in the CSV
# ---------------------------------------------------------------------------

@app.route("/ask_expiry_batch", methods=["GET"])
def ask_expiry_batch():
    """
    Shows the ask_expiry form for the first unknown item in session["pending_items"].
    Loops until the list is empty, then redirects to /inventory.
    """
    if "user" not in session:
        return redirect("/")

    pending = session.get("pending_items", [])
    if not pending:
        return redirect("/inventory")

    current = pending[0]
    return render_template(
        "ask_expiry.html",
        item=current["item"],
        qty=current["qty"],
        source="batch",
        expiry_date="",
        remaining=len(pending),
    )


@app.route("/save_expiry_batch", methods=["POST"])
def save_expiry_batch():
    """
    Saves one item from the batch queue, pops it, and loops back to
    /ask_expiry_batch until the queue is empty.
    """
    if "user" not in session:
        return redirect("/")

    item            = request.form.get("item", "").strip()
    qty             = request.form.get("qty", "1 pc").strip()
    fridge_days_str = request.form.get("fridge_days", "").strip()
    pantry_days_str = request.form.get("pantry_days", "").strip()
    expiry_date_str = request.form.get("expiry_date", "").strip()

    today_str = str(datetime.now().date())

    if expiry_date_str:
        try:
            expiry_dt = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
            days_left = (expiry_dt - datetime.now().date()).days
        except ValueError:
            days_left       = 10
            expiry_date_str = str(datetime.now().date() + timedelta(days=10))
    else:
        try:
            fridge_days = int(fridge_days_str) if fridge_days_str else None
        except ValueError:
            fridge_days = None
        try:
            pantry_days = int(pantry_days_str) if pantry_days_str else None
        except ValueError:
            pantry_days = None

        if item and (fridge_days or pantry_days):
            save_shelf_life(item, fridge_days, pantry_days)

        f = fridge_days if fridge_days and fridge_days > 0 else None
        p = pantry_days if pantry_days and pantry_days > 0 else None
        if f and p:
            days_left = round(0.7 * f + 0.3 * p)
        elif f:
            days_left = f
        elif p:
            days_left = p
        else:
            days_left = 10

        expiry_date_str = str(datetime.now().date() + timedelta(days=days_left))

    if item:
        username = session["user"]
        conn = sqlite3.connect("grocery_ocr.db")
        cur  = conn.cursor()
        cur.execute("SELECT MAX(serial_no) FROM grocery_items WHERE user_id = ?", (username,))
        res = cur.fetchone()[0]
        sno = (res if res else 0) + 1
        cur.execute("""
            INSERT INTO grocery_items(serial_no, item, quantity, purchase_date, expiry_date, days_left, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (sno, item, qty, today_str, expiry_date_str, days_left, username))
        conn.commit()
        conn.close()

    # Pop the processed item from the queue
    pending = session.get("pending_items", [])
    if pending:
        pending.pop(0)
    session["pending_items"] = pending
    session.modified = True

    if pending:
        return redirect("/ask_expiry_batch")
    return redirect("/inventory")


# ---------------------------
# CONSOLIDATE INVENTORY
# ---------------------------
def consolidate_inventory(username):
    conn = sqlite3.connect("grocery_ocr.db")
    # FIX-APP-2: use pd.read_sql_query for pandas >= 2.0 compatibility
    inv_df = pd.read_sql_query(
        "SELECT * FROM grocery_items WHERE user_id = ?", conn, params=(username,)
    )
    if inv_df.empty:
        conn.close()
        return

    def canonical(name):
        cleaned = clean_item_name(str(name))
        return cleaned if cleaned else str(name).lower().strip()

    inv_df["expiry_date"]   = pd.to_datetime(inv_df["expiry_date"],   format="mixed", errors="coerce")
    inv_df["purchase_date"] = pd.to_datetime(inv_df["purchase_date"], format="mixed", errors="coerce")
    inv_df = inv_df.dropna(subset=["expiry_date", "purchase_date"])
    inv_df["canonical"] = inv_df["item"].apply(canonical)
    to_delete = []

    for canon_key, group in inv_df.groupby("canonical"):
        primary_row = group.iloc[0]
        primary_id = int(primary_row["id"])
        total_qty = 0.0
        common_unit = None
        for q in group["quantity"]:
            val = parse_qty(q)
            u = get_unit(q)
            base_val, base_unit = convert_to_base(val, u)
            if common_unit is None:
                common_unit = base_unit
            if base_unit == common_unit:
                total_qty += base_val
        if common_unit:
            qty_num = int(total_qty) if total_qty == int(total_qty) else round(total_qty, 2)
            new_qty = f"{qty_num} {common_unit}"
        else:
            new_qty = str(primary_row["quantity"])
        earliest_purchase = group["purchase_date"].min()
        latest_expiry     = group["expiry_date"].max()
        today_dt  = datetime.now().date()
        days_left = (latest_expiry.date() - today_dt).days
        clean_name = canon_key.title()
        conn.execute(
            "UPDATE grocery_items SET item=?, quantity=?, purchase_date=?, expiry_date=?, days_left=? WHERE id=?",
            (clean_name, new_qty, str(earliest_purchase.date()),
             str(latest_expiry.date()), int(days_left), primary_id)
        )
        to_delete.extend(group["id"].iloc[1:].tolist())

    if to_delete:
        conn.execute(
            f"DELETE FROM grocery_items WHERE id IN ({','.join(['?'] * len(to_delete))})",
            tuple(to_delete)
        )
    conn.commit()
    conn.close()


# ---------------------------
# INVENTORY DASHBOARD
# ---------------------------
@app.route("/inventory")
def inventory():
    if "user" not in session:
        return redirect("/")
    username = session["user"]
    consolidate_inventory(username)
    conn = sqlite3.connect("grocery_ocr.db")
    inv_df = pd.read_sql_query(
        "SELECT * FROM grocery_items WHERE user_id = ?", conn, params=(username,)
    )
    conn.close()
    if inv_df.empty:
        return render_template("inventory.html", items=[], summary={}, chart_data={})
    inv_df["expiry_date"] = pd.to_datetime(inv_df["expiry_date"], errors="coerce")
    inv_df = inv_df.dropna(subset=["expiry_date"])
    today = pd.Timestamp(datetime.now().date())
    inv_df["days_left"] = (inv_df["expiry_date"] - today).dt.days

    def get_status(days):
        if days < 0:    return "Expired"
        elif days <= 3: return "Urgent"
        elif days <= 7: return "Use Soon"
        else:           return "Fresh"

    inv_df["status"] = inv_df["days_left"].apply(get_status)
    color_map = {"Fresh": "success", "Use Soon": "warning", "Urgent": "danger", "Expired": "dark"}
    inv_df["color"] = inv_df["status"].map(color_map)
    inv_df = inv_df.sort_values(by="days_left")
    summary = {
        "total":    len(inv_df),
        "fresh":    len(inv_df[inv_df.status == "Fresh"]),
        "use_soon": len(inv_df[inv_df.status == "Use Soon"]),
        "urgent":   len(inv_df[inv_df.status == "Urgent"]),
        "wasted":   len(inv_df[inv_df.status == "Expired"]),
    }
    chart_data = {
        "Fresh":   summary["fresh"],
        "UseSoon": summary["use_soon"],
        "Urgent":  summary["urgent"],
        "Expired": summary["wasted"],
    }
    inv_df["expiry_date"]   = inv_df["expiry_date"].dt.strftime("%Y-%m-%d")
    inv_df["purchase_date"] = pd.to_datetime(
        inv_df["purchase_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    items = inv_df.to_dict(orient="records")
    return render_template("inventory.html", items=items, summary=summary, chart_data=chart_data)


# ---------------------------
# UPDATE QUANTITY / EXPIRY / MARK USED / WASTED
# ---------------------------
@app.route("/update_quantity/<int:id>", methods=["POST"])
def update_quantity(id):
    quantity = request.form["quantity"]
    username = session["user"]
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    cur.execute("UPDATE grocery_items SET quantity=? WHERE id=? AND user_id=?",
                (quantity, id, username))
    conn.commit()
    conn.close()
    return redirect("/inventory")


@app.route("/update_expiry/<int:id>", methods=["POST"])
def update_expiry(id):
    new_expiry = request.form["expiry_date"]
    username = session["user"]
    expiry_dt = datetime.strptime(new_expiry, '%Y-%m-%d').date()
    today = datetime.now().date()
    days_left = (expiry_dt - today).days
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    cur.execute(
        "UPDATE grocery_items SET expiry_date=?, days_left=? WHERE id=? AND user_id=?",
        (new_expiry, days_left, id, username)
    )
    conn.commit()
    conn.close()
    return redirect("/inventory")


@app.route("/mark_used/<int:id>", methods=["POST"])
def mark_used(id):
    username = session["user"]
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM grocery_items WHERE id=? AND user_id=?", (id, username))
    conn.commit()
    conn.close()
    return redirect("/inventory")


@app.route("/inventory/clear", methods=["POST"])
def clear_inventory():
    username = session["user"]
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM grocery_items WHERE user_id=?", (username,))
    conn.commit()
    conn.close()
    return redirect("/inventory")


@app.route("/inventory/cleanup_billing", methods=["POST"])
def cleanup_billing_items():
    from ocr_utils import is_skip_line
    username = session.get("user", "default")
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    cur.execute("SELECT id, item FROM grocery_items WHERE user_id=?", (username,))
    rows = cur.fetchall()
    removed = 0
    for row_id, item_name in rows:
        if is_skip_line(item_name):
            cur.execute("DELETE FROM grocery_items WHERE id=?", (row_id,))
            removed += 1
    conn.commit()
    conn.close()
    return redirect("/inventory")


@app.route("/mark_wasted/<int:id>", methods=["POST"])
def mark_wasted(id):
    username = session["user"]
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM grocery_items WHERE id=? AND user_id=?", (id, username))
    conn.commit()
    conn.close()
    return redirect("/inventory")


# ---------------------------
# GROCERY LIST
# ---------------------------
@app.route("/grocery_list")
def grocery_list():
    if "user" not in session:
        return redirect("/")
    username = session["user"]
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM grocery_list WHERE user_id = ? ORDER BY added_date DESC", (username,)
    )
    items = cur.fetchall()
    conn.close()
    formatted_items = [
        {"id": item[0], "item": item[1], "quantity": item[2], "added_date": item[3]}
        for item in items
    ]
    return render_template("grocery_list.html", items=formatted_items)


@app.route("/grocery_list/export_pdf")
def export_grocery_pdf():
    if "user" not in session:
        return redirect("/")
    username = session["user"]
    
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    cur.execute("SELECT id, item, quantity FROM grocery_list WHERE user_id = ?", (username,))
    rows = cur.fetchall()
    conn.close()
    
    items = [{"id": r[0], "item": r[1], "quantity": r[2]} for r in rows]
    
    from recommender import generate_grocery_pdf
    pdf_path = generate_grocery_pdf(username, items)
    from flask import send_file
    return send_file(pdf_path, as_attachment=True)


@app.route("/grocery_list/add", methods=["POST"])
def add_grocery_item():
    raw_item = request.form.get("item", "").strip()
    qty = request.form.get("qty")
    today = datetime.now().date()
    if raw_item:
        cleaned = clean_item_name(raw_item)
        item = cleaned if cleaned else raw_item.title()
        username = session["user"]
        conn = sqlite3.connect("grocery_ocr.db")
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO grocery_list(item, quantity, added_date, user_id) VALUES(?,?,?,?)",
            (item, qty, today, username)
        )
        conn.commit()
        conn.close()
    return redirect("/grocery_list")


@app.route("/grocery_list/add_missing", methods=["POST"])
def add_missing_ingredients():
    missing_str = request.form.get("missing", "")
    if missing_str:
        items = [i.strip() for i in missing_str.split(",") if i.strip()]
        return render_template("confirm_grocery.html", items=items)
    return redirect("/grocery_list")


@app.route("/grocery_list/finalize_bulk_add", methods=["POST"])
def finalize_bulk_add():
    items = request.form.getlist("items[]")
    quantities = request.form.getlist("quantities[]")
    today = datetime.now().date()
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    username = session["user"]
    for raw_item, qty in zip(items, quantities):
        raw_item = raw_item.strip()
        if raw_item and qty:
            cleaned = clean_item_name(raw_item)
            item = cleaned if cleaned else raw_item.title()
            cur.execute(
                "INSERT INTO grocery_list(item, quantity, added_date, user_id) VALUES(?,?,?,?)",
                (item, qty, today, username)
            )
    conn.commit()
    conn.close()
    return redirect("/grocery_list")


@app.route("/grocery_list/delete/<int:id>", methods=["POST"])
def delete_grocery_item(id):
    username = session["user"]
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM grocery_list WHERE id=? AND user_id=?", (id, username))
    conn.commit()
    conn.close()
    return redirect("/grocery_list")


@app.route("/grocery_list/clear", methods=["POST"])
def clear_grocery_list():
    username = session["user"]
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM grocery_list WHERE user_id=?", (username,))
    conn.commit()
    conn.close()
    return redirect("/grocery_list")


# ---------------------------
# CONTEXT PROCESSOR
# FIX-APP-7: compute days_remaining freshly from expiry_date, not stale days_left
# ---------------------------
@app.context_processor
def inject_notifications():
    if "user" not in session:
        return {"expiry_notifs": []}
    from flask import request as flask_request
    if flask_request.path.startswith("/static"):
        return {"expiry_notifs": []}
    username = session["user"]
    conn = sqlite3.connect("grocery_ocr.db")
    cur = conn.cursor()
    cur.execute(
        """
        SELECT item,
               CAST(julianday(expiry_date) - julianday('now') AS INTEGER) AS days_remaining
        FROM grocery_items
        WHERE julianday(expiry_date) - julianday('now') <= 3
          AND julianday(expiry_date) - julianday('now') >= 0
          AND user_id = ?
        ORDER BY days_remaining ASC
        LIMIT 5
        """,
        (username,)
    )
    rows = cur.fetchall()
    conn.close()
    return {
        "expiry_notifs": rows,
        "MAIL_ENABLED": MAIL_ENABLED,
        "SENDER_EMAIL": SENDER_EMAIL
    }


# ---------------------------
# RECIPE RECOMMENDATION
# ---------------------------
@app.route("/recipes", methods=["GET", "POST"])
def recipes():
    cuisine = request.form.get("cuisine", request.args.get("cuisine", "Any"))
    diet = request.form.get("diet", request.args.get("diet", "Any"))
    meal = request.form.get("meal", request.args.get("meal", "Any"))
    search = request.form.get("search", request.args.get("search", "")).strip()
    offset = request.args.get("offset", 0, type=int)

    if request.method == "POST":
        # Reset offset on new search/filter submit
        offset = 0

    username = session.get("user", "default")
    recipes_list = recommend_recipes(username, cuisine, diet, meal, search_term=search, offset=offset)
    return render_template(
        "recipes.html",
        recipes=recipes_list,
        cuisine=cuisine,
        diet=diet,
        meal=meal,
        search=search,
        offset=offset,
    )


@app.route("/download_pdf/<recipe_name>")
def download_pdf(recipe_name):
    if "user" not in session:
        return redirect("/")
    df_csv = pd.read_csv("Indian_Food_Dataset.csv")
    recipe_row = df_csv[df_csv["RecipeName"] == recipe_name]
    if recipe_row.empty:
        return "Recipe not found", 404
    row = recipe_row.iloc[0]
    # FIX-APP-10: guard eval() against malformed CSV rows
    try:
        ingredients_list = eval(row["Cleaned_Ingredients"])
        ingredients = ", ".join(ingredients_list)
    except Exception:
        ingredients = str(row["Cleaned_Ingredients"])
    instructions = row["TranslatedInstructions"]
    pdf_path = generate_recipe_pdf(recipe_name, ingredients, "", instructions)
    from flask import send_file
    return send_file(pdf_path, as_attachment=True)


@app.route("/test_email")
def test_email_route():
    if "user" not in session:
        return "Please log in first", 401
    
    if not MAIL_ENABLED:
        return "Email is currently disabled in app.py (MAIL_ENABLED = False). Please add your credentials first.", 400
        
    username = session.get("username", "User")
    user_email = session.get("user") # This contains the email from login session
    
    # Mock some expiring items for the test
    test_items = [("Apples", 2), ("Milk", 0), ("Bread", 1)]
    
    success = send_expiry_notification(user_email, username, test_items)
    if success:
        return f"Test email sent successfully to {user_email}!"
    else:
        return "Failed to send test email. Check server logs for errors.", 500


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)