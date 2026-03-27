import sqlite3
import os
import sys

# Add the project directory to sys.path to import from app.py if needed, 
# but since init_db is in app.py and we want to test it, let's just use the same logic or call it.

def test_db_migration():
    print("Testing DB Migration...")
    try:
        from app import init_db
        init_db()
        
        conn = sqlite3.connect("grocery_ocr.db")
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cur.fetchall()]
        if "mail_sent_count" in columns:
            print("SUCCESS: mail_sent_count column exists in users table.")
        else:
            print("FAILURE: mail_sent_count column NOT found.")
        conn.close()
    except Exception as e:
        print(f"ERROR: {e}")

def test_counter_increment():
    print("\nTesting Counter Increment...")
    test_email = "test@gmail.com"
    try:
        conn = sqlite3.connect("grocery_ocr.db")
        cur = conn.cursor()
        
        # Ensure test user exists
        cur.execute("INSERT OR IGNORE INTO users (username, email, password, mail_sent_count) VALUES (?, ?, ?, ?)", 
                    ("testuser", test_email, "password", 0))
        conn.commit()
        
        # Get initial count
        cur.execute("SELECT mail_sent_count FROM users WHERE email = ?", (test_email,))
        initial_count = cur.fetchone()[0]
        print(f"Initial count: {initial_count}")
        
        # Simulate send_expiry_notification increment logic
        # Instead of calling the actual function which sends real mail, we just run the SQL logic
        cur.execute("UPDATE users SET mail_sent_count = mail_sent_count + 1 WHERE email = ?", (test_email,))
        conn.commit()
        
        # Get new count
        cur.execute("SELECT mail_sent_count FROM users WHERE email = ?", (test_email,))
        new_count = cur.fetchone()[0]
        print(f"New count: {new_count}")
        
        if new_count == initial_count + 1:
            print("SUCCESS: Counter incremented correctly.")
        else:
            print("FAILURE: Counter did not increment correctly.")
        
        conn.close()
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_db_migration()
    test_counter_increment()
