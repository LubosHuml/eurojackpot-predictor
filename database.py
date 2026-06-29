import sqlite3
import os

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eurojackpot.db")

def get_connection():
    return sqlite3.connect(DB_FILE)

def init_db():
    """Initializes the database and creates the draws table if it doesn't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS draws (
            date TEXT PRIMARY KEY,
            num1 INTEGER NOT NULL,
            num2 INTEGER NOT NULL,
            num3 INTEGER NOT NULL,
            num4 INTEGER NOT NULL,
            num5 INTEGER NOT NULL,
            euro1 INTEGER NOT NULL,
            euro2 INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def insert_draw(date, main_nums, euro_nums):
    """
    Inserts a single draw into the database.
    
    Parameters:
        date (str): Format YYYY-MM-DD
        main_nums (list): List of 5 integers (main numbers)
        euro_nums (list): List of 2 integers (Euro numbers)
        
    Returns:
        bool: True if inserted successfully, False if already exists (primary key conflict)
    """
    if len(main_nums) != 5 or len(euro_nums) != 2:
        raise ValueError("Invalid number of balls. Expected 5 main numbers and 2 Euro numbers.")
    
    # Ensure they are sorted
    main_sorted = sorted([int(n) for n in main_nums])
    euro_sorted = sorted([int(n) for n in euro_nums])
    
    # Guardrails
    for n in main_sorted:
        if not (1 <= n <= 50):
            raise ValueError(f"Main number {n} out of bounds (1-50).")
    for n in euro_sorted:
        if not (1 <= n <= 12):
            raise ValueError(f"Euro number {n} out of bounds (1-12).")
            
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO draws (date, num1, num2, num3, num4, num5, euro1, euro2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (date, main_sorted[0], main_sorted[1], main_sorted[2], main_sorted[3], main_sorted[4], euro_sorted[0], euro_sorted[1]))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_all_draws():
    """
    Returns a list of dicts with all draws sorted by date ascending.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT date, num1, num2, num3, num4, num5, euro1, euro2 FROM draws ORDER BY date ASC")
    rows = cursor.fetchall()
    conn.close()
    
    draws = []
    for r in rows:
        draws.append({
            'date': r[0],
            'main_nums': [r[1], r[2], r[3], r[4], r[5]],
            'euro_nums': [r[6], r[7]]
        })
    return draws

def get_latest_draw_date():
    """
    Returns the latest draw date in the database, or None if empty.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(date) FROM draws")
    row = cursor.fetchone()
    conn.close()
    return row[0] if (row and row[0]) else None
