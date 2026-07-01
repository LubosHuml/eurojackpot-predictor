import os
import sqlite3
import csv
from datetime import datetime

DB_PATH = "sportka.db"
CSV_PATH = "c:\\Users\\Acer\\Desktop\\Euro\\sportka.csv"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS draws (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            draw_date TEXT,
            draw_num INTEGER,
            num1 INTEGER,
            num2 INTEGER,
            num3 INTEGER,
            num4 INTEGER,
            num5 INTEGER,
            num6 INTEGER,
            supplementary INTEGER
        )
    """)
    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_date_draw ON draws(draw_date, draw_num)
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            draw_date TEXT,
            row_id INTEGER,
            profile TEXT,
            num1 INTEGER,
            num2 INTEGER,
            num3 INTEGER,
            num4 INTEGER,
            num5 INTEGER,
            num6 INTEGER,
            confidence REAL,
            matched_nums TEXT,
            prize_tier TEXT DEFAULT 'Pending'
        )
    """)
    conn.commit()
    conn.close()

def load_csv_data():
    if not os.path.exists(CSV_PATH):
        print(f"CSV file not found at {CSV_PATH}")
        return
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Read CSV
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        
        count = 0
        for row in reader:
            if len(row) < 18:
                continue
                
            try:
                # Parse date e.g. "28. 6. 2026"
                date_str = row[0].strip()
                # Convert to standard YYYY-MM-DD
                parts = [p.strip() for p in date_str.split(".")]
                if len(parts) != 3 or not parts[2]:
                    continue
                day = int(parts[0])
                month = int(parts[1])
                year = int(parts[2])
                db_date = f"{year:04d}-{month:02d}-{day:02d}"
                
                # Draw 1
                nums1 = sorted([int(row[4]), int(row[5]), int(row[6]), int(row[7]), int(row[8]), int(row[9])])
                supp1 = int(row[10])
                
                # Draw 2
                nums2 = sorted([int(row[11]), int(row[12]), int(row[13]), int(row[14]), int(row[15]), int(row[16])])
                supp2 = int(row[17])
                
                # Insert Draw 1
                cursor.execute("""
                    INSERT OR IGNORE INTO draws (draw_date, draw_num, num1, num2, num3, num4, num5, num6, supplementary)
                    VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?)
                """, (db_date, nums1[0], nums1[1], nums1[2], nums1[3], nums1[4], nums1[5], supp1))
                
                # Insert Draw 2
                cursor.execute("""
                    INSERT OR IGNORE INTO draws (draw_date, draw_num, num1, num2, num3, num4, num5, num6, supplementary)
                    VALUES (?, 2, ?, ?, ?, ?, ?, ?, ?)
                """, (db_date, nums2[0], nums2[1], nums2[2], nums2[3], nums2[4], nums2[5], supp2))
                
                count += 2
            except ValueError:
                # Skip header-like or empty rows
                continue
                
    inserted = conn.total_changes
    conn.commit()
    cursor.execute("SELECT COUNT(*) FROM draws")
    total = cursor.fetchone()[0]
    conn.close()
    print(f"Loaded CSV data. Total rows in database: {total}, new: {inserted}")
    return inserted

def get_all_draws():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT draw_date, draw_num, num1, num2, num3, num4, num5, num6, supplementary FROM draws ORDER BY draw_date ASC, draw_num ASC")
    rows = cursor.fetchall()
    conn.close()
    
    draws = []
    for r in rows:
        draws.append({
            "date": r[0],
            "draw_num": r[1],
            "numbers": [r[2], r[3], r[4], r[5], r[6], r[7]],
            "supplementary": r[8]
        })
    return draws

def save_ticket(draw_date, row_id, profile, nums, confidence):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO tickets (draw_date, row_id, profile, num1, num2, num3, num4, num5, num6, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (draw_date, row_id, profile, nums[0], nums[1], nums[2], nums[3], nums[4], nums[5], confidence))
    conn.commit()
    conn.close()

def get_all_tickets():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT draw_date, row_id, profile, num1, num2, num3, num4, num5, num6, confidence, matched_nums, prize_tier FROM tickets ORDER BY draw_date DESC, row_id ASC")
    rows = cursor.fetchall()
    conn.close()
    
    tickets = []
    for r in rows:
        tickets.append({
            "draw_date": r[0],
            "row_id": r[1],
            "profile": r[2],
            "nums": [r[3], r[4], r[5], r[6], r[7], r[8]],
            "confidence": r[9],
            "matched_nums": eval(r[10]) if r[10] else [],
            "prize_tier": r[11]
        })
    return tickets

def update_ticket_results(draw_date, row_id, matched_nums, prize_tier):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE tickets
        SET matched_nums = ?, prize_tier = ?
        WHERE draw_date = ? AND row_id = ?
    """, (str(matched_nums), prize_tier, draw_date, row_id))
    conn.commit()
    conn.close()

def has_ticket_for_date(draw_date):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM tickets WHERE draw_date = ?", (draw_date,))
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0

if __name__ == "__main__":
    init_db()
    load_csv_data()
