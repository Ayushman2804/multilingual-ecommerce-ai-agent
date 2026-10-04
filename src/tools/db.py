"""Mock SQLite database for E-commerce Orders, Products, and Returns."""
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "mock_db" / "store.db"


def get_db_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_db_initialized():
    """Checks if DB exists and has tables, initializes if not."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='orders'")
    if not cursor.fetchone():
        conn.close()
        init_mock_db()
    else:
        conn.close()



def init_mock_db():
    """Initializes tables and seeds realistic multilingual customer order data."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            customer_name TEXT NOT NULL,
            item_name TEXT NOT NULL,
            status TEXT NOT NULL,
            amount REAL NOT NULL,
            currency TEXT NOT NULL,
            order_date TEXT NOT NULL,
            tracking_number TEXT,
            days_since_delivery INTEGER
        );

        CREATE TABLE IF NOT EXISTS products (
            product_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            currency TEXT NOT NULL,
            in_stock INTEGER NOT NULL,
            description TEXT
        );

        CREATE TABLE IF NOT EXISTS refunds (
            refund_id TEXT PRIMARY KEY,
            order_id TEXT NOT NULL,
            status TEXT NOT NULL,
            reason TEXT NOT NULL,
            amount REAL NOT NULL,
            currency TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(order_id) REFERENCES orders(order_id)
        );
    """)

    # Seed mock data
    cursor.execute("DELETE FROM orders;")
    cursor.execute("DELETE FROM products;")
    cursor.execute("DELETE FROM refunds;")

    cursor.executemany("""
        INSERT INTO orders (order_id, customer_id, customer_name, item_name, status, amount, currency, order_date, tracking_number, days_since_delivery)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        ("ORD-1001", "CUST-01", "John Doe", "Wireless Noise-Canceling Headphones", "delivered", 199.99, "USD", "2026-09-15", "TRK-US-9912", 12),
        ("ORD-1002", "CUST-02", "Maria Garcia", "Smart Fitness Watch", "shipped", 149.50, "EUR", "2026-10-01", "TRK-ES-4421", None),
        ("ORD-1003", "CUST-03", "Pierre Dubois", "USB-C Mechanical Keyboard", "processing", 89.00, "EUR", "2026-10-03", None, None),
        ("ORD-1004", "CUST-04", "Hans Mueller", "Ergonomic Office Chair", "delivered", 320.00, "EUR", "2026-08-10", "TRK-DE-8823", 45),
        ("ORD-1005", "CUST-05", "Kenji Sato", "Ultra-HD 4K Monitor", "delivered", 38000, "JPY", "2026-09-28", "TRK-JP-1190", 5),
    ])

    cursor.executemany("""
        INSERT INTO products (product_id, name, category, price, currency, in_stock, description)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, [
        ("PROD-01", "Wireless Noise-Canceling Headphones", "electronics", 199.99, "USD", 25, "Active noise cancellation with 30-hour battery life."),
        ("PROD-02", "Smart Fitness Watch", "wearables", 149.50, "EUR", 12, "Waterproof health tracker with heart rate and sleep monitor."),
        ("PROD-03", "USB-C Mechanical Keyboard", "accessories", 89.00, "EUR", 40, "Tactile brown switches with RGB backlighting."),
        ("PROD-04", "Ultra-HD 4K Monitor", "electronics", 38000, "JPY", 8, "27-inch IPS panel with HDR400 support."),
    ])

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_mock_db()
    print("Database initialized successfully.")
