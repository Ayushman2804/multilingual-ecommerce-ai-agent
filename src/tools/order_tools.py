"""Order status, refund processing, and product catalog tools."""
import uuid
import datetime
from typing import Dict, Any, List
from src.tools.db import get_db_connection


def lookup_order(order_id: str) -> Dict[str, Any]:
    """Retrieves order details, delivery status, and tracking information."""
    from src.tools.db import ensure_db_initialized
    ensure_db_initialized()
    order_id = order_id.strip().upper()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT order_id, customer_id, customer_name, item_name, status, amount, currency, order_date, tracking_number, days_since_delivery
        FROM orders
        WHERE order_id = ?
    """, (order_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return {
            "found": False,
            "order_id": order_id,
            "message": f"Order {order_id} could not be found in our database."
        }

    return {
        "found": True,
        "order_id": row["order_id"],
        "customer_id": row["customer_id"],
        "customer_name": row["customer_name"],
        "item_name": row["item_name"],
        "status": row["status"],
        "amount": row["amount"],
        "currency": row["currency"],
        "order_date": row["order_date"],
        "tracking_number": row["tracking_number"],
        "days_since_delivery": row["days_since_delivery"],
    }


def request_return_or_refund(order_id: str, reason: str) -> Dict[str, Any]:
    """Validates 30-day return eligibility and creates a refund request."""
    order_id = order_id.strip().upper()
    order = lookup_order(order_id)
    if not order["found"]:
        return {
            "success": False,
            "order_id": order_id,
            "error": "Order not found. Please provide a valid order ID."
        }

    # Policy validation: must be delivered and within 30 days
    if order["status"] != "delivered":
        return {
            "success": False,
            "order_id": order_id,
            "status": order["status"],
            "error": f"Order {order_id} has not been delivered yet (current status: {order['status']}). Returns are only valid after delivery."
        }

    days_since = order.get("days_since_delivery")
    if days_since is not None and days_since > 30:
        return {
            "success": False,
            "order_id": order_id,
            "days_since_delivery": days_since,
            "error": f"Order was delivered {days_since} days ago, exceeding the 30-day return policy window. Escalation required."
        }

    # Record refund in database
    refund_id = f"REF-{uuid.uuid4().hex[:6].upper()}"
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO refunds (refund_id, order_id, status, reason, amount, currency, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        refund_id,
        order_id,
        "approved",
        reason,
        order["amount"],
        order["currency"],
        datetime.datetime.now(datetime.timezone.utc).isoformat(),
    ))
    conn.commit()
    conn.close()

    return {
        "success": True,
        "refund_id": refund_id,
        "order_id": order_id,
        "amount": order["amount"],
        "currency": order["currency"],
        "status": "approved",
        "instructions": "A prepaid return label has been generated. Once our warehouse receives the item, your refund will post in 3-5 business days."
    }


def lookup_product(query: str) -> Dict[str, Any]:
    """Searches product catalog by name or category."""
    from src.tools.db import ensure_db_initialized
    ensure_db_initialized()
    conn = get_db_connection()
    cursor = conn.cursor()
    search_term = f"%{query.strip()}%"
    cursor.execute("""
        SELECT product_id, name, category, price, currency, in_stock, description
        FROM products
        WHERE name LIKE ? OR category LIKE ? OR description LIKE ?
        ORDER BY 
            CASE 
                WHEN name LIKE ? THEN 1 
                WHEN category LIKE ? THEN 2 
                ELSE 3 
            END,
            price ASC
    """, (search_term, search_term, search_term, search_term, search_term))
    rows = cursor.fetchall()
    conn.close()

    items = [
        {
            "product_id": r["product_id"],
            "name": r["name"],
            "category": r["category"],
            "price": r["price"],
            "currency": r["currency"],
            "in_stock": r["in_stock"],
            "description": r["description"],
        }
        for r in rows
    ]
    return {
        "query": query,
        "count": len(items),
        "products": items,
    }
