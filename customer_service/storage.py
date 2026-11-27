"""Real SQLite storage over synthetic shop data; SQL is never a model tool."""
from contextlib import closing
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import sqlite3
import uuid

SHANGHAI_TZ = timezone(timedelta(hours=8), name='Asia/Shanghai')
CANCELLABLE_STATUSES = frozenset({'pending_payment', 'paid_unshipped'})


class StoreError(ValueError):
    """Safe storage failure without SQL, credentials or exception-body disclosure."""


class SQLiteStore:
    def __init__(self, database_path, allow_writes: bool = False):
        self.path = Path(database_path).resolve()
        self.allow_writes = bool(allow_writes)
        try:
            with closing(self._connect()) as connection:
                metadata = dict(connection.execute('SELECT key, value FROM metadata'))
                if metadata.get('schema_version') != '1' or metadata.get('simulated') != 'true':
                    raise StoreError('Unrecognized simulated shop database')
                self.dataset_version = metadata['dataset_version']
                if not self.dataset_version:
                    raise StoreError('Missing database dataset version')
                connection.execute('SELECT customer_id FROM customers LIMIT 0')
                connection.execute('SELECT product_id, name, price_cents, currency, capacity_ml FROM products LIMIT 0')
                connection.execute('SELECT order_id, customer_id, product_id, quantity, total_cents, currency, status FROM orders LIMIT 0')
                if self.allow_writes:
                    connection.execute('SELECT order_id, customer_id, previous_status, reason, cancelled_at, confirmation_id FROM order_cancellations LIMIT 0')
                    connection.execute('SELECT refund_id, order_id, customer_id, amount_cents, currency, status, reason, updated_at, estimated_arrival, reviewer_note FROM refunds LIMIT 0')
                    connection.execute('SELECT ticket_id, customer_id, order_id, category, priority, status, escalate_to_human, assigned_queue, summary, created_at FROM tickets LIMIT 0')
        except StoreError:
            raise
        except (sqlite3.Error, OSError, KeyError):
            raise StoreError('Simulated shop database is unavailable or invalid') from None

    def _connect(self):
        connection = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True, timeout=5)
        try:
            connection.execute('PRAGMA query_only = ON')
            connection.row_factory = sqlite3.Row
            return connection
        except sqlite3.Error:
            connection.close()
            raise

    def _connect_rw(self):
        if not self.allow_writes:
            raise StoreError('Simulated shop database is configured read-only')
        connection = sqlite3.connect(self.path.as_uri() + '?mode=rw', uri=True, timeout=5, isolation_level=None)
        try:
            connection.execute('PRAGMA foreign_keys = ON')
            connection.row_factory = sqlite3.Row
            return connection
        except sqlite3.Error:
            connection.close()
            raise

    def _read(self, statement, parameters):
        try:
            with closing(self._connect()) as connection:
                return [dict(row) for row in connection.execute(statement, parameters).fetchall()]
        except (sqlite3.Error, OSError):
            raise StoreError('Simulated shop database read failed') from None

    def _source(self, table, identifier):
        return self.path.name + '#' + table + '.' + identifier

    def has_customer(self, customer_id):
        return bool(self._read('SELECT customer_id FROM customers WHERE customer_id = ?', (customer_id,)))

    def get_product(self, product_id):
        rows = self._read('SELECT product_id, name, price_cents, currency, capacity_ml '
                          'FROM products WHERE product_id = ?', (product_id,))
        if not rows:
            return None
        row = rows[0]
        row['source'] = self._source('products', row['product_id'])
        return row

    def get_order(self, order_id, customer_id):
        # s03 adaptation: ownership is part of the SQL query, before data leaves storage.
        rows = self._read('SELECT order_id, product_id, quantity, total_cents, currency, status '
                          'FROM orders WHERE order_id = ? AND customer_id = ?', (order_id, customer_id))
        if not rows:
            return None
        row = rows[0]
        row['source'] = self._source('orders', row['order_id'])
        return row

    def list_orders(self, customer_id):
        rows = self._read('SELECT order_id, status, total_cents, currency FROM orders '
                          'WHERE customer_id = ? ORDER BY order_id', (customer_id,))
        for row in rows:
            row['source'] = self._source('orders', row['order_id'])
        return rows

    def get_cancellation(self, order_id, customer_id):
        rows = self._read(
            'SELECT order_id, previous_status, reason, cancelled_at, confirmation_id '
            'FROM order_cancellations WHERE order_id = ? AND customer_id = ?',
            (order_id, customer_id),
        )
        if not rows:
            return None
        row = rows[0]
        row['source'] = self._source('order_cancellations', row['order_id'])
        return row

    def get_refund_by_order(self, order_id, customer_id):
        rows = self._read(
            'SELECT refund_id, order_id, amount_cents, currency, status, reason, '
            'updated_at, estimated_arrival, reviewer_note '
            'FROM refunds WHERE order_id = ? AND customer_id = ? ORDER BY updated_at DESC LIMIT 1',
            (order_id, customer_id),
        )
        if not rows:
            return None
        row = rows[0]
        row['source'] = self._source('refunds', row['refund_id'])
        return row

    def list_products(self):
        rows = self._read(
            'SELECT product_id, name, price_cents, currency, capacity_ml '
            'FROM products ORDER BY product_id',
            (),
        )
        for row in rows:
            row['source'] = self._source('products', row['product_id'])
        return rows

    def list_refunds(self, customer_id):
        rows = self._read(
            'SELECT refund_id, order_id, amount_cents, currency, status, reason, '
            'updated_at, estimated_arrival, reviewer_note '
            'FROM refunds WHERE customer_id = ? ORDER BY updated_at DESC, refund_id DESC',
            (customer_id,),
        )
        for row in rows:
            row['source'] = self._source('refunds', row['refund_id'])
        return rows

    def list_tickets(self, customer_id):
        rows = self._read(
            'SELECT ticket_id, order_id, category, priority, status, escalate_to_human, '
            'assigned_queue, summary, created_at '
            'FROM tickets WHERE customer_id = ? ORDER BY created_at DESC, ticket_id DESC',
            (customer_id,),
        )
        for row in rows:
            row['escalate_to_human'] = bool(row['escalate_to_human'])
            row['source'] = self._source('tickets', row['ticket_id'])
        return rows

    def cancel_order(self, order_id, customer_id, reason, confirmation_id, cancelled_at=None):
        """Atomically verify ownership + state machine, cancel order, and create refund if paid."""
        timestamp = cancelled_at or datetime.now(SHANGHAI_TZ).isoformat(timespec='seconds')
        try:
            with closing(self._connect_rw()) as connection:
                connection.execute('BEGIN IMMEDIATE')
                try:
                    row = connection.execute(
                        'SELECT order_id, product_id, quantity, total_cents, currency, status '
                        'FROM orders WHERE order_id = ? AND customer_id = ?',
                        (order_id, customer_id),
                    ).fetchone()
                    if row is None:
                        connection.execute('ROLLBACK')
                        return {'outcome': 'not_found'}
                    order = dict(row)
                    current_status = order['status']
                    if current_status == 'cancelled':
                        cancel_row = connection.execute(
                            'SELECT order_id, previous_status, reason, cancelled_at, confirmation_id '
                            'FROM order_cancellations WHERE order_id = ? AND customer_id = ?',
                            (order_id, customer_id),
                        ).fetchone()
                        refund_row = connection.execute(
                            'SELECT refund_id, order_id, amount_cents, currency, status, reason, '
                            'updated_at, estimated_arrival, reviewer_note '
                            'FROM refunds WHERE order_id = ? AND customer_id = ? LIMIT 1',
                            (order_id, customer_id),
                        ).fetchone()
                        connection.execute('ROLLBACK')
                        cancellation = dict(cancel_row) if cancel_row is not None else None
                        if cancellation is not None:
                            cancellation['source'] = self._source('order_cancellations', order_id)
                        refund = dict(refund_row) if refund_row is not None else None
                        if refund is not None:
                            refund['source'] = self._source('refunds', refund['refund_id'])
                        return {
                            'outcome': 'already_cancelled',
                            'order': {**order, 'source': self._source('orders', order_id)},
                            'cancellation': cancellation,
                            'refund': refund,
                        }
                    if current_status not in CANCELLABLE_STATUSES:
                        connection.execute('ROLLBACK')
                        return {
                            'outcome': 'not_cancellable',
                            'order': {**order, 'source': self._source('orders', order_id)},
                        }
                    cursor = connection.execute(
                        "UPDATE orders SET status = 'cancelled' "
                        "WHERE order_id = ? AND customer_id = ? AND status IN ('pending_payment', 'paid_unshipped')",
                        (order_id, customer_id),
                    )
                    if cursor.rowcount != 1:
                        connection.execute('ROLLBACK')
                        return {'outcome': 'not_cancellable', 'order': {**order, 'source': self._source('orders', order_id)}}
                    connection.execute(
                        'INSERT INTO order_cancellations(order_id, customer_id, previous_status, reason, cancelled_at, confirmation_id) '
                        'VALUES (?, ?, ?, ?, ?, ?)',
                        (order_id, customer_id, current_status, reason, timestamp, confirmation_id),
                    )
                    refund = None
                    if current_status == 'paid_unshipped':
                        refund_id = 'RF-' + order_id
                        existing_refund = connection.execute(
                            'SELECT refund_id FROM refunds WHERE order_id = ? AND customer_id = ?',
                            (order_id, customer_id),
                        ).fetchone()
                        if existing_refund is None:
                            connection.execute(
                                'INSERT INTO refunds(refund_id, order_id, customer_id, amount_cents, currency, '
                                'status, reason, updated_at, estimated_arrival, reviewer_note) '
                                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                                (
                                    refund_id, order_id, customer_id, order['total_cents'], order['currency'],
                                    'approved_pending_payout', '未发货订单取消自动退款：' + reason,
                                    timestamp, '预计1-3个工作日原路退回', '未发货取消自动审核通过，已进入打款队列',
                                ),
                            )
                            refund = {
                                'refund_id': refund_id,
                                'order_id': order_id,
                                'amount_cents': order['total_cents'],
                                'currency': order['currency'],
                                'status': 'approved_pending_payout',
                                'reason': '未发货订单取消自动退款：' + reason,
                                'updated_at': timestamp,
                                'estimated_arrival': '预计1-3个工作日原路退回',
                                'reviewer_note': '未发货取消自动审核通过，已进入打款队列',
                                'source': self._source('refunds', refund_id),
                            }
                    connection.execute('COMMIT')
                    return {
                        'outcome': 'cancelled',
                        'order': {**order, 'status': 'cancelled', 'source': self._source('orders', order_id)},
                        'previous_status': current_status,
                        'cancelled_at': timestamp,
                        'confirmation_id': confirmation_id,
                        'refund': refund,
                    }
                except Exception:
                    if connection.in_transaction:
                        connection.execute('ROLLBACK')
                    raise
        except StoreError:
            raise
        except (sqlite3.Error, OSError):
            raise StoreError('Simulated shop database write failed') from None

    def create_support_ticket(self, customer_id, category, summary, priority,
                              order_id=None, escalate_to_human=False, created_at=None):
        """Atomically validate optional order ownership, deduplicate open tickets, and insert."""
        now = datetime.now(SHANGHAI_TZ)
        timestamp = created_at or now.isoformat(timespec='seconds')
        assigned_queue = 'human_specialist' if escalate_to_human else 'after_sales_support'
        try:
            with closing(self._connect_rw()) as connection:
                connection.execute('BEGIN IMMEDIATE')
                try:
                    if order_id is not None:
                        owned = connection.execute(
                            'SELECT order_id FROM orders WHERE order_id = ? AND customer_id = ?',
                            (order_id, customer_id),
                        ).fetchone()
                        if owned is None:
                            connection.execute('ROLLBACK')
                            return {'outcome': 'order_not_owned'}
                    duplicate = connection.execute(
                        "SELECT ticket_id, order_id, category, priority, status, escalate_to_human, "
                        "assigned_queue, summary, created_at "
                        "FROM tickets WHERE customer_id = ? AND category = ? AND COALESCE(order_id, '') = COALESCE(?, '') "
                        "AND summary = ? AND status = 'open' ORDER BY created_at DESC LIMIT 1",
                        (customer_id, category, order_id, summary),
                    ).fetchone()
                    if duplicate is not None:
                        connection.execute('ROLLBACK')
                        ticket = dict(duplicate)
                        ticket['escalate_to_human'] = bool(ticket['escalate_to_human'])
                        ticket['source'] = self._source('tickets', ticket['ticket_id'])
                        return {'outcome': 'deduplicated', 'ticket': ticket}
                    ticket_id = 'TK-' + now.strftime('%Y%m%d') + '-' + secrets.token_hex(3).upper()
                    connection.execute(
                        'INSERT INTO tickets(ticket_id, customer_id, order_id, category, priority, status, '
                        'escalate_to_human, assigned_queue, summary, created_at) '
                        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        (
                            ticket_id, customer_id, order_id, category, priority, 'open',
                            1 if escalate_to_human else 0, assigned_queue, summary, timestamp,
                        ),
                    )
                    connection.execute('COMMIT')
                    ticket = {
                        'ticket_id': ticket_id,
                        'order_id': order_id,
                        'category': category,
                        'priority': priority,
                        'status': 'open',
                        'escalate_to_human': bool(escalate_to_human),
                        'assigned_queue': assigned_queue,
                        'summary': summary,
                        'created_at': timestamp,
                        'source': self._source('tickets', ticket_id),
                    }
                    return {'outcome': 'created', 'ticket': ticket}
                except Exception:
                    if connection.in_transaction:
                        connection.execute('ROLLBACK')
                    raise
        except StoreError:
            raise
        except (sqlite3.Error, OSError):
            raise StoreError('Simulated shop database ticket write failed') from None


SCHEMA = """
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE customers (customer_id TEXT PRIMARY KEY);
CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    price_cents INTEGER NOT NULL CHECK(typeof(price_cents) = 'integer' AND price_cents >= 0),
    currency TEXT NOT NULL,
    capacity_ml INTEGER NOT NULL CHECK(capacity_ml > 0)
);
CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    product_id TEXT NOT NULL REFERENCES products(product_id),
    quantity INTEGER NOT NULL CHECK(typeof(quantity) = 'integer' AND quantity > 0),
    total_cents INTEGER NOT NULL CHECK(typeof(total_cents) = 'integer' AND total_cents >= 0),
    currency TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE INDEX orders_customer_id ON orders(customer_id, order_id);
CREATE TABLE order_cancellations (
    order_id TEXT PRIMARY KEY REFERENCES orders(order_id),
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    previous_status TEXT NOT NULL,
    reason TEXT NOT NULL,
    cancelled_at TEXT NOT NULL,
    confirmation_id TEXT NOT NULL
);
CREATE INDEX cancellations_customer_id ON order_cancellations(customer_id, order_id);
CREATE TABLE refunds (
    refund_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(order_id),
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    amount_cents INTEGER NOT NULL CHECK(typeof(amount_cents) = 'integer' AND amount_cents >= 0),
    currency TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    estimated_arrival TEXT NOT NULL,
    reviewer_note TEXT NOT NULL
);
CREATE INDEX refunds_customer_order ON refunds(customer_id, order_id);
CREATE TABLE tickets (
    ticket_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    order_id TEXT REFERENCES orders(order_id),
    category TEXT NOT NULL,
    priority TEXT NOT NULL,
    status TEXT NOT NULL,
    escalate_to_human INTEGER NOT NULL CHECK(escalate_to_human IN (0, 1)),
    assigned_queue TEXT NOT NULL,
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX tickets_customer_created ON tickets(customer_id, created_at);
"""


def initialize_database(database_path, seed_path):
    """Local setup only: atomically publish a fresh DB, never replace an existing one."""
    path = Path(database_path).resolve()
    if path.exists():
        store = SQLiteStore(path)
        return {'created': False, 'database_path': str(path), 'dataset_version': store.dataset_version}
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        seed_path = Path(seed_path)
        seed_bytes = seed_path.read_bytes()
        seed = json.loads(seed_bytes.decode('utf-8'))
        if seed.get('simulated') is not True or not seed.get('dataset_version'):
            raise StoreError('Seed must explicitly contain simulated data and a version')
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(temporary)) as connection:
            connection.execute('PRAGMA foreign_keys = ON')
            connection.executescript(SCHEMA)
            with connection:
                connection.executemany('INSERT INTO metadata(key, value) VALUES (?, ?)', [
                    ('schema_version', '1'), ('dataset_version', seed['dataset_version']),
                    ('simulated', 'true'), ('seed_file', seed_path.name),
                    ('seed_sha256', hashlib.sha256(seed_bytes).hexdigest()),
                ])
                connection.executemany('INSERT INTO customers(customer_id) VALUES (?)',
                                       [(identifier,) for identifier in seed['customers']])
                connection.executemany('INSERT INTO products(product_id, name, price_cents, currency, capacity_ml) '
                                       'VALUES (?, ?, ?, ?, ?)',
                    [(item['product_id'], item['name'], item['price_cents'], item['currency'], item['capacity_ml'])
                     for item in seed['products'].values()])
                # Historical order totals are copied exactly; never derived from product current prices.
                connection.executemany('INSERT INTO orders(order_id, customer_id, product_id, quantity, total_cents, currency, status) '
                                       'VALUES (?, ?, ?, ?, ?, ?, ?)',
                    [(item['order_id'], item['customer_id'], item['product_id'], item['quantity'],
                      item['total_cents'], item['currency'], item['status']) for item in seed['orders'].values()])
                if isinstance(seed.get('order_cancellations'), dict):
                    connection.executemany(
                        'INSERT INTO order_cancellations(order_id, customer_id, previous_status, reason, cancelled_at, confirmation_id) '
                        'VALUES (?, ?, ?, ?, ?, ?)',
                        [(item['order_id'], item['customer_id'], item['previous_status'],
                          item['reason'], item['cancelled_at'], item['confirmation_id'])
                         for item in seed['order_cancellations'].values()],
                    )
                if isinstance(seed.get('refunds'), dict):
                    connection.executemany(
                        'INSERT INTO refunds(refund_id, order_id, customer_id, amount_cents, currency, '
                        'status, reason, updated_at, estimated_arrival, reviewer_note) '
                        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        [(item['refund_id'], item['order_id'], item['customer_id'], item['amount_cents'],
                          item['currency'], item['status'], item['reason'], item['updated_at'],
                          item['estimated_arrival'], item['reviewer_note'])
                         for item in seed['refunds'].values()],
                    )
                if isinstance(seed.get('tickets'), dict):
                    connection.executemany(
                        'INSERT INTO tickets(ticket_id, customer_id, order_id, category, priority, status, '
                        'escalate_to_human, assigned_queue, summary, created_at) '
                        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        [(item['ticket_id'], item['customer_id'], item.get('order_id'), item['category'],
                          item['priority'], item['status'], 1 if item.get('escalate_to_human') else 0,
                          item['assigned_queue'], item['summary'], item['created_at'])
                         for item in seed['tickets'].values()],
                    )
        # Connection committed and closed; publication cannot overwrite an existing target.
        SQLiteStore(temporary)
        created = True
        try:
            os.link(temporary, path)
        except FileExistsError:
            created = False
        store = SQLiteStore(path)
        return {'created': created, 'database_path': str(path), 'dataset_version': store.dataset_version}
    except StoreError:
        raise
    except (sqlite3.Error, OSError, KeyError, TypeError, ValueError):
        raise StoreError('Unable to initialize simulated shop database; existing database was not replaced') from None
    finally:
        if temporary.exists():
            temporary.unlink()
