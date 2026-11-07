"""Real SQLite storage over synthetic shop data; SQL is never a model tool."""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import uuid


class StoreError(ValueError):
    """Safe storage failure without SQL, credentials or exception-body disclosure."""


class SQLiteStore:
    def __init__(self, database_path):
        self.path = Path(database_path).resolve()
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
        except (sqlite3.Error, OSError, KeyError) as error:
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
