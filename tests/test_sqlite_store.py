"""Offline SQLite acceptance: authoritative prices, historical totals and ownership."""
from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest

from customer_service.context import ContextManager, validate_tool_pairs
from customer_service.harness import Harness
from customer_service.shop import Shop, TOOL_DEFINITIONS
from customer_service.storage import initialize_database, SQLiteStore, StoreError

ROOT = Path(__file__).resolve().parents[1]
SQL_VERSION = "demo-sqlite-2026-11-03"
ORDER_FIELDS = {"order_id", "product_id", "quantity", "total_cents", "currency", "status", "source"}
LIST_FIELDS = {"order_id", "status", "total_cents", "currency", "source"}


class ScriptedModel:
    mode = "scripted-offline"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, tools, system):
        self.calls.append({"messages": deepcopy(messages), "tools": deepcopy(tools), "system": system})
        content = deepcopy(self.replies.pop(0))
        return {"content": content, "stop_reason": "tool_use" if any(b.get("type") == "tool_use" for b in content) else "end_turn", "usage": {}}


class SQLiteStoreTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "runtime").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="sqlite_acceptance_", dir=ROOT / "runtime")
        self.addCleanup(self.temporary.cleanup)
        self.runtime = Path(self.temporary.name)
        self.json_path = self.runtime / "shop.json"
        self.seed_path = self.runtime / "shop-sqlite-seed.json"
        shutil.copyfile(ROOT / "data/shop.json", self.json_path)
        shutil.copyfile(ROOT / "data/shop-sqlite-seed.json", self.seed_path)
        self.legacy = json.loads(self.json_path.read_text(encoding="utf-8"))
        self.seed = json.loads(self.seed_path.read_text(encoding="utf-8"))
        self.database_path = self.runtime / "customer-store.sqlite3"
        self.initialization = initialize_database(self.database_path, self.seed_path)
        self.store = SQLiteStore(self.database_path)
        self.shop = Shop("alice", self.json_path, database_path=self.database_path)

    def execute_sql(self, statement, parameters=(), database_path=None):
        with closing(sqlite3.connect(database_path or self.database_path)) as connection:
            connection.execute(statement, parameters)
            connection.commit()

    def variant(self, seed, name="variant"):
        seed_path = self.runtime / (name + ".json")
        seed_path.write_text(json.dumps(seed, ensure_ascii=False, indent=2), encoding="utf-8")
        database_path = self.runtime / (name + ".sqlite3")
        initialize_database(database_path, seed_path)
        return database_path, Shop("alice", self.json_path, database_path=database_path)

    def test_database_is_real_persistent_sqlite_and_reopens_with_metadata(self):
        self.assertTrue(self.initialization["created"])
        self.assertEqual(self.database_path.resolve(), Path(self.initialization["database_path"]).resolve())
        self.assertEqual(SQL_VERSION, self.initialization["dataset_version"])
        self.assertTrue(self.database_path.read_bytes().startswith(b"SQLite format 3\x00"))
        with closing(sqlite3.connect(self.database_path)) as connection:
            self.assertEqual(19900, connection.execute("SELECT price_cents FROM products WHERE product_id = ?", ("P100",)).fetchone()[0])
        reopened = SQLiteStore(self.database_path)
        self.assertEqual(SQL_VERSION, reopened.dataset_version)
        self.assertTrue(reopened.has_customer("alice"))
        self.assertTrue(reopened.has_customer("bob"))
        self.assertFalse(reopened.has_customer("unknown"))
        self.assertEqual(16900, reopened.get_order("A1001", "alice")["total_cents"])
        self.assertEqual(ORDER_FIELDS, set(reopened.get_order("A1001", "alice")))
        self.assertEqual(self.database_path.name + "#orders.A1001", reopened.get_order("A1001", "alice")["source"])

    def test_current_product_price_changes_without_repricing_historical_orders_or_caching_json(self):
        original_json = self.json_path.read_bytes()
        product = self.shop.call("get_product", {"product_id": "P100"})
        order = self.shop.call("get_order", {"order_id": "A1001"})
        self.assertEqual(19900, product["data"]["price_cents"])
        self.assertEqual(16900, order["data"]["total_cents"])
        self.assertEqual(SQL_VERSION, product["dataset_version"])
        self.execute_sql("UPDATE products SET price_cents = ? WHERE product_id = ?", (24900, "P100"))
        self.assertEqual(24900, self.shop.call("get_product", {"product_id": "P100"})["data"]["price_cents"])
        self.assertEqual(16900, self.shop.call("get_order", {"order_id": "A1001"})["data"]["total_cents"])
        self.assertEqual(33800, SQLiteStore(self.database_path).get_order("B2001", "bob")["total_cents"])
        self.assertEqual(16900, Shop("alice", self.json_path).call("get_product", {"product_id": "P100"})["data"]["price_cents"])
        self.assertEqual(original_json, self.json_path.read_bytes())

    def test_store_and_shop_filter_orders_by_bound_customer(self):
        self.assertIsNone(self.store.get_order("B2001", "alice"))
        self.assertIsNone(self.store.get_order("A1001", "bob"))
        self.assertEqual(["A1001"], [row["order_id"] for row in self.store.list_orders("alice")])
        self.assertEqual(["B2001"], [row["order_id"] for row in self.store.list_orders("bob")])
        self.assertTrue(all(set(row) == LIST_FIELDS for row in self.store.list_orders("alice")))
        self.assertFalse(self.shop.call("get_order", {"order_id": "B2001"})["ok"])
        self.assertFalse(self.shop.call("get_tracking", {"order_id": "B2001"})["ok"])
        own = self.shop.call("list_my_orders", {})
        self.assertEqual(["A1001"], [row["order_id"] for row in own["data"]["orders"]])
        self.assertNotIn("customer_id", json.dumps(own, ensure_ascii=False))
        self.assertNotIn("B2001", json.dumps(own, ensure_ascii=False))

    def test_sql_injection_strings_are_bound_as_values_and_cannot_expand_access(self):
        injection = "alice' OR '1'='1"
        self.assertFalse(self.store.has_customer(injection))
        self.assertEqual([], self.store.list_orders(injection))
        self.assertIsNone(self.store.get_order("B2001", injection))
        self.assertIsNone(self.store.get_order("A1001' OR 1=1 --", "alice"))
        self.assertIsNone(self.store.get_product("P100'; DROP TABLE orders;--"))
        for name, args in [("get_order", {"order_id": "A1001' OR 1=1 --"}),
                           ("get_product", {"product_id": "P100' OR 1=1 --"})]:
            with self.subTest(tool=name):
                answer = self.shop.call(name, args)
                self.assertFalse(answer["ok"])
                self.assertNotIn("data", answer)
        self.assertEqual(["A1001"], [row["order_id"] for row in SQLiteStore(self.database_path).list_orders("alice")])

    def test_empty_own_order_list_is_a_success_on_a_healthy_database(self):
        self.execute_sql("DELETE FROM orders WHERE customer_id = ?", ("alice",))
        self.assertTrue(SQLiteStore(self.database_path).has_customer("alice"))
        self.assertEqual([], self.store.list_orders("alice"))
        answer = self.shop.call("list_my_orders", {})
        self.assertTrue(answer["ok"])
        self.assertEqual([], answer["data"]["orders"])
        self.assertEqual(SQL_VERSION, answer["dataset_version"])
        self.assertNotIn("error", answer)

    def test_missing_or_corrupt_database_is_failure_not_json_fallback_or_empty_list(self):
        for mode in ("missing", "corrupt"):
            with self.subTest(mode=mode):
                database_path = self.runtime / (mode + ".sqlite3")
                initialize_database(database_path, self.seed_path)
                shop = Shop("alice", self.json_path, database_path=database_path)
                if mode == "missing":
                    database_path.unlink()
                else:
                    database_path.write_bytes(b"PRIVATE_INVALID_DATABASE_BYTES")
                for name, args in [("get_product", {"product_id": "P100"}),
                                   ("get_order", {"order_id": "A1001"}),
                                   ("get_tracking", {"order_id": "A1001"}), ("list_my_orders", {})]:
                    answer = shop.call(name, args)
                    self.assertFalse(answer["ok"])
                    self.assertEqual("DATA_STORE_UNAVAILABLE", answer["error"]["code"])
                    self.assertNotIn("data", answer)
                    self.assertNotIn("PRIVATE_INVALID_DATABASE_BYTES", json.dumps(answer, ensure_ascii=False))
                if mode == "missing":
                    self.assertFalse(database_path.exists())
                self.assertTrue(shop.call("get_policy", {"policy_id": "returns"})["ok"])

    def test_sql_order_list_is_sorted_and_includes_db_only_orders_missing_from_json(self):
        seed = deepcopy(self.seed)
        first, last = deepcopy(seed["orders"]["A1001"]), deepcopy(seed["orders"]["A1001"])
        first.update(order_id="A1000", total_cents=12900)
        last.update(order_id="A1003", total_cents=17900)
        seed["orders"] = {"A1003": last, "B2001": seed["orders"]["B2001"],
                          "A1001": seed["orders"]["A1001"], "A1000": first}
        database_path, shop = self.variant(seed, "sorted-orders")
        orders = shop.call("list_my_orders", {})["data"]["orders"]
        self.assertEqual(["A1000", "A1001", "A1003"], [row["order_id"] for row in orders])
        self.assertEqual([12900, 16900, 17900], [row["total_cents"] for row in orders])
        self.assertTrue(all(set(row) == LIST_FIELDS for row in orders))
        self.assertEqual(database_path.name + "#orders.A1000", orders[0]["source"])
        self.assertEqual(12900, shop.call("get_order", {"order_id": "A1000"})["data"]["total_cents"])

    def test_read_only_business_queries_do_not_change_database_seed_or_json_bytes(self):
        before = {path: path.read_bytes() for path in (self.database_path, self.seed_path, self.json_path)}
        for name, args in [("get_product", {"product_id": "P100"}), ("get_order", {"order_id": "A1001"}),
                           ("get_tracking", {"order_id": "A1001"}), ("list_my_orders", {}),
                           ("get_policy", {"policy_id": "returns"})]:
            self.assertTrue(self.shop.call(name, args)["ok"])
        self.store.get_product("P100")
        self.store.get_order("A1001", "alice")
        self.store.list_orders("bob")
        self.store.has_customer("alice")
        for path, original in before.items():
            self.assertEqual(original, path.read_bytes())

    def test_initialization_is_idempotent_and_preserves_database_edits_without_reading_seed(self):
        self.execute_sql("UPDATE products SET price_cents = ? WHERE product_id = ?", (21900, "P100"))
        before = self.database_path.read_bytes()
        self.seed_path.unlink()
        again = initialize_database(self.database_path, self.seed_path)
        self.assertFalse(again["created"])
        self.assertEqual(SQL_VERSION, again["dataset_version"])
        self.assertEqual(21900, SQLiteStore(self.database_path).get_product("P100")["price_cents"])
        self.assertEqual(before, self.database_path.read_bytes())

    def test_unknown_existing_files_are_rejected_without_overwrite_or_reseed(self):
        plain = self.runtime / "unknown-private.bin"
        plain.write_bytes(b"DO_NOT_REPLACE_PRIVATE_FILE")
        foreign = self.runtime / "foreign.sqlite3"
        with closing(sqlite3.connect(foreign)) as connection:
            connection.execute("CREATE TABLE foreign_table (value TEXT)")
            connection.execute("INSERT INTO foreign_table(value) VALUES (?)", ("keep me",))
            connection.commit()
        for path in (plain, foreign):
            original = path.read_bytes()
            with self.subTest(path=path.name), self.assertRaises(StoreError) as caught:
                initialize_database(path, self.seed_path)
            self.assertNotIn(str(path), str(caught.exception))
            self.assertNotIn("no such table", str(caught.exception))
            self.assertEqual(original, path.read_bytes())
        with closing(sqlite3.connect(foreign)) as connection:
            self.assertEqual([("keep me",)], connection.execute("SELECT value FROM foreign_table").fetchall())

    def test_missing_database_and_directory_path_errors_are_controlled_without_file_creation(self):
        missing = self.runtime / "PRIVATE_MISSING_DATABASE.sqlite3"
        directory = self.runtime / "database-directory"
        directory.mkdir()
        for path in (missing, directory):
            with self.subTest(path=path.name), self.assertRaises(StoreError) as caught:
                SQLiteStore(path)
            self.assertIsInstance(caught.exception, ValueError)
            self.assertNotIn(str(path), str(caught.exception))
            self.assertNotIn("unable to open database", str(caught.exception))
        self.assertFalse(missing.exists())
        with self.assertRaises(StoreError):
            initialize_database(directory, self.seed_path)
        with self.assertRaises(StoreError):
            Shop("alice", self.json_path, database_path=missing)
        self.assertFalse(missing.exists())

    def test_invalid_metadata_is_rejected_and_existing_database_is_not_repaired_from_seed(self):
        self.execute_sql("UPDATE metadata SET value = ? WHERE key = ?", ("", "dataset_version"))
        before = self.database_path.read_bytes()
        with self.assertRaises(StoreError):
            SQLiteStore(self.database_path)
        with self.assertRaises(StoreError):
            initialize_database(self.database_path, self.seed_path)
        self.assertEqual(before, self.database_path.read_bytes())

    def test_invalid_seed_never_publishes_a_partial_database_or_exposes_raw_body(self):
        cases = [b"PRIVATE_BAD_SEED_CONTENT", json.dumps({**self.seed, "simulated": False}).encode("utf-8")]
        negative = deepcopy(self.seed)
        negative["products"]["P100"]["price_cents"] = -1
        cases.append(json.dumps(negative, ensure_ascii=False).encode("utf-8"))
        for index, content in enumerate(cases):
            seed_path = self.runtime / (f"invalid-seed-{index}.json")
            destination = self.runtime / (f"invalid-seed-{index}.sqlite3")
            seed_path.write_bytes(content)
            with self.subTest(case=index), self.assertRaises(StoreError) as caught:
                initialize_database(destination, seed_path)
            self.assertNotIn("PRIVATE_BAD_SEED_CONTENT", str(caught.exception))
            self.assertFalse(destination.exists())
            self.assertEqual(content, seed_path.read_bytes())

    def test_policy_and_tracking_keep_json_provenance_while_sql_tools_use_database_source(self):
        product = self.shop.call("get_product", {"product_id": "P100"})
        order = self.shop.call("get_order", {"order_id": "A1001"})
        listing = self.shop.call("list_my_orders", {})
        self.assertEqual(self.database_path.name + "#products.P100", product["data"]["source"])
        self.assertEqual(self.database_path.name + "#orders.A1001", order["data"]["source"])
        self.assertEqual(self.database_path.name + "#orders.A1001", listing["data"]["orders"][0]["source"])
        self.assertTrue(all(item["dataset_version"] == SQL_VERSION for item in (product, order, listing)))
        policy = self.shop.call("get_policy", {"policy_id": "returns"})
        tracking = self.shop.call("get_tracking", {"order_id": "A1001"})
        self.assertTrue(policy["ok"])
        self.assertTrue(tracking["ok"])
        self.assertEqual(self.legacy["policies"]["returns"]["source"], policy["data"]["source"])
        self.assertEqual(self.legacy["orders"]["A1001"]["source"] + ".tracking", tracking["data"]["source"])
        self.assertEqual(self.legacy["dataset_version"], policy["dataset_version"])
        self.assertEqual(self.legacy["dataset_version"], tracking["dataset_version"])
        self.assertEqual(self.legacy["orders"]["A1001"]["tracking"]["events"], tracking["data"]["events"])

    def test_tracking_checks_current_database_ownership_before_returning_json_tracking(self):
        self.assertTrue(self.shop.call("get_tracking", {"order_id": "A1001"})["ok"])
        self.execute_sql("UPDATE orders SET customer_id = ? WHERE order_id = ?", ("bob", "A1001"))
        self.assertEqual("alice", self.legacy["orders"]["A1001"]["customer_id"])
        denied = self.shop.call("get_tracking", {"order_id": "A1001"})
        self.assertFalse(denied["ok"])
        self.assertEqual("ACCESS_DENIED_OR_NOT_FOUND", denied["error"]["code"])
        self.assertNotIn("data", denied)

    def test_sql_owned_order_without_json_tracking_returns_not_found(self):
        seed = deepcopy(self.seed)
        extra = deepcopy(seed["orders"]["A1001"])
        extra.update(order_id="A1000", total_cents=12900)
        seed["orders"]["A1000"] = extra
        _, shop = self.variant(seed, "db-only-order")
        self.assertTrue(shop.call("get_order", {"order_id": "A1000"})["ok"])
        self.assertNotIn("A1000", self.legacy["orders"])
        tracking = shop.call("get_tracking", {"order_id": "A1000"})
        self.assertFalse(tracking["ok"])
        self.assertEqual("NOT_FOUND", tracking["error"]["code"])
        self.assertNotIn("data", tracking)

    def test_database_backend_does_not_expand_model_parameters_or_override_identity(self):
        definition = next(tool for tool in TOOL_DEFINITIONS if tool["name"] == "list_my_orders")
        self.assertEqual({}, definition["input_schema"]["properties"])
        self.assertEqual([], definition["input_schema"]["required"])
        self.assertFalse(definition["input_schema"]["additionalProperties"])
        for name, args in [("list_my_orders", {"customer_id": "bob"}),
                           ("list_my_orders", {"status": "delivered"}),
                           ("get_order", {"order_id": "B2001", "customer_id": "bob"}),
                           ("get_product", {"product_id": "P100", "sql": "SELECT * FROM orders"})]:
            with self.subTest(tool=name):
                denied = self.shop.call(name, args)
                self.assertFalse(denied["ok"])
                self.assertEqual("INVALID_ARGUMENTS", denied["error"]["code"])
                self.assertNotIn("data", denied)
        self.assertEqual("UNKNOWN_TOOL", self.shop.call("sql", {"query": "SELECT * FROM orders"})["error"]["code"])
        self.assertEqual("alice", self.shop.customer_id)

    def test_harness_batch_keeps_current_product_price_and_historical_order_total_separate(self):
        first = [{"type": "text", "text": "分别核对商品现价、历史订单和本人订单列表。"},
                 {"type": "tool_use", "id": "sql_price", "name": "get_product", "input": {"product_id": "P100"}},
                 {"type": "tool_use", "id": "sql_order", "name": "get_order", "input": {"order_id": "A1001"}},
                 {"type": "tool_use", "id": "sql_list", "name": "list_my_orders", "input": {}}]
        model = ScriptedModel([first, [{"type": "text", "text": "商品现价199元，历史订单实付169元。"}]])
        context = ContextManager(self.runtime / "agent-runs", "alice")
        harness = Harness(self.shop, context, model)
        answer = harness.turn("核对现价和我以前的订单实付。")
        self.assertIn("199", answer)
        self.assertIn("169", answer)
        self.assertEqual(2, len(model.calls))
        results = model.calls[1]["messages"][-1]["content"]
        self.assertEqual({"sql_price", "sql_order", "sql_list"}, {item["tool_use_id"] for item in results})
        received = {item["tool_use_id"]: json.loads(item["content"]) for item in results}
        self.assertTrue(all(item["ok"] for item in received.values()))
        self.assertEqual(19900, received["sql_price"]["data"]["price_cents"])
        self.assertEqual(16900, received["sql_order"]["data"]["total_cents"])
        self.assertEqual(16900, received["sql_list"]["data"]["orders"][0]["total_cents"])
        self.assertNotIn("B2001", json.dumps(received, ensure_ascii=False))
        validate_tool_pairs(model.calls[1]["messages"])
        validate_tool_pairs(harness.messages)


if __name__ == "__main__":
    unittest.main(verbosity=2)
