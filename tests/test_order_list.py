"""Offline acceptance of the customer-bound, zero-argument order list tool."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from customer_service.context import ContextManager, validate_tool_pairs
from customer_service.harness import Harness
from customer_service.shop import Shop, TOOL_DEFINITIONS

ROOT = Path(__file__).resolve().parents[1]
ORDER_FIELDS = {"order_id", "status", "total_cents", "currency", "source"}


class ScriptedModel:
    mode = "scripted-offline"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, tools, system):
        self.calls.append({"messages": deepcopy(messages), "tools": deepcopy(tools), "system": system})
        content = deepcopy(self.replies.pop(0))
        return {"content": content, "stop_reason": "tool_use" if any(b.get("type") == "tool_use" for b in content) else "end_turn", "usage": {}}


class OrderListTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "runtime").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="order_list_acceptance_", dir=ROOT / "runtime")
        self.addCleanup(self.temporary.cleanup)
        self.runtime = Path(self.temporary.name)
        self.data_path = ROOT / "data/shop.json"
        self.fixture = json.loads(self.data_path.read_text(encoding="utf-8"))
        self.shop = Shop("alice", self.data_path)

    def custom_shop(self, fixture, customer="alice"):
        path = self.runtime / "shop-fixture.json"
        path.write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8")
        return Shop(customer, path), path

    def harness(self, model):
        return Harness(self.shop, ContextManager(self.runtime, "alice"), model)

    def test_alice_gets_only_own_order_and_public_summary_fields(self):
        answer = self.shop.list_my_orders()
        self.assertTrue(answer["ok"])
        self.assertTrue(answer["simulated"])
        self.assertEqual(self.fixture["dataset_version"], answer["dataset_version"])
        orders = answer["data"]["orders"]
        self.assertEqual(["A1001"], [order["order_id"] for order in orders])
        self.assertEqual(ORDER_FIELDS, set(orders[0]))
        self.assertEqual(16900, orders[0]["total_cents"])
        self.assertEqual("CNY", orders[0]["currency"])
        self.assertEqual("shipped", orders[0]["status"])
        self.assertEqual("shop.json#orders.A1001", orders[0]["source"])
        serialized = json.dumps(answer, ensure_ascii=False)
        for foreign_or_private in ("B2001", "DEMO-B2001", "customer_id", "tracking", "product_id"):
            self.assertNotIn(foreign_or_private, serialized)

    def test_another_bound_customer_gets_that_customers_own_order(self):
        bob = Shop("bob", self.data_path)
        answer = bob.list_my_orders()
        self.assertTrue(answer["ok"])
        self.assertEqual(["B2001"], [order["order_id"] for order in answer["data"]["orders"]])
        self.assertNotIn("A1001", json.dumps(answer, ensure_ascii=False))
        self.assertTrue(all(set(order) == ORDER_FIELDS for order in answer["data"]["orders"]))

    def test_multiple_own_orders_are_sorted_without_foreign_or_internal_fields(self):
        fixture = deepcopy(self.fixture)
        first, last = deepcopy(fixture["orders"]["A1001"]), deepcopy(fixture["orders"]["A1001"])
        first.update(order_id="A1000", status="paid", source="shop.json#orders.A1000", internal_note="PRIVATE_INTERNAL_NOTE")
        last.update(order_id="A1003", status="packing", source="shop.json#orders.A1003", internal_note="PRIVATE_INTERNAL_NOTE")
        fixture["orders"] = {"B2001": fixture["orders"]["B2001"], "A1003": last,
                             "A1001": fixture["orders"]["A1001"], "A1000": first}
        shop, _ = self.custom_shop(fixture)
        orders = shop.list_my_orders()["data"]["orders"]
        self.assertEqual(["A1000", "A1001", "A1003"], [row["order_id"] for row in orders])
        self.assertTrue(all(set(row) == ORDER_FIELDS for row in orders))
        self.assertNotIn("PRIVATE_INTERNAL_NOTE", json.dumps(orders, ensure_ascii=False))
        self.assertNotIn("B2001", json.dumps(orders, ensure_ascii=False))

    def test_valid_customer_with_no_orders_receives_successful_empty_list(self):
        fixture = deepcopy(self.fixture)
        fixture["orders"] = {"B2001": fixture["orders"]["B2001"]}
        shop, _ = self.custom_shop(fixture)
        answer = shop.list_my_orders()
        self.assertTrue(answer["ok"])
        self.assertEqual([], answer["data"]["orders"])
        self.assertTrue(answer["simulated"])
        self.assertEqual(fixture["dataset_version"], answer["dataset_version"])
        self.assertNotIn("error", answer)

    def test_listing_does_not_mutate_loaded_data_or_fixture_file(self):
        before_data = deepcopy(self.shop.data)
        before_file = self.data_path.read_bytes()
        self.shop.list_my_orders()
        self.shop.call("list_my_orders", {})
        self.assertEqual(before_data, self.shop.data)
        self.assertEqual(before_file, self.data_path.read_bytes())

    def test_returned_order_dicts_and_list_are_independent_from_later_calls(self):
        answer = self.shop.list_my_orders()
        expected = deepcopy(answer)
        answer["data"]["orders"][0]["status"] = "caller_modified"
        answer["data"]["orders"].append({"order_id": "FAKE_FOREIGN"})
        answer["data"]["orders"].clear()
        self.assertEqual(expected, self.shop.list_my_orders())
        self.assertEqual("shipped", self.shop.data["orders"]["A1001"]["status"])
        self.assertNotIn("FAKE_FOREIGN", self.shop.data["orders"])

    def test_zero_argument_router_matches_direct_business_result(self):
        self.assertEqual(self.shop.list_my_orders(), self.shop.call("list_my_orders", {}))
        self.assertEqual("alice", self.shop.customer_id)

    def test_extra_or_non_object_parameters_are_rejected_without_identity_override(self):
        invalid = [{"customer_id": "bob"}, {"customer_id": "alice"}, {"order_id": "B2001"},
                   {"status": "delivered"}, {"filter": {}}, {"customer_id": "bob", "order_id": "B2001"},
                   None, [], "{}", 0, False]
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                answer = self.shop.call("list_my_orders", arguments)
                self.assertFalse(answer["ok"])
                self.assertEqual("INVALID_ARGUMENTS", answer["error"]["code"])
                self.assertNotIn("data", answer)
                self.assertNotIn("DEMO-B2001", json.dumps(answer, ensure_ascii=False))
        self.assertEqual("alice", self.shop.customer_id)
        self.assertEqual(["A1001"], [row["order_id"] for row in self.shop.list_my_orders()["data"]["orders"]])

    def test_tool_schema_is_zero_argument_and_existing_four_schemas_remain_compatible(self):
        definitions = {tool["name"]: tool for tool in TOOL_DEFINITIONS}
        schema = definitions["list_my_orders"]["input_schema"]
        self.assertEqual("object", schema["type"])
        self.assertEqual({}, schema["properties"])
        self.assertEqual([], schema["required"])
        self.assertFalse(schema["additionalProperties"])
        old_arguments = {"get_product": "product_id", "get_policy": "policy_id",
                         "get_order": "order_id", "get_tracking": "order_id"}
        for name, argument in old_arguments.items():
            with self.subTest(tool=name):
                existing = definitions[name]["input_schema"]
                self.assertEqual({argument: {"type": "string"}}, existing["properties"])
                self.assertEqual([argument], existing["required"])
                self.assertFalse(existing["additionalProperties"])

    def test_harness_lists_current_customer_orders_in_the_next_paired_tool_result(self):
        first = [{"type": "text", "text": "先查询当前客户的订单列表。"},
                 {"type": "tool_use", "id": "orders_listing_1", "name": "list_my_orders", "input": {}}]
        final = "当前客户的订单为 A1001。"
        model = ScriptedModel([first, [{"type": "text", "text": final}]])
        harness = self.harness(model)
        answer = harness.turn("请列出我的订单。")
        self.assertEqual(final, answer)
        self.assertIn("list_my_orders", {tool["name"] for tool in harness.tools})
        self.assertEqual(2, len(model.calls))
        self.assertEqual(first, model.calls[1]["messages"][-2]["content"])
        results = model.calls[1]["messages"][-1]["content"]
        self.assertEqual(1, len(results))
        self.assertEqual("orders_listing_1", results[0]["tool_use_id"])
        received = json.loads(results[0]["content"])
        self.assertTrue(received["ok"])
        self.assertEqual(["A1001"], [row["order_id"] for row in received["data"]["orders"]])
        self.assertNotIn("B2001", json.dumps(received, ensure_ascii=False))
        validate_tool_pairs(model.calls[1]["messages"])
        validate_tool_pairs(harness.messages)

    def test_harness_rejects_identity_injection_and_returns_no_foreign_order_data(self):
        first = [{"type": "tool_use", "id": "malicious_order_listing", "name": "list_my_orders",
                  "input": {"customer_id": "bob"}}]
        model = ScriptedModel([first, [{"type": "text", "text": "参数被拒绝，不能更换客户身份。"}]])
        harness = self.harness(model)
        harness.turn("列出订单。")
        received = json.loads(model.calls[1]["messages"][-1]["content"][0]["content"])
        self.assertFalse(received["ok"])
        self.assertEqual("INVALID_ARGUMENTS", received["error"]["code"])
        self.assertNotIn("data", received)
        self.assertNotIn("B2001", json.dumps(received, ensure_ascii=False))
        self.assertNotIn("DEMO-B2001", json.dumps(received, ensure_ascii=False))
        self.assertEqual("alice", harness.shop.customer_id)
        self.assertEqual("malicious_order_listing", model.calls[1]["messages"][-1]["content"][0]["tool_use_id"])
        validate_tool_pairs(harness.messages)


if __name__ == "__main__":
    unittest.main(verbosity=2)
