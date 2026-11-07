"""Offline acceptance tests for the first customer-service milestone.

No model provider is called. Every temporary artifact stays under project/runtime.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from customer_service.context import ContextManager, validate_tool_pairs
from customer_service.harness import Harness
from customer_service.shop import Shop

ROOT = Path(__file__).resolve().parents[1]


def tool_use(call_id, name="get_order", **arguments):
    return {"type": "tool_use", "id": call_id, "name": name,
            "input": arguments or {"order_id": "A1001"}}


def tool_result(call_id, content="ok"):
    return {"type": "tool_result", "tool_use_id": call_id, "content": content}


def history50():
    return [{"role": "user" if i % 2 == 0 else "assistant",
             "content": f"订单 A1001 核对步骤 {i}"} for i in range(50)]


class ScriptedModel:
    """A deterministic protocol driver; this is not a real language model."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, tools, system):
        self.calls.append({"messages": deepcopy(messages),
                           "tools": deepcopy(tools), "system": system})
        return {"content": self.replies.pop(0),
                "stop_reason": "end_turn", "usage": {}}


class RuntimeCase(unittest.TestCase):
    def setUp(self):
        (ROOT / "runtime").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="acceptance_", dir=ROOT / "runtime")
        self.addCleanup(self.temp.cleanup)
        self.runtime = Path(self.temp.name)

    def context(self, customer="alice"):
        return ContextManager(self.runtime, customer)


class BusinessBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.shop = Shop("alice", ROOT / "data/shop.json")

    def test_own_order_and_tracking_are_available(self):
        for name in ("get_order", "get_tracking"):
            with self.subTest(name=name):
                answer = self.shop.call(name, {"order_id": "A1001"})
                self.assertTrue(answer["ok"])
                self.assertIn("A1001", json.dumps(answer["data"], ensure_ascii=False))

    def test_foreign_order_and_tracking_are_both_rejected(self):
        for name in ("get_order", "get_tracking"):
            with self.subTest(name=name):
                answer = self.shop.call(name, {"order_id": "B2001"})
                self.assertFalse(answer["ok"])
                self.assertIn("code", answer["error"])
                self.assertNotIn("data", answer)

    def test_customer_identity_cannot_be_overridden_in_tool_arguments(self):
        for name in ("get_order", "get_tracking"):
            with self.subTest(name=name):
                answer = self.shop.call(name, {"order_id": "B2001", "customer_id": "bob"})
                self.assertFalse(answer["ok"])

    def test_missing_unknown_and_bad_arguments_are_rejected(self):
        for name, arguments in [("get_order", {}), ("get_order", {"order_id": 42}),
                                ("get_order", {"order_id": "UNKNOWN"}),
                                ("refund", {"order_id": "A1001"})]:
            with self.subTest(name=name, arguments=arguments):
                answer = self.shop.call(name, arguments)
                self.assertFalse(answer["ok"])
                self.assertIn("code", answer["error"])

    def test_catalog_and_policy_return_evidence(self):
        for name, arguments in [("get_product", {"product_id": "P100"}),
                                ("get_policy", {"policy_id": "returns"})]:
            with self.subTest(name=name):
                answer = self.shop.call(name, arguments)
                self.assertTrue(answer["ok"])
                self.assertTrue(answer["data"])


class ArchiveBoundaryTests(RuntimeCase):
    def test_default_threshold_and_complete_archive(self):
        context = self.context()
        messages = history50()
        before = deepcopy(messages)
        result = context.prepare(messages, messages[48]["content"])
        self.assertEqual(30, len(result))
        self.assertEqual(1, len(context.archives))
        archive_path = next(iter(context.archives.values()))
        archived = [json.loads(line) for line in Path(archive_path).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(before, archived)
        validate_tool_pairs(result)

    def test_constraint_removed_from_context_can_be_queried_from_archive(self):
        context = self.context()
        messages = history50()
        messages[10]["content"] = "我只查物流，不要退款，也不要创建售后单。"
        result = context.prepare(messages, messages[48]["content"])
        self.assertNotIn("不要退款", json.dumps(result, ensure_ascii=False))
        archive_id = next(iter(context.archives))
        answer = context.query_archive(archive_id, "不要退款")
        self.assertTrue(answer["ok"])
        matches = answer["data"]["matches"]
        self.assertTrue(matches)
        self.assertIn("不要退款", json.dumps(matches, ensure_ascii=False))
        self.assertIn(10, [item["message_index"] for item in matches])

    def test_tail_cut_moves_back_to_keep_entire_tool_batch(self):
        context = self.context()
        messages = history50()
        messages[23] = {"role": "assistant", "content": [
            tool_use("call_23a"), tool_use("call_23b", "get_tracking")]}
        messages[24] = {"role": "user", "content": [
            tool_result("call_23a"), tool_result("call_23b")]}
        result = context.prepare(messages, messages[48]["content"])
        self.assertEqual(31, len(result))
        self.assertIn(messages[23], result)
        self.assertIn(messages[24], result)
        validate_tool_pairs(result)

    def test_active_request_is_restored_when_the_middle_is_archived(self):
        context = self.context()
        messages = history50()
        request = "当前请求是查询 A1001 的最新物流。"
        messages[10]["content"] = request
        result = context.prepare(messages, request)
        self.assertEqual({"role": "user", "content": request}, result[-1])
        self.assertEqual(31, len(result))

    def test_archive_id_is_not_authority_to_read_another_session(self):
        owner = self.context()
        messages = history50()
        owner.prepare(messages, messages[48]["content"])
        archive_id = next(iter(owner.archives))
        for other in (self.context(), self.context("bob")):
            with self.subTest(session=other.session_dir):
                self.assertFalse(other.query_archive(archive_id, "订单")["ok"])

    def test_arbitrary_paths_and_directory_traversal_are_rejected(self):
        context = self.context()
        for value in ("../shop", "../../data/shop.json", str(ROOT / "data/shop.json"),
                      "transcript_missing"):
            with self.subTest(value=value):
                self.assertFalse(context.query_archive(value, "订单")["ok"])


class MessageProtocolTests(unittest.TestCase):
    def test_multiple_complete_batches_are_valid(self):
        messages = [{"role": "user", "content": "查询订单"},
                    {"role": "assistant", "content": [tool_use("a"), tool_use("b")]},
                    {"role": "user", "content": [tool_result("b"), tool_result("a")]},
                    {"role": "assistant", "content": [tool_use("c")]},
                    {"role": "user", "content": [tool_result("c")]}]
        self.assertIsNone(validate_tool_pairs(messages))

    def test_orphan_mismatched_missing_and_duplicate_results_are_invalid(self):
        cases = [
            [{"role": "user", "content": [tool_result("orphan")]}],
            [{"role": "assistant", "content": [tool_use("a")]}],
            [{"role": "assistant", "content": [tool_use("a")]},
             {"role": "user", "content": [tool_result("b")]}],
            [{"role": "assistant", "content": [tool_use("a"), tool_use("b")]},
             {"role": "user", "content": [tool_result("a")]}],
            [{"role": "assistant", "content": [tool_use("a"), tool_use("a")]},
             {"role": "user", "content": [tool_result("a"), tool_result("a")]}],
            [{"role": "assistant", "content": [tool_use("a")]},
             {"role": "user", "content": [tool_result("a"), tool_result("a")]}],
        ]
        for index, messages in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                validate_tool_pairs(messages)


class HarnessTests(RuntimeCase):
    def shop(self):
        return Shop("alice", ROOT / "data/shop.json")

    def test_full_assistant_content_and_all_batch_results_are_retained(self):
        first = [{"type": "text", "text": "先核对本人订单，并检查另一个订单的查询权限。"},
                 tool_use("order_a", "get_order", order_id="A1001"),
                 tool_use("tracking_b", "get_tracking", order_id="B2001")]
        final = "本人订单已核对，另一个客户的物流查询被拒绝。"
        model = ScriptedModel([first, [{"type": "text", "text": final}]])
        harness = Harness(self.shop(), self.context(), model)
        answer = harness.turn("查询两个订单。")
        self.assertEqual(final, answer)
        self.assertEqual(first, harness.messages[1]["content"])
        results = harness.messages[2]["content"]
        self.assertEqual({"order_a", "tracking_b"}, {item["tool_use_id"] for item in results})
        parsed = {item["tool_use_id"]: json.loads(item["content"]) for item in results}
        self.assertTrue(parsed["order_a"]["ok"])
        self.assertFalse(parsed["tracking_b"]["ok"])
        self.assertEqual(2, len(model.calls))
        validate_tool_pairs(model.calls[1]["messages"])
        validate_tool_pairs(harness.messages)

    def test_step_limit_leaves_complete_pairs_and_explicit_incomplete_result(self):
        model = ScriptedModel([[tool_use("loop_1")], [tool_use("loop_2")]])
        harness = Harness(self.shop(), self.context(), model, max_steps=2)
        answer = harness.turn("持续核对订单。")
        self.assertIn("上限", answer)
        self.assertIn("未完成", answer)
        self.assertEqual(2, len(model.calls))
        validate_tool_pairs(harness.messages)


if __name__ == "__main__":
    unittest.main(verbosity=2)
