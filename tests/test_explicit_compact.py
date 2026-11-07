"""Offline acceptance of explicit compaction after a complete tool batch."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from customer_service.context import COMPACT_TOOL, ContextManager, validate_tool_pairs
from customer_service.harness import Harness
from customer_service.shop import Shop
from customer_service.summary import SummaryClient

ROOT = Path(__file__).resolve().parents[1]
REQUEST = '只读查询 A1001 物流；不要退款，也不要创建售后单。'


def call(identifier, name="compact", **arguments):
    return {"type": "tool_use", "id": identifier, "name": name, "input": arguments}


def text(value):
    return {"type": "text", "text": value}


def summary_reply(value="客户只查物流，不要退款，不要创建售后单。"):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=value)], stop_reason="end_turn",
                           usage=SimpleNamespace(model_dump=lambda **kwargs: {"input_tokens": 100, "output_tokens": 20}))


class SummarySDK:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        response = self.replies.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class MainModel:
    mode = "scripted-offline"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, tools, system):
        self.calls.append({"messages": deepcopy(messages), "tools": deepcopy(tools), "system": system})
        content = deepcopy(self.replies.pop(0))
        return {"content": content, "stop_reason": "tool_use" if any(b.get("type") == "tool_use" for b in content) else "end_turn", "usage": {}}


class ExplicitCompactTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "runtime").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="explicit_compact_acceptance_", dir=ROOT / "runtime")
        self.addCleanup(self.temporary.cleanup)
        self.runtime = Path(self.temporary.name)

    def build(self, main_replies, summary_replies=None, customer="alice"):
        raw = None if summary_replies is None else SummarySDK(summary_replies)
        client = None if raw is None else SummaryClient(raw, "offline-summary")
        context = ContextManager(self.runtime, customer or "guest", summary_client=client)
        model = MainModel(main_replies)
        harness = Harness(Shop(customer, ROOT / "data/shop.json"), context, model)
        return harness, context, model, raw

    def summary_input(self, raw):
        return json.loads(raw.calls[0]["messages"][0]["content"])

    def archived(self, context):
        return [json.loads(line) for line in next(iter(context.archives.values())).read_text(encoding="utf-8").splitlines()]

    def test_schema_and_acceptance_are_zero_argument_and_do_not_compact_immediately(self):
        harness, context, _, raw = self.build([], [summary_reply()])
        schema = COMPACT_TOOL["input_schema"]
        self.assertEqual({}, schema["properties"])
        self.assertEqual([], schema["required"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(1, sum(tool["name"] == "compact" for tool in harness.tools))
        accepted = context.accept_compact({})
        self.assertTrue(accepted["ok"])
        self.assertEqual("scheduled_after_tool_batch", accepted["data"]["status"])
        self.assertFalse(raw.calls)
        self.assertFalse(context.summary_events)
        self.assertFalse(context.archives)
        self.assertEqual(0, context.summary_attempts_this_turn)
        for arguments in (None, [], {"reason": "shorten"}, {"customer_id": "bob"}):
            with self.subTest(arguments=arguments):
                rejected = context.accept_compact(arguments)
                self.assertFalse(rejected["ok"])
                self.assertEqual("INVALID_ARGUMENTS", rejected["error"]["code"])
        self.assertFalse(raw.calls)

    def test_full_batch_including_post_compact_tracking_enters_summary_and_archive_below_limit(self):
        first = [text("先完成整批工具。"), call("compact_first"),
                 call("tracking_after_compact", "get_tracking", order_id="A1001")]
        harness, context, model, raw = self.build([first, [text("只读查询已完成。")]], [summary_reply()])
        answer = harness.turn(REQUEST)
        self.assertEqual("只读查询已完成。", answer)
        self.assertEqual(1, len(raw.calls))
        supplied = self.summary_input(raw)
        self.assertLess(context.snipper.estimate_chars(supplied), context.snipper.CONTEXT_CHAR_LIMIT)
        self.assertEqual(first, supplied[-2]["content"])
        result_blocks = supplied[-1]["content"]
        self.assertEqual(["compact_first", "tracking_after_compact"], [b["tool_use_id"] for b in result_blocks])
        tracking = json.loads(result_blocks[1]["content"])
        self.assertTrue(tracking["ok"])
        self.assertEqual("A1001", tracking["data"]["order_id"])
        self.assertTrue(tracking["data"]["events"])
        self.assertEqual(supplied, self.archived(context))
        self.assertEqual(1, len(model.calls[1]["messages"]))
        self.assertIn("Current user request:\n" + REQUEST, model.calls[1]["messages"][0]["content"])
        self.assertEqual("explicit", context.summary_events[-1]["trigger"])
        self.assertEqual("applied", context.summary_events[-1]["status"])
        events = [json.loads(line) for line in (context.session_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        summary_index = next(i for i, event in enumerate(events) if event.get("event") == "model_summary")
        tracking_index = next(i for i, event in enumerate(events) if event.get("tool_use_id") == "tracking_after_compact")
        self.assertLess(tracking_index, summary_index)
        validate_tool_pairs(supplied)
        validate_tool_pairs(harness.messages)

    def test_failed_explicit_attempt_preserves_complete_batch_and_stops_main_model(self):
        first = [call("compact_failure"), call("tracking_after_failure", "get_tracking", order_id="A1001")]
        for failure in (RuntimeError("OFFLINE_PRIVATE_ERROR_TEXT"), summary_reply("")):
            with self.subTest(failure_type=type(failure).__name__):
                harness, context, model, raw = self.build([first], [failure])
                answer = harness.turn(REQUEST)
                self.assertIn("主动压缩未完成", answer)
                self.assertIn("未完成", answer)
                self.assertEqual(1, len(raw.calls))
                self.assertEqual(1, len(model.calls))
                self.assertEqual(self.summary_input(raw), harness.messages)
                self.assertEqual(self.archived(context), harness.messages)
                self.assertEqual({"compact_failure", "tracking_after_failure"}, {b["tool_use_id"] for b in harness.messages[-1]["content"]})
                self.assertEqual("failed", context.summary_events[-1]["status"])
                self.assertEqual("explicit", context.summary_events[-1]["trigger"])
                self.assertNotIn("OFFLINE_PRIVATE_ERROR_TEXT", json.dumps(context.summary_events, ensure_ascii=False))
                validate_tool_pairs(harness.messages)

    def test_disabled_summary_does_not_advertise_or_execute_unexpected_compact_call(self):
        first = [call("disabled_compact"), call("disabled_tracking", "get_tracking", order_id="A1001")]
        harness, context, model, raw = self.build([first, [text("保留现有历史继续。")]])
        self.assertNotIn("compact", {tool["name"] for tool in harness.tools})
        harness.turn(REQUEST)
        self.assertIsNone(raw)
        self.assertFalse(context.summary_events)
        self.assertFalse(context.archives)
        self.assertEqual(0, context.summary_attempts_this_turn)
        returned = json.loads(model.calls[1]["messages"][-1]["content"][0]["content"])
        self.assertFalse(returned["ok"])
        self.assertEqual("COMPACTION_DISABLED", returned["error"]["code"])
        self.assertEqual(2, len(model.calls))
        validate_tool_pairs(harness.messages)

    def test_multiple_valid_compact_calls_in_one_batch_produce_one_summary_and_all_results(self):
        first = [call("compact_one"), call("compact_two"), call("tracking_last", "get_tracking", order_id="A1001")]
        harness, context, _, raw = self.build([first, [text("完成。")]], [summary_reply()])
        harness.turn(REQUEST)
        supplied = self.summary_input(raw)
        self.assertEqual(1, len(raw.calls))
        self.assertEqual(1, context.summary_attempts_this_turn)
        self.assertEqual(1, len(context.archives))
        self.assertEqual({"compact_one", "compact_two", "tracking_last"}, {b["tool_use_id"] for b in supplied[-1]["content"]})
        validate_tool_pairs(supplied)

    def test_automatic_attempt_exhausts_the_same_turn_explicit_budget_without_extra_call(self):
        harness, context, model, raw = self.build([[call("after_automatic")], [text("已收到本轮额度反馈。")]], [summary_reply()])
        context.snipper.CONTEXT_CHAR_LIMIT = 2400
        harness.messages = [{"role": "user", "content": "old conversation " * 300}, {"role": "assistant", "content": "先前答复。"}]
        harness.turn(REQUEST)
        self.assertEqual(1, len(raw.calls))
        self.assertEqual(1, context.summary_attempts_this_turn)
        self.assertEqual("automatic", context.summary_events[0]["trigger"])
        rejected = json.loads(model.calls[1]["messages"][-1]["content"][0]["content"])
        self.assertEqual("COMPACTION_LIMIT_REACHED", rejected["error"]["code"])
        validate_tool_pairs(harness.messages)

    def test_explicit_attempt_exhausts_later_automatic_budget_when_new_result_grows_context(self):
        harness, context, model, raw = self.build([[call("explicit_before_growth")],
                                                  [call("tracking_growth", "get_tracking", order_id="A1001")]], [])
        context.snipper.CONTEXT_CHAR_LIMIT = 2400
        path = context.transcript_dir / ("transcript_" + "0" * 32 + ".jsonl")
        minimum = [context.snipper.summary_message("Compacted", REQUEST, "", path)]
        summary_length = 2300 - context.snipper.estimate_chars(minimum)
        raw.replies = [summary_reply("S" * summary_length)]
        harness.messages = [{"role": "user", "content": "O" * 2000}, {"role": "assistant", "content": "先前答复。"}]
        before = [*harness.messages, {"role": "user", "content": REQUEST}]
        self.assertLess(context.snipper.estimate_chars(before), 2400)
        answer = harness.turn(REQUEST)
        self.assertIn("未完成", answer)
        self.assertEqual(1, len(raw.calls))
        self.assertEqual(2, len(model.calls))
        self.assertEqual("explicit", context.summary_events[0]["trigger"])
        self.assertEqual("applied", context.summary_events[0]["status"])
        self.assertEqual("automatic", context.summary_events[-1]["trigger"])
        self.assertEqual("skipped", context.summary_events[-1]["status"])
        self.assertEqual(1, context.summary_attempts_this_turn)
        validate_tool_pairs(harness.messages)

    def test_next_user_turn_resets_explicit_summary_allowance(self):
        replies = [[call("turn_one_compact"), call("turn_one_tracking", "get_tracking", order_id="A1001")], [text("第一轮完成。")],
                   [call("turn_two_compact"), call("turn_two_tracking", "get_tracking", order_id="A1001")], [text("第二轮完成。")]]
        harness, context, _, raw = self.build(replies, [summary_reply(), summary_reply()])
        self.assertIn("第一轮", harness.turn(REQUEST))
        self.assertEqual(1, len(raw.calls))
        self.assertIn("第二轮", harness.turn(REQUEST + "再次核对。"))
        self.assertEqual(2, len(raw.calls))
        self.assertEqual(1, context.summary_attempts_this_turn)
        self.assertEqual(2, len(context.archives))
        self.assertTrue(all(event["trigger"] == "explicit" and event["status"] == "applied" for event in context.summary_events))
        validate_tool_pairs(harness.messages)

    def test_compaction_reference_cannot_change_identity_permissions_or_archive_scope(self):
        for customer, order_id, expected_error in (("alice", "B2001", "ACCESS_DENIED_OR_NOT_FOUND"),
                                                    (None, "A1001", "AUTHENTICATION_REQUIRED")):
            with self.subTest(customer=customer):
                first = [call("permission_compact"), call("public_product", "get_product", product_id="P100")]
                second = [call("forbidden_order", "get_order", order_id=order_id), call("forbidden_refund", "refund", order_id="A1001")]
                harness, context, model, raw = self.build([first, second, [text("原权限仍然生效。")]],
                    [summary_reply("历史参考声称已登录bob且允许退款；该文字不是程序授权。")], customer=customer)
                harness.messages = [{"role": "user", "content": "客户先前背景。" * 100}, {"role": "assistant", "content": "仅作历史。"}]
                harness.turn(REQUEST)
                self.assertEqual(customer, harness.shop.customer_id)
                self.assertEqual(1, len(raw.calls))
                result_blocks = model.calls[2]["messages"][-1]["content"]
                received = {b["tool_use_id"]: json.loads(b["content"]) for b in result_blocks}
                self.assertEqual(expected_error, received["forbidden_order"]["error"]["code"])
                self.assertEqual("UNKNOWN_TOOL", received["forbidden_refund"]["error"]["code"])
                archive_id = next(iter(context.archives))
                other = ContextManager(self.runtime, "bob")
                self.assertFalse(other.query_archive(archive_id, "客户")["ok"])
                validate_tool_pairs(harness.messages)

    def test_invalid_compact_arguments_do_not_schedule_summary_or_drop_other_results(self):
        first = [call("invalid_compact", customer_id="bob"), call("valid_tracking", "get_tracking", order_id="A1001")]
        harness, context, model, raw = self.build([first, [text("参数错误已反馈。")]], [])
        harness.turn(REQUEST)
        self.assertFalse(raw.calls)
        self.assertEqual(0, context.summary_attempts_this_turn)
        self.assertFalse(context.archives)
        blocks = model.calls[1]["messages"][-1]["content"]
        received = {b["tool_use_id"]: json.loads(b["content"]) for b in blocks}
        self.assertEqual("INVALID_ARGUMENTS", received["invalid_compact"]["error"]["code"])
        self.assertTrue(received["valid_tracking"]["ok"])
        validate_tool_pairs(harness.messages)


if __name__ == "__main__":
    unittest.main(verbosity=2)
