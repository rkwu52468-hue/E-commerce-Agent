"""Offline acceptance of old-result micro compaction and controlled retrieval.

JSON budgets use the complete messages serialization, not just tool body sizes.
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


def estimate(messages):
    return len(json.dumps(messages, default=str, ensure_ascii=False))


def payload(length, label="tracking"):
    # No quotes, slashes or newlines: each added body character adds one JSON character.
    unit = label + ": courier checkpoint awaiting parcel handoff; "
    return (unit * (length // len(unit) + 1))[:length]


def call(call_id):
    return {"type": "tool_use", "id": call_id, "name": "get_tracking",
            "input": {"order_id": "A1001"}}


def result(call_id, body):
    return {"type": "tool_result", "tool_use_id": call_id, "content": body,
            "is_error": False}


def history(bodies, call_ids=None, one_batch=False, consumed=True):
    call_ids = call_ids or [f"tracking_{index}" for index in range(len(bodies))]
    messages = [{"role": "user", "content": "查询 A1001 物流，保持只读。"}]
    if one_batch:
        messages += [{"role": "assistant", "content": [call(i) for i in call_ids]},
                     {"role": "user", "content": [result(i, b) for i, b in zip(call_ids, bodies)]}]
    else:
        for identifier, body in zip(call_ids, bodies):
            messages += [{"role": "assistant", "content": [call(identifier)]},
                         {"role": "user", "content": [result(identifier, body)]}]
    if consumed:
        messages.append({"role": "assistant", "content": "已读取上述物流记录。"})
    return messages


def result_blocks(messages):
    return [block for message in messages if isinstance(message.get("content"), list)
            for block in message["content"] if block.get("type") == "tool_result"]


def structure_without_result_bodies(messages):
    cloned = deepcopy(messages)
    for block in result_blocks(cloned):
        block["content"] = "<result body>"
    return cloned


class NeverCalledModel:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools, system):
        self.calls += 1
        raise AssertionError("Over-budget histories must never reach a model")


class MicroTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "runtime").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="micro_acceptance_", dir=ROOT / "runtime")
        self.addCleanup(self.temp.cleanup)
        self.runtime = Path(self.temp.name)

    def context(self, customer="alice"):
        return ContextManager(self.runtime, customer)

    def prepared_large_history(self, context=None, call_ids=None):
        context = context or self.context()
        bodies = [payload(14000, f"tracking evidence {i}") for i in range(5)]
        messages = history(bodies, call_ids=call_ids)
        self.assertGreater(estimate(messages), 50000)
        prepared = context.prepare(messages, messages[0]["content"])
        return context, bodies, prepared

    def test_prepare_uses_full_json_budget_and_preserves_message_structure(self):
        context = self.context()
        messages = history([""] * 8)
        overhead = estimate(messages)
        total_body_chars = 50001 - overhead
        lengths = [total_body_chars // 8] * 8
        for index in range(total_body_chars % 8):
            lengths[index] += 1
        messages = history([payload(n, f"record{i}") for i, n in enumerate(lengths)])
        before = deepcopy(messages)
        self.assertLess(sum(len(b["content"]) for b in result_blocks(messages)), 50000)
        self.assertEqual(50001, estimate(messages))
        self.assertLess(len(messages), 50)
        prepared = context.prepare(messages, messages[0]["content"])
        self.assertEqual(len(before), len(prepared))
        self.assertEqual(structure_without_result_bodies(before),
                         structure_without_result_bodies(prepared))
        self.assertLessEqual(estimate(prepared), 40000)
        self.assertTrue(context.tool_outputs)
        self.assertTrue(context.micro_events)
        validate_tool_pairs(prepared)

    def test_micro_protects_three_result_blocks_even_in_one_user_message(self):
        context = self.context()
        bodies = [payload(1000, f"batch{i}") for i in range(8)]
        messages = history(bodies, one_batch=True)
        before = deepcopy(messages)
        returned = context.snipper.micro_compact(messages)
        blocks = result_blocks(returned)
        self.assertIs(messages, returned)
        self.assertEqual(len(before), len(returned))
        self.assertTrue(all(blocks[i]["content"] != bodies[i] for i in range(5)))
        self.assertEqual(bodies[-3:], [b["content"] for b in blocks[-3:]])
        self.assertEqual(structure_without_result_bodies(before),
                         structure_without_result_bodies(returned))
        validate_tool_pairs(returned)

    def test_latest_unseen_result_is_protected_separately_from_three_consumed_results(self):
        context = self.context()
        bodies = [payload(1000, f"consumed{i}") for i in range(5)]
        messages = history(bodies)
        newest = payload(12000, "newest unseen")
        messages += [{"role": "assistant", "content": [call("newest")]},
                     {"role": "user", "content": [result("newest", newest)]}]
        returned = context.snipper.micro_compact(messages)
        blocks = result_blocks(returned)
        self.assertNotEqual(bodies[0], blocks[0]["content"])
        self.assertNotEqual(bodies[1], blocks[1]["content"])
        self.assertEqual(bodies[2:], [b["content"] for b in blocks[2:5]])
        self.assertEqual(newest, blocks[-1]["content"])
        validate_tool_pairs(returned)

    def test_result_length_120_is_skipped_but_121_is_a_candidate(self):
        context = self.context()
        bodies = [payload(120, "short"), payload(121, "candidate")]
        bodies += [payload(500, f"recent{i}") for i in range(3)]
        returned = context.snipper.micro_compact(history(bodies))
        blocks = result_blocks(returned)
        self.assertEqual(bodies[0], blocks[0]["content"])
        self.assertNotEqual(bodies[1], blocks[1]["content"])
        saved = context.snipper.persisted_output_path(blocks[1]["content"])
        self.assertIsNotNone(saved)
        self.assertEqual(bodies[1], Path(saved).read_text(encoding="utf-8"))
        self.assertEqual(bodies[2:], [b["content"] for b in blocks[2:]])

    def test_target_budget_stops_before_all_eligible_old_results_are_replaced(self):
        context = self.context()
        bodies = [payload(7000, f"target{i}") for i in range(8)]
        messages = history(bodies)
        self.assertGreater(estimate(messages), 50000)
        returned = context.snipper.micro_compact(messages, target_chars=40000)
        changed = [index for index, (old, block) in enumerate(zip(bodies, result_blocks(returned)))
                   if old != block["content"]]
        self.assertGreater(len(changed), 0)
        self.assertLess(len(changed), len(bodies) - 3)
        self.assertLessEqual(estimate(returned), 40000)
        self.assertEqual(list(range(len(changed))), changed)

    def test_prepared_output_uuids_avoid_sanitized_tool_id_filename_collisions(self):
        ids = ["call/a", "call?a", "keep0", "keep1", "keep2"]
        context, bodies, prepared = self.prepared_large_history(call_ids=ids)
        self.assertEqual(2, len(context.tool_outputs))
        self.assertEqual(2, len(set(context.tool_outputs.values())))
        saved_bodies = []
        for output_id, path in context.tool_outputs.items():
            self.assertRegex(output_id, r"^[0-9a-f]{32}$")
            self.assertEqual(output_id, Path(path).stem)
            saved_bodies.append(Path(path).read_text(encoding="utf-8"))
        self.assertCountEqual(bodies[:2], saved_bodies)
        self.assertEqual(ids, [b["tool_use_id"] for b in result_blocks(prepared)])

    def test_controlled_query_recovers_original_old_result_text(self):
        context, bodies, prepared = self.prepared_large_history()
        source = next((key, path) for key, path in context.tool_outputs.items()
                      if Path(path).read_text(encoding="utf-8") == bodies[0])
        output_id, _ = source
        self.assertNotIn("tracking evidence 0", json.dumps(prepared, ensure_ascii=False))
        answer = context.query_tool_output(output_id, "tracking evidence 0")
        self.assertTrue(answer["ok"])
        self.assertTrue(answer["data"]["matches"])
        self.assertIn("tracking evidence 0", json.dumps(answer["data"]["matches"], ensure_ascii=False))

    def test_output_ids_cannot_read_other_sessions_or_arbitrary_paths(self):
        context, _, _ = self.prepared_large_history()
        output_id = next(iter(context.tool_outputs))
        for other in (self.context(), self.context("bob")):
            with self.subTest(session=other.session_dir):
                self.assertFalse(other.query_tool_output(output_id, "tracking")["ok"])
        invalid_ids = ["../shop", "../../data/shop.json", "0" * 32,
                       str(ROOT / "data/shop.json")]
        for value in invalid_ids:
            with self.subTest(output_id=value):
                self.assertFalse(context.query_tool_output(value, "tracking")["ok"])

    def test_unregistered_direct_micro_file_does_not_grant_read_permission(self):
        context = self.context()
        returned = context.snipper.micro_compact(history([payload(1000)] * 4))
        saved = context.snipper.persisted_output_path(result_blocks(returned)[0]["content"])
        self.assertIsNotNone(saved)
        self.assertTrue(Path(saved).is_file())
        self.assertFalse(context.tool_outputs)
        self.assertFalse(context.query_tool_output(Path(saved).stem, "tracking")["ok"])

    def test_unshortenable_user_text_can_leave_budget_unmet_without_any_model_request(self):
        context = self.context()
        original_request = payload(55000, "unshortenable customer discussion")
        messages = [{"role": "user", "content": original_request}]
        self.assertGreater(estimate(messages), 50000)
        model = NeverCalledModel()
        harness = Harness(Shop("alice", ROOT / "data/shop.json"), context, model)
        harness.messages = messages
        answer = harness.turn("继续核对当前物流。")
        self.assertIn("未完成", answer)
        self.assertEqual(0, model.calls)
        self.assertFalse(context.tool_outputs)
        self.assertEqual(original_request, harness.messages[0]["content"])
        validate_tool_pairs(harness.messages)

    def test_dispatcher_and_model_schema_only_accept_output_id_and_query(self):
        context, _, _ = self.prepared_large_history()
        output_id = next(iter(context.tool_outputs))
        harness = Harness(Shop("alice", ROOT / "data/shop.json"), context, NeverCalledModel())
        definition = next(t for t in harness.tools if t["name"] == "query_tool_output")
        self.assertEqual({"output_id", "query"}, set(definition["input_schema"]["properties"]))
        self.assertFalse(definition["input_schema"]["additionalProperties"])
        self.assertTrue(harness.call_tool("query_tool_output",
                        {"output_id": output_id, "query": "tracking"})["ok"])
        for arguments in [{"output_id": output_id},
                          {"output_id": output_id, "query": "tracking", "customer_id": "bob"},
                          {"path": str(ROOT / "data/shop.json"), "query": "tracking"}]:
            with self.subTest(arguments=arguments):
                self.assertFalse(harness.call_tool("query_tool_output", arguments)["ok"])

    def test_below_budget_prepare_does_not_eagerly_save_old_results(self):
        context = self.context()
        messages = history([payload(1000, f"small{i}") for i in range(8)])
        before = deepcopy(messages)
        self.assertLess(estimate(messages), 50000)
        prepared = context.prepare(messages, messages[0]["content"])
        self.assertEqual(before, prepared)
        self.assertFalse(context.tool_outputs)
        self.assertFalse(context.micro_events)


if __name__ == "__main__":
    unittest.main(verbosity=2)
