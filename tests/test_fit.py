"""Offline acceptance for fit_tool_results after micro compaction.

All histories and logistics facts are synthetic, and all artifacts stay in runtime.
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
    unit = label + ": simulated parcel transit checkpoint; "
    return (unit * (length // len(unit) + 1))[:length]


def use(identifier):
    return {"type": "tool_use", "id": identifier, "name": "get_tracking",
            "input": {"order_id": "A1001"}}


def history(bodies, consumed=False, request="只查询 A1001 的模拟物流。"):
    messages = [{"role": "user", "content": request}]
    for index, body in enumerate(bodies):
        identifier = f"fit_tracking_{index}"
        messages += [{"role": "assistant", "content": [use(identifier)]},
                     {"role": "user", "content": [{"type": "tool_result",
                      "tool_use_id": identifier, "content": body, "is_error": False}]}]
    if consumed:
        messages.append({"role": "assistant", "content": "已读取上述模拟物流。"})
    return messages


def blocks(messages):
    return [b for m in messages if isinstance(m.get("content"), list)
            for b in m["content"] if b.get("type") == "tool_result"]


def shape(messages):
    cloned = deepcopy(messages)
    for block in blocks(cloned):
        block["content"] = "<body>"
    return cloned


def preview_of(marker):
    return marker.split("Preview:\n", 1)[1].removesuffix("\n</persisted-output>")


class NeverCalledModel:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools, system):
        self.calls += 1
        raise AssertionError("Unshortenable user text must stop before the model")


class FitTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "runtime").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="fit_acceptance_", dir=ROOT / "runtime")
        self.addCleanup(self.temporary.cleanup)
        self.runtime = Path(self.temporary.name)

    def context(self, customer="alice"):
        return ContextManager(self.runtime, customer)

    def test_fit_can_shorten_the_latest_unseen_result_without_changing_ids_or_count(self):
        context = self.context()
        body = payload(55000, "latest unseen logistics")
        messages = history([body])
        before = deepcopy(messages)
        self.assertGreater(estimate(messages), 50000)
        prepared = context.prepare(messages, messages[0]["content"])
        self.assertEqual(len(before), len(prepared))
        self.assertEqual(shape(before), shape(prepared))
        self.assertTrue(blocks(prepared)[0]["content"].startswith("<persisted-output>\n"))
        self.assertLessEqual(estimate(prepared), 40000)
        self.assertFalse(context.micro_events)
        self.assertTrue(context.fit_events)
        event = context.fit_events[-1]
        self.assertTrue(event["attempted"])
        self.assertEqual(["fit_tracking_0"], event["shortened_tool_use_ids"])
        self.assertEqual(len(prepared), event["message_count"])
        self.assertEqual(estimate(before), event["chars_before"])
        self.assertEqual(estimate(prepared), event["chars_after"])
        validate_tool_pairs(prepared)

    def test_fit_can_shorten_the_recent_three_results_protected_by_micro(self):
        context = self.context()
        bodies = [payload(19000, "recent0"), payload(18000, "recent1"), payload(17000, "recent2")]
        messages = history(bodies, consumed=True)
        before = deepcopy(messages)
        prepared = context.prepare(messages, messages[0]["content"])
        result_blocks = blocks(prepared)
        self.assertFalse(context.micro_events)
        self.assertNotEqual(bodies[0], result_blocks[0]["content"])
        self.assertEqual(bodies[1:], [b["content"] for b in result_blocks[1:]])
        self.assertEqual(["fit_tracking_0"], context.fit_events[-1]["shortened_tool_use_ids"])
        self.assertEqual(shape(before), shape(prepared))
        self.assertLessEqual(estimate(prepared), 40000)
        validate_tool_pairs(prepared)

    def test_largest_result_is_tried_first_and_reaching_target_leaves_others_unchanged(self):
        context = self.context()
        bodies = [payload(8000, "first small"), payload(24000, "middle largest"), payload(13000, "last medium")]
        messages = history(bodies)
        before = deepcopy(messages)
        self.assertGreater(estimate(messages), 40000)
        returned = context.snipper.fit_tool_results(messages, target_chars=40000)
        result_blocks = blocks(returned)
        self.assertEqual(bodies[0], result_blocks[0]["content"])
        self.assertNotEqual(bodies[1], result_blocks[1]["content"])
        self.assertEqual(bodies[2], result_blocks[2]["content"])
        self.assertEqual(shape(before), shape(returned))
        self.assertLessEqual(estimate(returned), 40000)
        self.assertEqual(1, len(list(context.tool_results_dir.glob("*.txt"))))
        validate_tool_pairs(returned)

    def test_preview_is_exact_first_1000_characters_and_full_file_keeps_tail_fact(self):
        context = self.context()
        tail = "TAIL_FACT_FIT_991: package handoff requires source verification"
        body = payload(55000, "logistics source") + "\n" + tail
        messages = history([body])
        prepared = context.prepare(messages, messages[0]["content"])
        marker = blocks(prepared)[0]["content"]
        self.assertEqual(body[:1000], preview_of(marker))
        self.assertNotIn(tail, marker)
        self.assertEqual(1, len(context.tool_outputs))
        output_id, path = next(iter(context.tool_outputs.items()))
        self.assertEqual(body, Path(path).read_text(encoding="utf-8"))
        answer = context.query_tool_output(output_id, "TAIL_FACT_FIT_991")
        self.assertTrue(answer["ok"])
        self.assertIn(tail, json.dumps(answer["data"]["matches"], ensure_ascii=False))

    def test_existing_large_preview_is_shortened_using_the_same_original_file(self):
        context = self.context()
        body = payload(55000, "existing complete output") + "\nORIGINAL_FILE_TAIL_992"
        original_marker = context.snipper.persisted_preview("create-source", body, preview_chars=len(body))
        source_path = Path(context.snipper.persisted_output_path(original_marker))
        self.assertFalse(context.tool_outputs)
        self.assertEqual(1, len(list(context.tool_results_dir.glob("*.txt"))))
        messages = history([original_marker])
        prepared = context.prepare(messages, messages[0]["content"])
        replacement = blocks(prepared)[0]["content"]
        reused_path = Path(context.snipper.persisted_output_path(replacement))
        self.assertEqual(source_path, reused_path)
        self.assertEqual(1, len(list(context.tool_results_dir.glob("*.txt"))))
        self.assertEqual(body, source_path.read_text(encoding="utf-8"))
        self.assertFalse(source_path.read_text(encoding="utf-8").startswith("<persisted-output>"))
        self.assertEqual(body[:1000], preview_of(replacement))
        self.assertEqual({source_path.stem: source_path.resolve()}, context.tool_outputs)

    def test_fit_reuses_a_micro_saved_file_instead_of_persisting_its_path_marker(self):
        context = self.context()
        bodies = [payload(6000, "micro original")] + [payload(500, f"recent{i}") for i in range(3)]
        messages = context.snipper.micro_compact(history(bodies, consumed=True))
        micro_marker = blocks(messages)[0]["content"]
        source_path = Path(context.snipper.persisted_output_path(micro_marker))
        files_before = set(context.tool_results_dir.glob("*.txt"))
        context.snipper.fit_tool_results(messages, target_chars=0)
        self.assertEqual(micro_marker, blocks(messages)[0]["content"])
        self.assertEqual(bodies[0], source_path.read_text(encoding="utf-8"))
        # fit considers the other results too, so inspect only the original source identity.
        self.assertTrue(source_path in set(context.tool_results_dir.glob("*.txt")))
        self.assertEqual(source_path, Path(context.snipper.persisted_output_path(micro_marker)))
        candidates_with_marker_as_body = [path for path in context.tool_results_dir.glob("*.txt")
                                         if path.read_text(encoding="utf-8") == micro_marker]
        self.assertFalse(candidates_with_marker_as_body)
        self.assertTrue(files_before)

    def test_unapplied_preview_does_not_authorize_a_short_result_file(self):
        context = self.context()
        request = payload(55000, "unshortenable user discussion")
        short_body = payload(500, "small tracking result")
        messages = history([short_body], request=request)
        before = deepcopy(messages)
        prepared = context.prepare(messages, request)
        self.assertEqual(before, prepared)
        self.assertFalse(context.tool_outputs)
        self.assertTrue(context.fit_events[-1]["attempted"])
        self.assertEqual([], context.fit_events[-1]["shortened_tool_use_ids"])
        self.assertEqual(estimate(before), context.fit_events[-1]["chars_after"])
        created = list(context.tool_results_dir.glob("*.txt"))
        self.assertTrue(created)
        for path in created:
            self.assertEqual(short_body, path.read_text(encoding="utf-8"))
            self.assertFalse(context.query_tool_output(path.stem, "small")["ok"])

    def test_fit_outputs_remain_unavailable_to_other_sessions_and_customers(self):
        context = self.context()
        messages = history([payload(55000, "private logistics")])
        context.prepare(messages, messages[0]["content"])
        output_id = next(iter(context.tool_outputs))
        self.assertTrue(context.query_tool_output(output_id, "private logistics")["ok"])
        for other in (self.context(), self.context("bob")):
            with self.subTest(session=other.session_dir):
                self.assertFalse(other.query_tool_output(output_id, "private logistics")["ok"])

    def test_fit_is_not_run_when_micro_gets_under_limit_even_if_target_is_not_reached(self):
        context = self.context()
        messages = history([payload(14000, f"consumed{i}") for i in range(5)], consumed=True)
        prepared = context.prepare(messages, messages[0]["content"])
        self.assertTrue(context.micro_events)
        self.assertGreater(estimate(prepared), 40000)
        self.assertLessEqual(estimate(prepared), 50000)
        self.assertFalse(context.fit_events)
        self.assertEqual(2, len(context.tool_outputs))

    def test_unshortenable_user_text_still_stops_without_any_model_request(self):
        context = self.context()
        request = payload(55000, "customer supplied discussion")
        model = NeverCalledModel()
        harness = Harness(Shop("alice", ROOT / "data/shop.json"), context, model)
        answer = harness.turn(request)
        self.assertIn("未完成", answer)
        self.assertEqual(0, model.calls)
        self.assertFalse(context.tool_outputs)
        self.assertGreater(estimate(harness.messages), 50000)
        self.assertEqual(request, harness.messages[0]["content"])
        validate_tool_pairs(harness.messages)


if __name__ == "__main__":
    unittest.main(verbosity=2)
