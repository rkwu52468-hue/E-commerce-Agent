"""No-network acceptance of the opt-in, two-request live micro trial.

All SDK clients here are deterministic fakes. No key is read and no provider is called.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from customer_service.context import ContextManager, validate_tool_pairs
import customer_service.live_micro as live_micro
from vendor.snip import ContextCompactor

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {"model": "deepseek-flash", "base_url": "https://api.deepseek.com/anthropic"}


def as_text(text):
    return {"type": "text", "text": text}


def lookup(trial, call_id="lookup_a", **overrides):
    arguments = {"output_id": trial["output_id"], "query": "historical_note"}
    arguments.update(overrides)
    return {"type": "tool_use", "id": call_id, "name": "query_tool_output", "input": arguments}


def reply(content, stop_reason="end_turn", usage=None):
    return {"content": content, "stop_reason": stop_reason,
            "usage": {"input_tokens": 100, "output_tokens": 10} if usage is None else usage}


class SDKObject:
    def __init__(self, data):
        self.data = deepcopy(data)

    def model_dump(self, **kwargs):
        return deepcopy(self.data)

    def __getattr__(self, name):
        try:
            return self.data[name]
        except KeyError:
            raise AttributeError(name) from None


class SDKReply:
    def __init__(self, data):
        self.content = [SDKObject(block) for block in data["content"]]
        self.stop_reason = data["stop_reason"]
        self.usage = SDKObject(data["usage"])

    def model_dump(self, **kwargs):
        return {"content": [b.model_dump() for b in self.content],
                "stop_reason": self.stop_reason, "usage": self.usage.model_dump()}


class FakeSDKClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        value = self.replies.pop(0)
        if isinstance(value, BaseException):
            raise value
        return SDKReply(value)


class LiveMicroTrialTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "runtime").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="live_micro_acceptance_", dir=ROOT / "runtime")
        self.addCleanup(self.temporary.cleanup)
        self.trial_root = Path(self.temporary.name)
        (self.trial_root / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")

    def trial(self):
        return live_micro.prepare_trial(self.trial_root)

    def model(self, replies):
        client = FakeSDKClient(replies)
        model = live_micro.BudgetedModel("offline-fake-not-a-real-key", CONFIG, client=client)
        return model, client

    def test_trial_threshold_is_instance_only_and_answer_stays_in_saved_output(self):
        trial = self.trial()
        self.assertEqual(5000, trial["context"].snipper.CONTEXT_CHAR_LIMIT)
        self.assertEqual(50000, ContextCompactor.CONTEXT_CHAR_LIMIT)
        other = ContextManager(self.trial_root / "runtime", "alice")
        self.assertEqual(50000, other.snipper.CONTEXT_CHAR_LIMIT)
        self.assertTrue(trial["context"].micro_events)
        self.assertNotIn(trial["expected_code"], json.dumps(trial["messages"], ensure_ascii=False))
        path = trial["context"].tool_outputs[trial["output_id"]]
        self.assertIn(trial["expected_code"], Path(path).read_text(encoding="utf-8"))
        self.assertLessEqual(len(json.dumps(trial["messages"], ensure_ascii=False)), 5000)
        validate_tool_pairs(trial["messages"])

    def test_success_needs_two_requests_and_second_request_receives_queried_answer(self):
        trial = self.trial()
        model, client = self.model([reply([lookup(trial)], "tool_use"),
                                    reply([as_text("查回编号：" + trial["expected_code"])])])
        report = live_micro.execute_trial(trial, model)
        self.assertTrue(report["success"])
        self.assertEqual(2, report["api_calls"])
        self.assertEqual(2, len(client.calls))
        first, second = client.calls
        self.assertNotIn(trial["expected_code"], json.dumps(first, ensure_ascii=False))
        self.assertIn(trial["expected_code"], json.dumps(second["messages"], ensure_ascii=False))
        for request in client.calls:
            self.assertEqual(512, request["max_tokens"])
            self.assertEqual({"type": "disabled"}, request["thinking"])
            self.assertLessEqual(len(json.dumps(request["messages"], ensure_ascii=False)), 5000)
            self.assertLessEqual(len(json.dumps(request, ensure_ascii=False)), 10000)
            validate_tool_pairs(request["messages"])
        tool_results = [b for m in second["messages"] if isinstance(m.get("content"), list)
                        for b in m["content"] if b.get("type") == "tool_result"]
        queried = json.loads(tool_results[-1]["content"])
        matches = queried["data"]["matches"]
        self.assertLessEqual(len(matches), 1)
        self.assertTrue(all(len(item["content"]) <= 240 for item in matches))

    def test_no_tool_first_answer_is_failure_even_if_it_guesses_the_code(self):
        trial = self.trial()
        model, client = self.model([reply([as_text(trial["expected_code"])])])
        report = live_micro.execute_trial(trial, model)
        self.assertFalse(report["success"])
        self.assertTrue(report["failure_reason"])
        self.assertEqual(1, report["api_calls"])
        self.assertEqual(1, len(client.calls))

    def test_wrong_ids_queries_extra_arguments_and_multiple_tools_never_execute_lookup(self):
        makers = [
            lambda t: [lookup(t, output_id="0" * 32)],
            lambda t: [lookup(t, query="guess-the-answer")],
            lambda t: [lookup(t, customer_id="bob")],
            lambda t: [lookup(t), lookup(t, call_id="lookup_b")],
            lambda t: [{"type": "tool_use", "id": "wrong_tool", "name": "get_order",
                        "input": {"order_id": "A1001"}}],
        ]
        for index, maker in enumerate(makers):
            trial = self.trial()
            model, client = self.model([reply(maker(trial), "tool_use")])
            with self.subTest(case=index), patch.object(trial["context"], "query_tool_output",
                                                       side_effect=AssertionError("Disallowed lookup executed")) as query:
                report = live_micro.execute_trial(trial, model)
                self.assertFalse(report["success"])
                self.assertTrue(report["failure_reason"])
                self.assertEqual(1, report["api_calls"])
                self.assertEqual(1, len(client.calls))
                query.assert_not_called()

    def test_first_truncated_response_stops_before_tool_execution(self):
        trial = self.trial()
        model, client = self.model([reply([lookup(trial)], "max_tokens")])
        with patch.object(trial["context"], "query_tool_output",
                          side_effect=AssertionError("Truncated reply must not execute lookup")) as query:
            report = live_micro.execute_trial(trial, model)
            self.assertFalse(report["success"])
            self.assertEqual(1, report["api_calls"])
            self.assertEqual(1, len(client.calls))
            query.assert_not_called()

    def test_second_round_tool_call_fails_without_a_third_request(self):
        trial = self.trial()
        model, client = self.model([reply([lookup(trial)], "tool_use"),
                                    reply([lookup(trial, call_id="lookup_again")], "tool_use")])
        report = live_micro.execute_trial(trial, model)
        self.assertFalse(report["success"])
        self.assertTrue(report["failure_reason"])
        self.assertEqual(2, report["api_calls"])
        self.assertEqual(2, len(client.calls))

    def test_wrong_final_answer_does_not_count_as_success(self):
        trial = self.trial()
        model, client = self.model([reply([lookup(trial)], "tool_use"),
                                    reply([as_text("无法确定旧编号。")])])
        report = live_micro.execute_trial(trial, model)
        self.assertFalse(report["success"])
        self.assertTrue(report["failure_reason"])
        self.assertEqual(2, report["api_calls"])
        self.assertEqual(2, len(client.calls))

    def test_over_message_budget_stops_before_any_sdk_request(self):
        trial = self.trial()
        trial["messages"].append({"role": "user", "content": "budget-overflow " * 500})
        self.assertGreater(len(json.dumps(trial["messages"], ensure_ascii=False)), 5000)
        model, client = self.model([])
        report = live_micro.execute_trial(trial, model)
        self.assertFalse(report["success"])
        self.assertTrue(report["failure_reason"])
        self.assertEqual(0, report["api_calls"])
        self.assertFalse(client.calls)

    def test_full_serialized_request_budget_is_checked_before_sdk_request(self):
        model, client = self.model([])
        with self.assertRaises(Exception):
            model.complete([{"role": "user", "content": "short question"}], [], "request-overflow " * 1000)
        self.assertFalse(client.calls)

    def test_adapter_request_cap_disables_thinking_and_limits_each_output(self):
        model, client = self.model([reply([as_text("first")]), reply([as_text("second")])])
        for _ in range(2):
            model.complete([{"role": "user", "content": "short question"}], [], "short system")
        with self.assertRaises(Exception):
            model.complete([{"role": "user", "content": "third question"}], [], "short system")
        self.assertEqual(2, len(client.calls))
        for request in client.calls:
            self.assertEqual(512, request["max_tokens"])
            self.assertEqual({"type": "disabled"}, request["thinking"])

    def test_usage_and_peak_estimate_do_not_double_count_cached_input(self):
        trial = self.trial()
        usage1 = {"input_tokens": 100, "cache_read_input_tokens": 30,
                  "cache_creation_input_tokens": 7, "output_tokens": 10}
        usage2 = {"input_tokens": 80, "cache_read_input_tokens": 20,
                  "cache_creation_input_tokens": 3, "output_tokens": 15}
        model, _ = self.model([reply([lookup(trial)], "tool_use", usage1),
                               reply([as_text(trial["expected_code"])], usage=usage2)])
        report = live_micro.execute_trial(trial, model)
        self.assertTrue(report["success"])
        self.assertEqual({"input_tokens": 180, "cache_read_input_tokens": 50,
                          "cache_creation_input_tokens": 10, "output_tokens": 25}, report["usage_totals"])
        self.assertAlmostEqual(0.0000873, report["estimated_peak_cost_usd"], places=12)

    def test_missing_usage_does_not_report_an_artificial_zero_cost(self):
        trial = self.trial()
        model, _ = self.model([reply([lookup(trial)], "tool_use"),
                               reply([as_text(trial["expected_code"])], usage={"output_tokens": 5})])
        report = live_micro.execute_trial(trial, model)
        self.assertTrue(report["success"])
        self.assertIsNone(report["estimated_peak_cost_usd"])

    def test_second_request_exception_marks_total_cost_unknown_and_preserves_known_usage(self):
        trial = self.trial()
        model, client = self.model([reply([lookup(trial)], "tool_use"),
                                    TimeoutError("offline simulated second-request timeout")])
        report = live_micro.execute_trial(trial, model)
        self.assertFalse(report["success"])
        self.assertEqual(2, report["api_calls"])
        self.assertEqual(2, len(client.calls))
        self.assertFalse(report["usage_complete"])
        self.assertEqual(1, report["usage_missing_requests"])
        self.assertIsNone(report["estimated_peak_cost_usd"])
        self.assertEqual(100, report["usage_totals"]["input_tokens"])
        self.assertEqual(10, report["usage_totals"]["output_tokens"])
        self.assertAlmostEqual(0.000042, report["estimated_known_usage_peak_cost_usd"], places=12)

    def test_sdk_exception_has_no_automatic_retry(self):
        trial = self.trial()
        model, client = self.model([RuntimeError("offline simulated SDK failure")])
        report = live_micro.execute_trial(trial, model)
        self.assertFalse(report["success"])
        self.assertTrue(report["failure_reason"])
        self.assertEqual(1, report["api_calls"])
        self.assertEqual(1, len(client.calls))

    def test_dry_run_requires_no_key_and_never_constructs_an_api_adapter(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "", "ANTHROPIC_API_KEY": ""}), \
             patch("getpass.getpass", side_effect=AssertionError("dry run requested a key")), \
             patch.object(live_micro, "BudgetedModel", side_effect=AssertionError("dry run created API adapter")):
            report = live_micro.run_trial(self.trial_root, dry_run=True)
        self.assertEqual(0, report["api_calls"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
