"""No-network acceptance of the shared-ledger three-request summary trial."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from customer_service.context import ARCHIVE_TOOL, ContextManager, validate_tool_pairs
import customer_service.live_summary as live_summary
from vendor.snip import ContextCompactor

ROOT = Path(__file__).resolve().parents[1]
QUERY = "SUMMARY-CUSTOMER-CONSTRAINT"
_DEFAULT = object()


def text(value):
    return {"type": "text", "text": value}


def response(content, stop="end_turn", usage=_DEFAULT):
    if usage is _DEFAULT:
        usage = {"input_tokens": 100, "output_tokens": 20}
    return {"content": content, "stop_reason": stop, "usage": usage}


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
        self.usage = None if data["usage"] is None else SDKObject(data["usage"])


class FakeSDKClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        item = self.replies.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            item = item(kwargs)
        return SDKReply(item)


class LiveSummaryTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "runtime").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="live_summary_acceptance_", dir=ROOT / "runtime")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "data").mkdir()
        (self.root / "runtime").mkdir()
        shutil.copyfile(ROOT / "config.json", self.root / "config.json")
        shutil.copyfile(ROOT / "data/shop.json", self.root / "data/shop.json")
        self.config = json.loads((self.root / "config.json").read_text(encoding="utf-8"))

    def trial(self):
        return live_summary.prepare_trial(self.root)

    def budgeted(self, replies):
        raw = FakeSDKClient(replies)
        budgeted = live_summary.BudgetedClient("offline-fake-key-not-real", self.config, client=raw)
        self.assertFalse(budgeted.is_live)
        return budgeted, raw

    def lookup(self, trial, call_id="accepted_lookup", **overrides):
        archive_id = next(event["archive_id"] for event in trial["context"].summary_events
                          if event["status"] == "applied")
        arguments = {"archive_id": archive_id, "query": QUERY}
        arguments.update(overrides)
        return {"type": "tool_use", "id": call_id, "name": "query_archive", "input": arguments}

    def answer(self, trial):
        return "核对编号：" + trial["expected_code"] + "。客户要求：不要退款、不要创建售后单。"

    def normal_responses(self, trial, include_summary_facts=True):
        summary = ("已知客户编号" + trial["expected_code"] + "，不要退款，不要创建售后单。"
                   if include_summary_facts else "历史已讨论订单物流，具体原话仍需核对归档。")
        return [response([text(summary)]),
                lambda kwargs: response([text("先核对客户原话。"), self.lookup(trial)], "tool_use"),
                response([text(self.answer(trial))])]

    def saved_history(self, trial):
        data = json.loads((trial["context"].session_dir / "history.json").read_text(encoding="utf-8"))
        return data["messages"] if isinstance(data, dict) else data

    def test_prepare_is_keyless_and_local_with_instance_only_threshold(self):
        with patch("getpass.getpass", side_effect=AssertionError("prepare requested a key")), \
             patch.object(live_summary, "BudgetedClient", side_effect=AssertionError("prepare created SDK adapter")):
            trial = self.trial()
        self.assertEqual(5000, trial["context"].snipper.CONTEXT_CHAR_LIMIT)
        self.assertEqual(50000, ContextCompactor.CONTEXT_CHAR_LIMIT)
        default_context = ContextManager(self.root / "runtime", "alice")
        self.assertEqual(50000, default_context.snipper.CONTEXT_CHAR_LIMIT)
        before = [*deepcopy(trial["history"]), {"role": "user", "content": trial["active_request"]}]
        self.assertLess(len(before), 50)
        self.assertGreater(len(json.dumps(before, ensure_ascii=False)), 6500)
        self.assertIn(trial["expected_code"], json.dumps(trial["history"], ensure_ascii=False))
        self.assertNotIn(trial["expected_code"], trial["active_request"])
        self.assertIsInstance(trial["plan"], dict)
        validate_tool_pairs(before)

    def test_success_uses_three_fake_requests_original_summary_and_paired_task_history(self):
        trial = self.trial()
        before = [*deepcopy(trial["history"]), {"role": "user", "content": trial["active_request"]}]
        budgeted, raw = self.budgeted(self.normal_responses(trial))
        report = live_summary.execute_trial(trial, budgeted)
        self.assertTrue(report["success"])
        self.assertEqual(0, report["api_calls"])
        self.assertEqual(3, report["sdk_requests_attempted"])
        self.assertEqual(1, report["summary_client_calls"])
        self.assertEqual(2, report["task_client_calls"])
        self.assertEqual(["summary", "task_lookup", "task_answer"], [c["phase"] for c in budgeted.calls])
        self.assertEqual(3, len(raw.calls))
        for index, request in enumerate(raw.calls):
            self.assertEqual({"type": "disabled"}, request["thinking"])
            self.assertEqual(2000 if index == 0 else 512, request["max_tokens"])
            self.assertLessEqual(len(json.dumps(request, ensure_ascii=False)), 12000)
            self.assertNotIn("phase", request)
            if index:
                self.assertLessEqual(len(json.dumps(request["messages"], ensure_ascii=False)), 5000)
                validate_tool_pairs(request["messages"])
        self.assertNotIn("tools", raw.calls[0])
        self.assertEqual(before, json.loads(raw.calls[0]["messages"][0]["content"]))
        self.assertIn("Current user request:\n" + trial["active_request"], raw.calls[1]["messages"][0]["content"])
        self.assertTrue(report["summary_contains_code"])
        self.assertTrue(report["summary_contains_constraint"])
        saved = self.saved_history(trial)
        validate_tool_pairs(saved)
        accepted = next(m for m in saved if isinstance(m.get("content"), list)
                        and any(b.get("id") == "accepted_lookup" for b in m["content"]))
        self.assertEqual("先核对客户原话。", accepted["content"][0]["text"])
        self.assertIn(trial["expected_code"], report["answer"])
        self.assertTrue((trial["context"].session_dir / "trial-report.json").is_file())
        self.assertTrue((trial["context"].session_dir / "events.jsonl").is_file())
        self.assertFalse((self.root / "runtime/live-summary-latest.json").exists())

    def test_summary_may_omit_details_and_archive_lookup_still_recovers_them(self):
        trial = self.trial()
        budgeted, raw = self.budgeted(self.normal_responses(trial, include_summary_facts=False))
        report = live_summary.execute_trial(trial, budgeted)
        self.assertTrue(report["success"])
        self.assertFalse(report["summary_contains_code"])
        self.assertFalse(report["summary_contains_constraint"])
        self.assertNotIn(trial["expected_code"], json.dumps(raw.calls[1]["messages"], ensure_ascii=False))
        self.assertIn(trial["expected_code"], json.dumps(raw.calls[2]["messages"], ensure_ascii=False))
        self.assertEqual(3, report["sdk_requests_attempted"])

    def test_empty_truncated_tool_and_oversized_summary_stop_after_one_request(self):
        candidates = [response([text("")]), response([], "end_turn"),
                      response([text("truncated summary")], "max_tokens"),
                      response([{"type": "tool_use", "id": "REJECTED_SUMMARY_TOOL", "name": "refund", "input": {}}], "tool_use"),
                      response([text("summary overflow " * 400)])]
        for index, summary in enumerate(candidates):
            with self.subTest(case=index):
                trial = self.trial()
                budgeted, raw = self.budgeted([summary])
                report = live_summary.execute_trial(trial, budgeted)
                self.assertFalse(report["success"])
                self.assertTrue(report["failure_reason"])
                self.assertEqual(1, report["sdk_requests_attempted"])
                self.assertEqual(0, report["task_client_calls"])
                self.assertEqual(1, len(raw.calls))
                validate_tool_pairs(self.saved_history(trial))
                self.assertNotIn("REJECTED_SUMMARY_TOOL", json.dumps(self.saved_history(trial), ensure_ascii=False))

    def test_invalid_lookup_responses_stop_at_two_and_never_execute_archive_query(self):
        makers = [
            lambda t: response([text("REJECTED_LOOKUP_TEXT " + self.answer(t))]),
            lambda t: response([self.lookup(t)], "end_turn"),
            lambda t: response([self.lookup(t)], "max_tokens"),
            lambda t: response([self.lookup(t, call_id="")], "tool_use"),
            lambda t: response([self.lookup(t, archive_id="transcript_" + "0" * 32)], "tool_use"),
            lambda t: response([self.lookup(t, query="wrong-query")], "tool_use"),
            lambda t: response([self.lookup(t, customer_id="bob")], "tool_use"),
            lambda t: response([self.lookup(t), self.lookup(t, call_id="second_lookup")], "tool_use"),
            lambda t: response([{"type": "tool_use", "id": "wrong_tool", "name": "refund", "input": {}}], "tool_use"),
        ]
        for index, maker in enumerate(makers):
            with self.subTest(case=index):
                trial = self.trial()
                replies = self.normal_responses(trial)
                replies[1] = lambda kwargs, maker=maker, trial=trial: maker(trial)
                budgeted, raw = self.budgeted(replies[:2])
                with patch.object(trial["context"], "query_archive", side_effect=AssertionError("Invalid lookup executed")) as query:
                    report = live_summary.execute_trial(trial, budgeted)
                self.assertFalse(report["success"])
                self.assertTrue(report["failure_reason"])
                self.assertEqual(2, report["sdk_requests_attempted"])
                self.assertEqual(2, len(raw.calls))
                query.assert_not_called()
                saved = self.saved_history(trial)
                validate_tool_pairs(saved)
                self.assertFalse(any(b.get("type") == "tool_use" for m in saved
                                     if isinstance(m.get("content"), list) for b in m["content"]))
                self.assertIn("response_content", budgeted.calls[-1])
                self.assertNotIn("REJECTED_LOOKUP_TEXT", json.dumps(saved, ensure_ascii=False))

    def test_failed_empty_or_wrong_archive_lookup_stops_before_final_sdk_request(self):
        cases = [
            {"ok": False, "error": {"code": "SIMULATED_ARCHIVE_FAILURE", "message": "offline fixture failure"}},
            {"ok": True, "data": {"matches": []}},
            {"ok": True, "data": {"matches": [{"message_index": 2, "role": "user",
                "content": "SUMMARY-CUSTOMER-CONSTRAINT: WRONG_CODE; 不要退款，不要创建售后单。"}]}},
        ]
        for index, result in enumerate(cases):
            with self.subTest(case=index):
                trial = self.trial()
                budgeted, raw = self.budgeted(self.normal_responses(trial))
                with patch.object(trial["context"], "query_archive", return_value=deepcopy(result)) as query:
                    report = live_summary.execute_trial(trial, budgeted)
                self.assertFalse(report["success"])
                self.assertEqual("archive_lookup_failed_or_missing_code", report["failure_reason"])
                self.assertEqual("", report["answer"])
                self.assertEqual(0, report["api_calls"])
                self.assertEqual(2, report["sdk_requests_attempted"])
                self.assertEqual(1, report["summary_client_calls"])
                self.assertEqual(1, report["task_client_calls"])
                self.assertEqual(2, len(raw.calls))
                self.assertEqual(1, len(raw.replies))
                query.assert_called_once()
                saved = self.saved_history(trial)
                validate_tool_pairs(saved)
                results = [b for message in saved if isinstance(message.get("content"), list)
                           for b in message["content"] if b.get("type") == "tool_result"]
                self.assertEqual(1, len(results))
                self.assertEqual("accepted_lookup", results[0]["tool_use_id"])
                self.assertEqual(result["ok"], json.loads(results[0]["content"])["ok"])

    def test_invalid_final_response_stops_at_three_and_is_not_appended_to_valid_history(self):
        makers = [
            lambda t: response([text("REJECTED_FINAL_TRUNCATED " + self.answer(t))], "max_tokens"),
            lambda t: response([text("")]),
            lambda t: response([text("REJECTED_FINAL_NO_CONSTRAINT " + t["expected_code"])]),
            lambda t: response([text("REJECTED_FINAL_NO_CODE 不要退款，不要创建售后单。")]),
            lambda t: response([{"type": "tool_use", "id": "REJECTED_FINAL_TOOL", "name": "query_archive", "input": {}}], "tool_use"),
        ]
        for index, maker in enumerate(makers):
            with self.subTest(case=index):
                trial = self.trial()
                replies = self.normal_responses(trial)
                replies[2] = maker(trial)
                budgeted, raw = self.budgeted(replies)
                report = live_summary.execute_trial(trial, budgeted)
                self.assertFalse(report["success"])
                self.assertEqual("", report["answer"])
                self.assertTrue(report["failure_reason"])
                self.assertEqual(3, report["sdk_requests_attempted"])
                self.assertEqual(3, len(raw.calls))
                saved = self.saved_history(trial)
                validate_tool_pairs(saved)
                self.assertNotIn("REJECTED_FINAL_", json.dumps(saved, ensure_ascii=False))
                self.assertIn("REJECTED_FINAL_" if index != 1 else '"text": ""',
                              json.dumps(budgeted.calls[-1]["response_content"], ensure_ascii=False))

    def test_timeout_at_each_phase_has_no_retry_unknown_total_and_safe_error_metadata(self):
        secret = "NEVER_LOG_FAKE_AUTH_TOKEN_EXCEPTION_TEXT"
        for failed_index in range(3):
            with self.subTest(phase=failed_index):
                trial = self.trial()
                replies = self.normal_responses(trial)
                replies[failed_index] = TimeoutError(secret)
                budgeted, raw = self.budgeted(replies[:failed_index + 1])
                report = live_summary.execute_trial(trial, budgeted)
                self.assertFalse(report["success"])
                self.assertEqual(failed_index + 1, report["sdk_requests_attempted"])
                self.assertEqual(failed_index + 1, len(raw.calls))
                self.assertEqual(0, report["api_calls"])
                self.assertFalse(report["usage_complete"])
                self.assertIsNone(report["usage_totals"])
                self.assertIsNone(report["estimated_peak_cost_usd"])
                self.assertEqual(100 * failed_index, report["known_usage_totals"]["input_tokens"])
                self.assertEqual("TimeoutError", budgeted.calls[-1]["error_type"])
                metadata = json.dumps({"report": report, "ledger": budgeted.calls}, ensure_ascii=False)
                self.assertNotIn(secret, metadata)
                self.assertNotIn(secret, (trial["context"].session_dir / "events.jsonl").read_text(encoding="utf-8"))
                validate_tool_pairs(self.saved_history(trial))

    def test_partial_or_invalid_usage_keeps_total_unknown_without_affecting_behavior(self):
        candidates = [{"input_tokens": 200}, {"input_tokens": True, "output_tokens": 2},
                      {"input_tokens": 200, "output_tokens": -1},
                      {"input_tokens": 200, "output_tokens": 30, "cache_read_input_tokens": "3"}]
        for usage in candidates:
            with self.subTest(usage=usage):
                trial = self.trial()
                replies = self.normal_responses(trial)
                lookup = replies[1]
                replies[1] = lambda kwargs, lookup=lookup, usage=usage: {**lookup(kwargs), "usage": usage}
                budgeted, _ = self.budgeted(replies)
                report = live_summary.execute_trial(trial, budgeted)
                self.assertTrue(report["success"])
                self.assertFalse(report["usage_complete"])
                self.assertIsNone(report["usage_totals"])
                self.assertIsNone(report["estimated_peak_cost_usd"])
                self.assertEqual(usage, budgeted.calls[1]["usage"])
                self.assertFalse(budgeted.calls[1]["usage_complete"])
                self.assertIsInstance(report["known_usage_totals"], dict)

    def test_missing_usage_is_none_in_ledger_and_does_not_become_zero_total(self):
        trial = self.trial()
        replies = self.normal_responses(trial)
        lookup = replies[1]
        replies[1] = lambda kwargs: {**lookup(kwargs), "usage": None}
        budgeted, _ = self.budgeted(replies)
        report = live_summary.execute_trial(trial, budgeted)
        self.assertTrue(report["success"])
        self.assertIsNone(budgeted.calls[1]["usage"])
        self.assertFalse(report["usage_complete"])
        self.assertIsNone(report["usage_totals"])
        self.assertEqual(200, report["known_usage_totals"]["input_tokens"])
        self.assertEqual(40, report["known_usage_totals"]["output_tokens"])

    def test_complete_usage_aggregates_each_phase_without_double_counting_cache(self):
        trial = self.trial()
        replies = self.normal_responses(trial)
        usages = [{"input_tokens": 100, "cache_read_input_tokens": 10, "output_tokens": 20},
                  {"input_tokens": 200, "cache_read_input_tokens": 20, "cache_creation_input_tokens": 5, "output_tokens": 30},
                  {"input_tokens": 300, "cache_read_input_tokens": 30, "output_tokens": 40}]
        original_lookup = replies[1]
        replies = [{**replies[0], "usage": usages[0]},
                   lambda kwargs: {**original_lookup(kwargs), "usage": usages[1]},
                   {**replies[2], "usage": usages[2]}]
        budgeted, _ = self.budgeted(replies)
        report = live_summary.execute_trial(trial, budgeted)
        self.assertTrue(report["success"])
        self.assertTrue(report["usage_complete"])
        self.assertEqual({"input_tokens": 600, "cache_read_input_tokens": 60,
                          "cache_creation_input_tokens": 5, "output_tokens": 90}, report["usage_totals"])
        self.assertEqual(report["usage_totals"], report["known_usage_totals"])
        self.assertEqual(3, len(report["request_usage"]))

    def test_oversized_summary_payload_stops_before_any_sdk_request(self):
        trial = self.trial()
        trial["history"] = [{"role": "user", "content": "oversized synthetic history " * 600}]
        budgeted, raw = self.budgeted([])
        report = live_summary.execute_trial(trial, budgeted)
        self.assertFalse(report["success"])
        self.assertEqual(0, report["sdk_requests_attempted"])
        self.assertEqual(0, report["api_calls"])
        self.assertFalse(raw.calls)
        self.assertFalse(budgeted.calls)
        validate_tool_pairs(self.saved_history(trial))

    def test_task_message_budget_rejects_second_request_without_consuming_ledger_entry(self):
        budgeted, raw = self.budgeted([response([text("short summary")])])
        budgeted.create(model=self.config["model"], messages=[{"role": "user", "content": "history"}],
                        system="summary only", max_tokens=2000)
        oversized = [{"role": "user", "content": "task history " * 500}]
        self.assertGreater(len(json.dumps(oversized, ensure_ascii=False)), 5000)
        with self.assertRaises(RuntimeError):
            budgeted.create(model=self.config["model"], messages=oversized,
                            tools=[ARCHIVE_TOOL], system="task", max_tokens=512)
        self.assertEqual(1, len(raw.calls))
        self.assertEqual(1, len(budgeted.calls))

    def test_global_three_request_limit_blocks_a_fourth_sdk_attempt(self):
        trial = self.trial()
        budgeted, raw = self.budgeted(self.normal_responses(trial))
        report = live_summary.execute_trial(trial, budgeted)
        self.assertTrue(report["success"])
        with self.assertRaises(RuntimeError):
            budgeted.create(**raw.calls[-1])
        self.assertEqual(3, len(raw.calls))
        self.assertEqual(3, len(budgeted.calls))

    def test_dry_run_is_keyless_and_never_overwrites_a_saved_live_report(self):
        destination = self.root / "runtime/live-summary-latest.json"
        sentinel = b'{"simulated_previous_live_report":true,"must_be_preserved":true}\n'
        destination.write_bytes(sentinel)
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "", "ANTHROPIC_API_KEY": ""}), \
             patch("getpass.getpass", side_effect=AssertionError("dry-run requested key")), \
             patch.object(live_summary, "BudgetedClient", side_effect=AssertionError("dry-run created SDK adapter")):
            preview = live_summary.run_summary_trial(self.root, dry_run=True)
        self.assertEqual(0, preview["api_calls"])
        self.assertEqual(sentinel, destination.read_bytes())
        self.assertTrue(list((self.root / "runtime").rglob("*preview*.json")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
