"""Offline regressions for verbatim source recovery and safe memory diagnostics."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

from customer_service.context import ContextManager, validate_tool_pairs
from customer_service.harness import Harness
from customer_service.memory import MemoryExtractor, MemoryStore, admit_memory_candidates
from customer_service.memory_records import resolve_source_evidence
from customer_service.shop import Shop
from test_memory import RawSDK, reply
from test_memory_extraction import Model, candidate, text

ROOT = Path(__file__).resolve().parents[1]
FEEDBACK = '以后回答时，请先给结论，再说明依据和资料缺口'
DIAGNOSTIC_FIELDS = {'kind', 'scope', 'reason', 'evidence_adjusted'}
REASONS = {'accepted', 'blocked_source', 'nonpersistent_scope',
           'evidence_not_source_sentence', 'temporary_scope',
           'invalid_source_characters', 'no_long_term_intent',
           'unsupported_record_context', 'conditional_global_preference',
           'conflicting_preference', 'unsupported_speaker',
           'unsupported_preference_value'}


def record(kind, evidence, scope='persistent'):
    return {'type': kind, 'scope': scope, 'evidence': evidence}


def provenance(query):
    return {'kind': 'user_utterance', 'session_id': 'a' * 32, 'turn': 1,
            'evidence_sha256': hashlib.sha256(query.encode('utf-8')).hexdigest()}


class MemoryDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='memory_diagnostics_', dir=ROOT / 'runtime')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = MemoryStore(self.root / 'memory', 'alice')

    def snapshot(self):
        return {p.name: p.read_bytes() for p in self.store.directory.glob('*.md')}

    def assert_safe_diagnostics(self, diagnostics):
        self.assertIsInstance(diagnostics, list)
        for item in diagnostics:
            self.assertEqual(DIAGNOSTIC_FIELDS, set(item))
            self.assertIn(item['kind'], ('user', 'feedback', 'project', 'reference', 'typed_preference'))
            self.assertIn(item['scope'], ('persistent', 'current_task'))
            self.assertIn(item['reason'], REASONS)
            self.assertIs(type(item['evidence_adjusted']), bool)

    def harness(self, proposals):
        raw = RawSDK([reply(proposals)])
        extractor = MemoryExtractor(raw, 'offline-extraction')
        context = ContextManager(self.root / 'runtime', 'alice')
        model = Model([[text('当前答复已完成。')]])
        h = Harness(Shop('alice', ROOT / 'data/shop.json'), context, model,
                    memory_store=self.store, memory_extractor=extractor)
        return h, raw, model, extractor

    def rows(self, context, filename):
        return [json.loads(line) for line in (context.session_dir / filename).read_text(encoding='utf-8').splitlines()]

    def test_single_terminal_full_stop_is_compatible_but_storage_keeps_exact_customer_source(self):
        cases = [(FEEDBACK, FEEDBACK + '。'),
                 (FEEDBACK + '。', FEEDBACK),
                 (FEEDBACK, '  ' + FEEDBACK + '.  '),
                 (FEEDBACK + '.', FEEDBACK + '。')]
        for i, (query, evidence) in enumerate(cases):
            with self.subTest(case=i):
                self.assertEqual(query, resolve_source_evidence(query, evidence))
                diagnostics = []
                admitted = admit_memory_candidates(query, [record('feedback', evidence)], diagnostics)
                self.assertEqual([record('feedback', query)], admitted)
                self.assert_safe_diagnostics(diagnostics)
                self.assertEqual([{'kind': 'feedback', 'scope': 'persistent', 'reason': 'accepted',
                                   'evidence_adjusted': True}], diagnostics)
                store = MemoryStore(self.root / ('case_' + str(i)), 'alice')
                # Commit the raw proposal too: the store must use admission's source text.
                store.save_candidates([record('feedback', evidence)], query, provenance(query))
                self.assertEqual(query, store.list_records()[0]['body'])

    def test_exact_full_source_preserves_legacy_api_and_reports_unadjusted_acceptance(self):
        cases = [('user', '我更喜欢大容量保温杯，以后选购时优先考虑容量。'),
                 ('feedback', FEEDBACK + '。'),
                 ('project', '我的采购项目长期用于社区团购，主要面向社区居民。'),
                 ('reference', '请记住这个资料网址，今后参考：https://example.com/Guide.v1')]
        for kind, query in cases:
            with self.subTest(kind=kind):
                original = record(kind, query)
                self.assertEqual([original], admit_memory_candidates(query, [original]))
                diagnostics = []
                self.assertEqual([original], admit_memory_candidates(query, [original], diagnostics))
                self.assert_safe_diagnostics(diagnostics)
                self.assertEqual([{'kind': kind, 'scope': 'persistent', 'reason': 'accepted',
                                   'evidence_adjusted': False}], diagnostics)
        query = '以后请用中文回复。'
        diagnostics = []
        proposed = candidate('reply_language', 'zh-CN', query)
        self.assertEqual([proposed], admit_memory_candidates(query, [proposed], diagnostics))
        self.assertEqual('typed_preference', diagnostics[0]['kind'])
        self.assertFalse(diagnostics[0]['evidence_adjusted'])
        self.assert_safe_diagnostics(diagnostics)

    def test_omitted_conditions_negation_and_current_task_cannot_be_repaired_into_persistent_memory(self):
        conditional = '以后平时回复简洁，涉及退货流程时详细说明。'
        negated = '以后回答时，不要先省略依据，再给结论。'
        cases = [(conditional, record('user', '以后平时回复简洁。'), 'evidence_not_source_sentence'),
                 (negated, record('feedback', '以后回答时，先省略依据，再给结论。'), 'evidence_not_source_sentence'),
                 (FEEDBACK, record('feedback', FEEDBACK, 'current_task'), 'nonpersistent_scope'),
                 ('以后请用中文回复。', candidate('reply_language', 'zh-CN', '以后请用中文回复。', 'current_task'), 'nonpersistent_scope'),
                 (conditional, candidate('reply_detail', 'concise', conditional), 'conflicting_preference'),
                 ('以后涉及退货流程时详细说明。', candidate('reply_detail', 'detailed', '以后涉及退货流程时详细说明。'), 'conditional_global_preference'),
                 (FEEDBACK + '。不要保存到长期记忆。', record('feedback', FEEDBACK + '。'), 'blocked_source')]
        for query, proposed, reason in cases:
            with self.subTest(reason=reason, kind=proposed.get('type', 'typed_preference')):
                diagnostics = []
                self.assertEqual([], admit_memory_candidates(query, [proposed], diagnostics))
                self.assert_safe_diagnostics(diagnostics)
                self.assertEqual(reason, diagnostics[0]['reason'])
        # A full conditional statement can still be kept as a general user record.
        self.assertEqual([record('user', conditional)], admit_memory_candidates(conditional, [record('user', conditional)]))

    def test_urls_questions_internal_punctuation_and_ambiguous_matches_remain_exact(self):
        url = '请记住这个资料网址，今后参考：https://example.com/Guide.v1'
        for evidence in (url + '。', url.replace('/Guide.v1', '/Guide.v2'),
                         url.replace('/Guide.v1', '/guide.v1'), url.replace('https://', 'http://')):
            with self.subTest(evidence=evidence):
                self.assertIsNone(resolve_source_evidence(url, evidence))
                diagnostics = []
                self.assertEqual([], admit_memory_candidates(url, [record('reference', evidence)], diagnostics))
                self.assertEqual('evidence_not_source_sentence', diagnostics[0]['reason'])
        for query, evidence in [(FEEDBACK + '？', FEEDBACK + '。'),
                                (FEEDBACK + '?', FEEDBACK + '.'),
                                (FEEDBACK, FEEDBACK.replace('，', ',')),
                                (FEEDBACK, FEEDBACK + '..'),
                                (FEEDBACK + '。' + FEEDBACK + '. ', FEEDBACK)]:
            with self.subTest(query=query, evidence=evidence):
                self.assertIsNone(resolve_source_evidence(query, evidence))

    def test_extractor_receipt_reports_only_safe_reason_metadata_and_never_retries(self):
        query = '以后平时回复简洁，涉及退货流程时详细说明。'
        rejected = record('user', '以后平时回复简洁。')
        raw = RawSDK([reply([rejected])])
        extractor = MemoryExtractor(raw, 'offline-extraction')
        receipt = extractor.extract(query)
        self.assertEqual('rejected', receipt['status'])
        self.assertEqual([], receipt['candidates'])
        self.assertEqual(1, receipt['raw_candidate_count'])
        self.assertEqual(['evidence_not_source_sentence'], receipt['rejection_reasons'])
        self.assert_safe_diagnostics(receipt['candidate_diagnostics'])
        self.assertEqual(1, len(raw.calls))
        self.assertEqual(1, len(extractor.calls))
        self.assertEqual(600, raw.calls[0]['max_tokens'])
        self.assertNotIn('tools', raw.calls[0])
        self.assertEqual({'type': 'disabled'}, raw.calls[0]['thinking'])
        self.assertTrue(receipt['usage_complete'])
        safe = json.dumps({'diagnostics': receipt['candidate_diagnostics'],
                           'reasons': receipt['rejection_reasons']}, ensure_ascii=False)
        self.assertNotIn(query, safe)
        self.assertNotIn(rejected['evidence'], safe)
        self.assertNotIn('alice', safe)
        self.assertNotIn(str(self.root), safe)
        for response in (reply([rejected], stop='max_tokens'), reply('not-json')):
            with self.subTest(response_stop=response['stop_reason']):
                other_raw = RawSDK([response])
                other = MemoryExtractor(other_raw, 'offline-extraction')
                invalid = other.extract(query)
                self.assertEqual('invalid', invalid['status'])
                self.assertEqual([], invalid['candidates'])
                self.assertEqual(['invalid_or_incomplete_response'], invalid['rejection_reasons'])
                self.assertEqual(1, len(other_raw.calls))
                self.assertEqual(1, len(other.calls))

    def test_rejection_reasons_reach_terminal_events_and_audit_without_changing_files(self):
        self.store.set_preference('reply_detail', 'concise')
        before = self.snapshot()
        query = '以后平时回复简洁，涉及退货流程时详细说明。'
        h, raw, model, extractor = self.harness([record('user', '以后平时回复简洁。')])
        output = io.StringIO()
        with redirect_stdout(output):
            answer = h.turn(query)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(1, len(raw.calls))
        self.assertEqual(1, len(model.calls))
        self.assertTrue(answer.startswith('当前答复已完成。'))
        self.assertIn('未保存', answer)
        self.assertNotIn('已保存', answer)
        self.assertEqual('当前答复已完成。', h.messages[-1]['content'][0]['text'])
        self.assertIn('reasons=evidence_not_source_sentence', output.getvalue())
        self.assertNotIn(query, output.getvalue())
        events = [row for row in self.rows(h.context, 'events.jsonl') if row['event'] == 'memory_extraction']
        audit = [row for row in self.rows(h.context, 'audit.jsonl') if row['event'] == 'MemoryExtraction']
        self.assertEqual(1, len(events))
        self.assertEqual('rejected', events[0]['status'])
        finished = next(row for row in audit if row['phase'] == 'finished')
        for row in (events[0], finished):
            self.assertEqual(1, row['raw_candidate_count'])
            self.assertEqual(['evidence_not_source_sentence'], row['rejection_reasons'])
            self.assertEqual(extractor.calls[0]['candidate_diagnostics'], row['candidate_diagnostics'])
            self.assert_safe_diagnostics(row['candidate_diagnostics'])
        self.assertFalse(any(row['phase'] == 'write' for row in audit))
        serialized = json.dumps(audit, ensure_ascii=False)
        for secret in (query, '以后平时回复简洁。', 'concise', str(self.root)):
            self.assertNotIn(secret, serialized)
        validate_tool_pairs(h.messages)

    def test_end_to_end_adjustment_saves_original_customer_body_without_forging_main_answer(self):
        h, raw, model, _ = self.harness([record('feedback', FEEDBACK + '。')])
        with redirect_stdout(io.StringIO()):
            answer = h.turn(FEEDBACK)
        saved = self.store.list_records()
        self.assertEqual(1, len(saved))
        self.assertEqual(FEEDBACK, saved[0]['body'])
        self.assertEqual(hashlib.sha256(FEEDBACK.encode('utf-8')).hexdigest(), saved[0]['provenance']['evidence_sha256'])
        self.assertEqual('当前答复已完成。', h.messages[-1]['content'][0]['text'])
        self.assertTrue(answer.startswith('当前答复已完成。'))
        self.assertIn('已保存长期记忆', answer)
        self.assertEqual(1, len(raw.calls))
        self.assertEqual(1, len(model.calls))
        events = [row for row in self.rows(h.context, 'events.jsonl') if row['event'] == 'memory_extraction']
        self.assertEqual('saved', events[0]['status'])
        self.assertEqual([], events[0]['rejection_reasons'])
        diagnostics = events[0]['candidate_diagnostics']
        self.assert_safe_diagnostics(diagnostics)
        self.assertTrue(diagnostics[0]['evidence_adjusted'])
        audit = self.rows(h.context, 'audit.jsonl')
        self.assertLess(next(i for i, row in enumerate(audit) if row['event'] == 'Stop'),
                        next(i for i, row in enumerate(audit) if row.get('phase') == 'write'))
        self.assertNotIn(FEEDBACK, json.dumps(audit, ensure_ascii=False))
        validate_tool_pairs(h.messages)


if __name__ == '__main__':
    unittest.main(verbosity=2)
