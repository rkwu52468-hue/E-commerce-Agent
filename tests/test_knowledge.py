"""Control/protocol tests with a real local Qdrant and deliberately fake vectors.

These tests contain no semantic model evaluation and never download a model.
"""
import copy
from datetime import date, datetime, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from customer_service.knowledge import (
    build_index, KnowledgeSearch, KnowledgeError, load_documents,
    chunk_documents, COLLECTION, TOP_K, MIN_SCORE, MAX_RESULT_CHARS,
    STORE_TIMEZONE, EFFECTIVE_TIMEZONE, _today,
)


class FakeEmbedder:
    model_name = 'unit-test-fake-embedding'
    dimension = 3
    artifact_fingerprint = '1' * 64

    def __init__(self):
        self.passages = []
        self.queries = []

    def token_count(self, text):
        # Pessimistic fake token count exercises token-budget subdivision.
        return len(text) * 2

    def vector(self, text):
        if 'return' in text:
            return [1., 0., 0.]
        if 'clean' in text:
            return [0., 1., 0.]
        return [0., 0., 1.]

    def passage_embed(self, texts):
        self.passages.extend(texts)
        return [self.vector(text) for text in texts]

    def query_embed(self, text):
        self.queries.append(text)
        return [self.vector(text)]


def document(identifier='returns', body='return within seven days; keep packaging.', **changes):
    record = {'doc_id': identifier, 'title': 'Demo ' + identifier, 'version': 'v1',
              'store_id': 'demo-shop', 'visibility': 'public', 'status': 'active',
              'effective_from': '2020-01-01', 'effective_to': None,
              'kind': 'policy', 'body': body, 'source': 'knowledge_demo/' + identifier + '.md'}
    record.update(changes)
    return record


class ChunkAndCorpusTests(unittest.TestCase):
    def test_store_date_changes_at_utc_1600(self):
        for instant, expected in ((datetime(2026, 11, 6, 15, 59, 59, tzinfo=timezone.utc), date(2026, 11, 6)),
                                  (datetime(2026, 11, 6, 16, 0, 0, tzinfo=timezone.utc), date(2026, 11, 7))):
            with self.subTest(utc=instant.isoformat()):
                with patch('customer_service.knowledge.datetime', wraps=datetime) as clock:
                    clock.now.side_effect = lambda tz: instant.astimezone(tz)
                    self.assertEqual(_today(), expected)
                    clock.now.assert_called_once_with(STORE_TIMEZONE)

    def test_deterministic_chunks_keep_text_and_token_budget(self):
        docs = [document(body='return ' + '汉' * 700, title='标题' * 50)]
        encoder = FakeEmbedder()
        chunks = chunk_documents(docs, encoder, today=date(2026, 11, 6))
        self.assertEqual(''.join(c['text'] for c in chunks), docs[0]['body'])
        self.assertTrue(all(encoder.token_count(c['title'] + '\n' + c['text']) <= 512 for c in chunks))
        self.assertEqual(chunks, chunk_documents(docs, encoder, today=date(2026, 11, 6)))
        self.assertEqual(len({c['chunk_id'] for c in chunks}), len(chunks))

    def test_scope_and_effective_versions_only(self):
        docs = [document(), document('private', visibility='private'),
                document('other', store_id='other-shop'), document('retired', status='retired'),
                document('expired', effective_to='2025-12-31'),
                document('future', effective_from='2027-01-01')]
        self.assertEqual([c['doc_id'] for c in chunk_documents(docs, FakeEmbedder(), today=date(2026, 11, 6))], ['returns'])

    def test_overlapping_versions_rejected(self):
        with self.assertRaises(KnowledgeError):
            chunk_documents([document(), document(version='v2')], FakeEmbedder())

    def test_invalid_metadata_rejected_before_embedding(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'documents.json'
            for changes in ({'source': '../secret'}, {'source': 'C:/secret'}, {'body': ''},
                            {'effective_from': '2026-02-30'}, {'effective_to': '2019-01-01'},
                            {'doc_id': '../../x'}, {'status': 'deleted'}):
                with self.subTest(changes=changes):
                    path.write_text(json.dumps([document(**changes)]), encoding='utf-8')
                    with self.assertRaises(KnowledgeError):
                        load_documents(path)


@unittest.skipUnless(importlib.util.find_spec('qdrant_client'), 'optional RAG dependencies are not installed')
class LocalQdrantControlTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = self.root / 'documents.json'
        self.index = self.root / 'index'
        self.cache = self.root / 'cache'
        self.docs = [document(), document('cleaning', body='clean the cup with warm water.')]
        self.write()
        self.embedder = FakeEmbedder()
        self.manifest = build_index(self.path, self.index, self.cache, embedder=self.embedder)
        self.searcher = KnowledgeSearch(self.path, self.index, self.cache, embedder=self.embedder)

    def write(self):
        self.path.write_text(json.dumps(self.docs, ensure_ascii=False), encoding='utf-8')

    def tearDown(self):
        self.searcher.close()
        self.temporary.cleanup()

    def test_real_local_search_provenance_and_separate_embedding_calls(self):
        result = self.searcher.search('return please')
        self.assertTrue(result['ok'])
        hit = result['data']['hits'][0]
        self.assertEqual(hit['doc_id'], 'returns')
        self.assertEqual(hit['text'], self.docs[0]['body'])
        self.assertEqual(hit['source'], self.docs[0]['source'])
        self.assertEqual(hit['version'], 'v1')
        self.assertIn(hit['chunk_id'], self.searcher.chunks)
        self.assertEqual(self.embedder.queries, ['return please'])
        self.assertEqual(len(self.embedder.passages), 2)
        self.assertEqual(self.manifest['embedding_evidence'], 'injected_adapter')

    def test_no_match_is_successful_empty_result(self):
        result = self.searcher.search('unrelated')
        self.assertTrue(result['ok'])
        self.assertEqual(result['data']['status'], 'no_match')
        self.assertEqual(result['data']['hits'], [])

    def test_query_validation_precedes_embedding(self):
        for query in (None, '', ' ', 123, ['return'], 'x' * 501, '汉' * 300):
            with self.subTest(query=str(query)[:30]):
                self.assertEqual(self.searcher.search(query)['error']['code'], 'INVALID_ARGUMENTS')
        self.assertEqual(self.embedder.queries, [])

    def test_edit_or_delete_source_invalidates_old_index(self):
        for edit in ('changed', 'delete'):
            with self.subTest(edit=edit):
                if edit == 'changed':
                    self.docs[0]['body'] = 'return now requires a different rule'
                else:
                    self.docs.pop()
                self.write()
                self.assertEqual(self.searcher.search('return')['error']['code'], 'KNOWLEDGE_REBUILD_REQUIRED')

    def test_new_generation_retains_old_and_removes_deleted_document(self):
        self.searcher.close()
        old_generation = self.manifest['generation']
        self.docs = [document('cleaning', body='clean the cup with warm water.')]
        self.write()
        new = build_index(self.path, self.index, self.cache, embedder=self.embedder)
        self.assertNotEqual(old_generation, new['generation'])
        self.assertTrue((self.index / old_generation / 'manifest.json').exists())
        self.searcher = KnowledgeSearch(self.path, self.index, self.cache, embedder=self.embedder)
        self.assertEqual(self.searcher.search('return')['data']['hits'], [])

    def test_failed_build_preserves_current(self):
        self.searcher.close()
        broken = FakeEmbedder()
        broken.passage_embed = lambda texts: [[0., 0., 0.] for _ in texts]
        before = (self.index / 'CURRENT').read_text()
        with self.assertRaises(KnowledgeError):
            build_index(self.path, self.index, self.cache, embedder=broken)
        self.assertEqual((self.index / 'CURRENT').read_text(), before)
        self.searcher = KnowledgeSearch(self.path, self.index, self.cache, embedder=self.embedder)
        self.assertTrue(self.searcher.search('return')['data']['hits'])

    def test_database_scope_filter_and_original_payload_validation(self):
        from qdrant_client import models
        original = next(c for c in self.searcher.chunks.values() if c['doc_id'] == 'returns')
        for field, value in [('store_id', 'other-shop'), ('visibility', 'private'), ('status', 'retired'),
                             ('effective_to_day', date(2020, 1, 1).toordinal())]:
            payload = dict(original, **{field: value})
            self.searcher.client.upsert(COLLECTION, points=[models.PointStruct(
                id=str(uuid4()), vector=[1., 0., 0.], payload=payload)], wait=True)
        result = self.searcher.search('return')
        self.assertTrue(result['ok'])
        self.assertEqual(len(result['data']['hits']), 1)
        self.searcher.client.set_payload(COLLECTION, payload={'text': 'forged rule'}, points=[original['point_id']])
        self.assertEqual(self.searcher.search('return')['error']['code'], 'INVALID_INDEX')

    def test_error_is_not_reported_as_no_match_and_close_is_idempotent(self):
        with patch.object(self.searcher.client, 'query_points', side_effect=RuntimeError('private path')):
            result = self.searcher.search('return')
        self.assertFalse(result['ok'])
        self.assertEqual(result['error']['code'], 'KNOWLEDGE_UNAVAILABLE')
        self.assertNotIn('private path', json.dumps(result))
        self.searcher.close()
        self.searcher.close()
        self.assertFalse(self.searcher.search('return')['ok'])

    def test_future_activation_requires_explicit_rebuild(self):
        self.searcher.close()
        self.docs.append(document('next', body='return tomorrow', effective_from='2027-01-01'))
        self.write()
        with patch('customer_service.knowledge._today', return_value=date(2026, 11, 6)):
            build_index(self.path, self.index, self.cache, embedder=self.embedder)
            self.searcher = KnowledgeSearch(self.path, self.index, self.cache, embedder=self.embedder)
        with patch('customer_service.knowledge._today', return_value=date(2027, 1, 1)):
            self.assertEqual(self.searcher.search('return')['error']['code'], 'KNOWLEDGE_REBUILD_REQUIRED')

    def test_expiration_is_rechecked_each_query(self):
        self.searcher.close()
        self.docs[0]['effective_to'] = '2026-12-31'
        self.write()
        with patch('customer_service.knowledge._today', return_value=date(2026, 11, 6)):
            build_index(self.path, self.index, self.cache, embedder=self.embedder)
            self.searcher = KnowledgeSearch(self.path, self.index, self.cache, embedder=self.embedder)
        with patch('customer_service.knowledge._today', return_value=date(2027, 1, 1)):
            result = self.searcher.search('return')
            self.assertTrue(result['ok'])
            self.assertEqual(result['data']['hits'], [])

    def test_dimension_or_manifest_changes_require_rebuild(self):
        self.searcher.close()
        changed = FakeEmbedder()
        changed.dimension = 4
        with self.assertRaises(KnowledgeError) as error:
            KnowledgeSearch(self.path, self.index, self.cache, embedder=changed)
        self.assertEqual(error.exception.code, 'KNOWLEDGE_REBUILD_REQUIRED')
        manifest_path = self.index / self.manifest['generation'] / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['chunk_config']['max_chars'] = 999
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaises(KnowledgeError) as error:
            KnowledgeSearch(self.path, self.index, self.cache, embedder=self.embedder)
        self.assertEqual(error.exception.code, 'INVALID_INDEX')

    def test_same_model_and_dimension_changed_artifacts_require_rebuild(self):
        self.searcher.close()
        changed = FakeEmbedder()
        changed.artifact_fingerprint = '2' * 64
        with self.assertRaises(KnowledgeError) as error:
            KnowledgeSearch(self.path, self.index, self.cache, embedder=changed)
        self.assertEqual(error.exception.code, 'KNOWLEDGE_REBUILD_REQUIRED')
        self.assertEqual(self.manifest['embedding_fingerprint'], self.embedder.artifact_fingerprint)

    def test_effective_calendar_dates_include_start_and_end_day(self):
        self.searcher.close()
        self.docs = [document('baseline', body='clean baseline'),
                     document('boundary', body='return only on November seven',
                              effective_from='2026-11-07', effective_to='2026-11-07')]
        self.write()
        with patch('customer_service.knowledge._today', return_value=date(2026, 11, 7)):
            manifest = build_index(self.path, self.index, self.cache, embedder=self.embedder)
            self.searcher = KnowledgeSearch(self.path, self.index, self.cache, embedder=self.embedder)
        self.assertEqual(manifest['effective_timezone'], EFFECTIVE_TIMEZONE)
        for instant, expected_count in ((datetime(2026, 11, 6, 15, 59, 59, tzinfo=timezone.utc), 0),
                                        (datetime(2026, 11, 6, 16, 0, 0, tzinfo=timezone.utc), 1),
                                        (datetime(2026, 11, 7, 15, 59, 59, tzinfo=timezone.utc), 1),
                                        (datetime(2026, 11, 7, 16, 0, 0, tzinfo=timezone.utc), 0)):
            with self.subTest(utc=instant.isoformat()):
                with patch('customer_service.knowledge.datetime', wraps=datetime) as clock:
                    clock.now.side_effect = lambda tz: instant.astimezone(tz)
                    result = self.searcher.search('return')
                    self.assertTrue(result['ok'])
                    self.assertEqual(len(result['data']['hits']), expected_count)

    def test_corrupt_pointer_and_original_corpus_checked_on_open(self):
        self.searcher.close()
        self.path.write_text('[]', encoding='utf-8')
        with self.assertRaises(KnowledgeError):
            KnowledgeSearch(self.path, self.index, self.cache, embedder=self.embedder)
        (self.index / 'CURRENT').write_text('../somewhere')
        with self.assertRaises(KnowledgeError) as error:
            KnowledgeSearch(self.path, self.index, self.cache, embedder=self.embedder)
        self.assertEqual(error.exception.code, 'INVALID_INDEX')

    def test_default_reader_uses_local_files_only(self):
        self.searcher.close()
        with patch('customer_service.knowledge.FastEmbedder', return_value=self.embedder) as factory:
            self.searcher = KnowledgeSearch(self.path, self.index, self.cache)
        factory.assert_called_once_with(self.cache, local_files_only=True)

    def test_top_k_and_full_result_budget(self):
        self.searcher.close()
        self.docs = [document('returns' + str(i), body='return ' + '汉' * 180,
                              source='knowledge_demo/' + 's' * 240 + str(i)) for i in range(10)]
        self.write()
        build_index(self.path, self.index, self.cache, embedder=self.embedder)
        self.searcher = KnowledgeSearch(self.path, self.index, self.cache, embedder=self.embedder)
        result = self.searcher.search('return')
        hits = result['data']['hits']
        self.assertLessEqual(len(hits), TOP_K)
        self.assertLessEqual(len(json.dumps(hits, ensure_ascii=False, sort_keys=True, separators=(',', ':'))), MAX_RESULT_CHARS)
        self.assertTrue(all(hit['score'] >= MIN_SCORE for hit in hits))


if __name__ == '__main__':
    unittest.main()
