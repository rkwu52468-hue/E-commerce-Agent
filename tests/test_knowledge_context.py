"""Adjacent-source context tests; fake embeddings, real local Qdrant, no API calls."""
from copy import deepcopy
from datetime import date
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from customer_service.knowledge import (
    assemble_context, chunk_documents, build_index, KnowledgeSearch, KnowledgeError,
    _canonical, MAX_RESULT_CHARS, MAX_CONTEXT_CHUNKS, TOP_K, MIN_SCORE, CONTEXT_POLICY,
)

TODAY = date(2026, 11, 6)


class FakeEmbedder:
    model_name = 'context-test-fake-embedding'
    dimension = 3
    artifact_fingerprint = '2' * 64

    def token_count(self, text):
        return len(text)

    def passage_embed(self, texts):
        for text in texts:
            if 'MATERIALS' in text:
                yield [.2, .98, 0.]
            elif 'STEEL' in text:
                yield [1., 0., 0.]
            elif 'DRY' in text:
                yield [.8, .6, 0.]
            elif 'OTHER' in text:
                yield [.9, .4359, 0.]
            else:
                yield [1., 0., 0.]

    def query_embed(self, text):
        return [[0., 0., 1.] if text == 'unrelated' else [1., 0., 0.]]


def document(identifier='manual', body='materials\n\nanchor\n\ndry', **changes):
    row = {'doc_id': identifier, 'title': 'Demo ' + identifier, 'version': 'v1',
           'store_id': 'demo-shop', 'visibility': 'public', 'status': 'active',
           'effective_from': '2020-01-01', 'effective_to': None,
           'kind': 'manual', 'body': body, 'source': 'knowledge_demo/' + identifier + '.md'}
    row.update(changes)
    return row


def chunks_for(*docs):
    rows = chunk_documents(list(docs), FakeEmbedder(), today=TODAY)
    return {row['chunk_id']: row for row in rows}


def at(chunks, position, doc_id='manual'):
    return next(row for row in chunks.values() if row['position'] == position and row['doc_id'] == doc_id)


def hit_for(chunk, score=.8):
    fields = ('chunk_id', 'doc_id', 'title', 'version', 'source', 'text', 'content_hash', 'kind')
    return {**{key:chunk[key] for key in fields}, 'score':score}


class AdjacentContextTests(unittest.TestCase):
    def test_adds_previous_and_next_with_real_ids_and_no_cosine_score(self):
        chunks = chunks_for(document())
        hits = [hit_for(at(chunks, 1))]
        original = deepcopy(hits)
        context, truncated = assemble_context(hits, chunks, {('manual', 'v1')}, TODAY)
        self.assertEqual([row['position'] for row in context], [0, 2])
        self.assertEqual(hits, original)
        self.assertFalse(truncated)
        for row in context:
            self.assertIsNone(row['score'])
            self.assertEqual(row['retrieved_via'], 'adjacent_context')
            self.assertEqual(row['anchor_chunk_id'], hits[0]['chunk_id'])
            source = chunks[row['chunk_id']]
            for key in ('text', 'source', 'version', 'content_hash'):
                self.assertEqual(row[key], source[key])

    def test_deduplicates_seeds_and_shared_neighbors(self):
        chunks = chunks_for(document(body='zero\n\none\n\ntwo\n\nthree'))
        hits = [hit_for(at(chunks, 1)), hit_for(at(chunks, 2))]
        context, truncated = assemble_context(hits, chunks, {('manual','v1')}, TODAY)
        self.assertEqual([row['position'] for row in context], [0, 3])
        ids = [row['chunk_id'] for row in hits + context]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertFalse(truncated)

    def test_expansion_is_one_hop_not_recursive(self):
        chunks = chunks_for(document(body='zero\n\none\n\ntwo\n\nthree\n\nfour'))
        context, _ = assemble_context([hit_for(at(chunks, 2))], chunks, {('manual','v1')}, TODAY)
        self.assertEqual([row['position'] for row in context], [1, 3])

    def test_no_seed_means_no_context(self):
        context, truncated = assemble_context([], chunks_for(document()), {('manual','v1')}, TODAY)
        self.assertEqual(context, [])
        self.assertFalse(truncated)

    def test_never_crosses_version_source_store_visibility_or_validity(self):
        changes = [{'doc_id':'other'}, {'version':'v2'}, {'source':'knowledge_demo/other.md'},
                   {'store_id':'other-shop'}, {'visibility':'private'}, {'status':'retired'},
                   {'effective_from':'2027-01-01'}, {'effective_to':'2025-12-31'},
                   {'kind':'faq'}, {'title':'Different document title'}]
        for mutation in changes:
            with self.subTest(mutation=mutation):
                chunks = chunks_for(document())
                middle = at(chunks, 1)
                seed = hit_for(middle)
                original = at(chunks, 0)
                altered = deepcopy(original)
                altered.update(mutation)
                chunks = {middle['chunk_id']:middle, altered['chunk_id']:altered}
                eligible = {('manual','v1'), ('manual','v2'), ('other','v1')}
                context, truncated = assemble_context([seed], chunks, eligible, TODAY)
                self.assertEqual(context, [])
                self.assertFalse(truncated)

    def test_current_eligible_ids_are_checked_for_anchor(self):
        chunks = chunks_for(document())
        with self.assertRaises(KnowledgeError):
            assemble_context([hit_for(at(chunks,1))], chunks, set(), TODAY)

    def test_inconsistent_neighbor_hash_is_rejected(self):
        chunks = chunks_for(document())
        at(chunks,0)['content_hash'] = '0' * 64
        with self.assertRaises(KnowledgeError) as error:
            assemble_context([hit_for(at(chunks,1))], chunks, {('manual','v1')}, TODAY)
        self.assertEqual(error.exception.code, 'INVALID_INDEX')

    def test_bounded_context_count_does_not_replace_dense_hits(self):
        chunks = chunks_for(document(), document('other'))
        hits = [hit_for(at(chunks,1)), hit_for(at(chunks,1,'other'))]
        old = deepcopy(hits)
        context, truncated = assemble_context(hits, chunks, {('manual','v1'),('other','v1')}, TODAY)
        self.assertEqual(len(context), MAX_CONTEXT_CHUNKS)
        self.assertEqual(hits, old)
        self.assertTrue(truncated)
        self.assertLessEqual(len(_canonical(hits + context)), MAX_RESULT_CHARS)

    def test_large_neighbor_is_skipped_and_shorter_later_neighbor_can_fit(self):
        chunks = chunks_for(document(body='L'*210+'\n\nanchor\n\nx'))
        seed = hit_for(at(chunks,1))
        short = at(chunks,2)
        fields = ('chunk_id','doc_id','title','version','source','text','content_hash','kind','position')
        expected = {**{k:short[k] for k in fields}, 'score':None,
                    'retrieved_via':'adjacent_context','anchor_chunk_id':seed['chunk_id']}
        budget = len(_canonical([seed,expected]))
        with patch('customer_service.knowledge.MAX_RESULT_CHARS', budget):
            context,truncated = assemble_context([seed], chunks, {('manual','v1')}, TODAY)
        self.assertEqual([row['position'] for row in context], [2])
        self.assertTrue(truncated)
        self.assertLessEqual(len(_canonical([seed,*context])), budget)

    def test_metadata_overhead_is_counted_in_evidence_budget(self):
        chunks = chunks_for(document())
        hits = [hit_for(at(chunks,1))]
        with patch('customer_service.knowledge.MAX_RESULT_CHARS', len(_canonical(hits))):
            context,truncated = assemble_context(hits,chunks,{('manual','v1')},TODAY)
        self.assertEqual(context, [])
        self.assertTrue(truncated)


@unittest.skipUnless(importlib.util.find_spec('qdrant_client'), 'optional RAG dependencies are not installed')
class LocalQdrantContextTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='knowledge-context-')
        self.root = Path(self.temporary.name)
        self.documents = self.root/'documents.json'
        rows = [document(body='MATERIALS use warm water and a soft brush.\n\nSTEEL avoid steel wool.\n\nDRY air dry after washing.'),
                document('other',body='OTHER initial setup note.')]
        self.documents.write_text(json.dumps(rows),encoding='utf-8')
        self.index,self.cache = self.root/'index',self.root/'cache'
        self.embedder = FakeEmbedder()
        self.manifest = build_index(self.documents,self.index,self.cache,embedder=self.embedder)
        self.searcher = KnowledgeSearch(self.documents,self.index,self.cache,embedder=self.embedder)

    def tearDown(self):
        self.searcher.close()
        self.temporary.cleanup()

    def test_real_qdrant_keeps_seed_contract_and_adds_missing_lower_score_neighbor(self):
        result = self.searcher.search('clean')
        self.assertTrue(result['ok'])
        data = result['data']
        self.assertEqual(len(data['hits']), TOP_K)
        self.assertTrue(all(row['score'] >= MIN_SCORE for row in data['hits']))
        self.assertFalse(any('MATERIALS' in row['text'] for row in data['hits']))
        self.assertTrue(any('MATERIALS' in row['text'] for row in data['context']))
        self.assertTrue(all(row['score'] is None for row in data['context']))
        self.assertEqual(data['context_policy'], CONTEXT_POLICY)
        self.assertLessEqual(len(_canonical(data['hits']+data['context'])), MAX_RESULT_CHARS)
        self.assertEqual(self.manifest['retrieval'], {'top_k':3,'min_score':.5,'max_result_chars':3000})
        self.assertTrue(all('retrieved_via' not in row for row in data['hits']))

    def test_real_qdrant_no_match_has_no_context(self):
        result = self.searcher.search('unrelated')
        self.assertTrue(result['ok'])
        self.assertEqual(result['data']['status'], 'no_match')
        self.assertEqual(result['data']['hits'], [])
        self.assertEqual(result['data']['context'], [])
        self.assertFalse(result['data']['context_truncated'])

    def test_full_tool_result_including_context_reaches_next_model_messages(self):
        from customer_service.context import ContextManager, validate_tool_pairs
        from customer_service.harness import Harness
        from customer_service.shop import Shop
        runtime = self.root/'runtime'
        # Shop has no relation to the index and still exposes the original tools.
        shop_path = self.root/'shop.json'
        shop_path.write_text(json.dumps({'dataset_version':'test','customers':[],
                                        'products':{},'policies':{},'orders':{}}),encoding='utf-8')
        class Driver:
            mode = 'scripted-context-protocol-test'
            def __init__(self): self.calls=[]
            def complete(self,messages,tools,system):
                self.calls.append(deepcopy(messages))
                content = ([{'type':'tool_use','id':'context_test_call','name':'search_knowledge',
                             'input':{'query':'clean'}}] if len(self.calls)==1 else
                           [{'type':'text','text':'Scripted end; no live generation.'}])
                return {'content':content,'stop_reason':'tool_use' if len(self.calls)==1 else 'end_turn','usage':{}}
        driver=Driver()
        context=ContextManager(runtime,'guest')
        harness=Harness(Shop(None,shop_path),context,driver,enable_subagents=False,knowledge=self.searcher)
        try:
            harness.turn('How should the demo item be cleaned?')
            validate_tool_pairs(harness.messages)
            results=[b for m in driver.calls[1] if isinstance(m['content'],list)
                     for b in m['content'] if b.get('type')=='tool_result']
            body=json.loads(results[0]['content'])
            self.assertTrue(any('MATERIALS' in row['text'] for row in body['data']['context']))
            self.assertEqual(json.loads((context.session_dir/'history.json').read_text(encoding='utf-8'))['messages'],harness.messages)
        finally:
            harness.close()


if __name__ == '__main__':
    unittest.main()
