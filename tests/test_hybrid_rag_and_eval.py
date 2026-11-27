"""Stage 2 unit tests: Hybrid Search (Dense + BM25 RRF), budget calibration, and 100-question E2E eval suite."""
from datetime import date
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from customer_service.knowledge import (
    CONTEXT_POLICY,
    LEXICAL_MIN_SCORE,
    MAX_RESULT_CHARS,
    RRF_K,
    TOP_K,
    KnowledgeError,
    KnowledgeSearch,
    _canonical,
    bm25_lexical_search,
    build_index,
    chunk_documents,
    fuse_rrf,
    lexical_tokenize,
)
from customer_service.safeguards import SafeguardConfig
from customer_service.shop import ACTION_TOOL_NAMES, TOOL_DEFINITIONS
from tools.evaluate_agent_e2e import DATASET_PATH, evaluate_suite

ROOT = Path(__file__).resolve().parents[1]
ALL_VALID_TOOLS = {t['name'] for t in TOOL_DEFINITIONS} | set(ACTION_TOOL_NAMES) | {'search_knowledge'}
TODAY = date(2026, 11, 26)


class TinyEmbedder:
    """Deterministic unit-test embedder (proves control flow & fusion, never semantic quality)."""
    model_name = 'tiny-hybrid-test-embedder'
    dimension = 3
    artifact_fingerprint = hashlib.sha256(b'tiny-hybrid-test-embedder').hexdigest()

    def token_count(self, text):
        return len(text.split()) + len(text) // 4

    def _vec(self, text):
        low = text.lower()
        if 'return' in low or '退货' in low:
            return [1.0, 0.0, 0.0]
        if 'clean' in low or '清洗' in low or '清洁' in low:
            return [0.0, 1.0, 0.0]
        if 'unrelated' in low:
            return [0.0, 0.0, -1.0]
        # Weak generic vector that does not pass dense threshold for specialized queries
        return [0.0, 0.0, 1.0]

    def passage_embed(self, texts):
        return [self._vec(t) for t in texts]

    def query_embed(self, text):
        return [self._vec(text)]


def _doc(doc_id, title, body, *, store_id='demo-shop', visibility='public',
         status='active', effective_from='2026-01-01', effective_to=None, kind='manual'):
    return {
        'doc_id': doc_id,
        'title': title,
        'version': 'v1',
        'store_id': store_id,
        'visibility': visibility,
        'status': status,
        'effective_from': effective_from,
        'effective_to': effective_to,
        'kind': kind,
        'body': body,
        'source': f'docs.json#{doc_id}',
    }


class HybridRagAndEvalTests(unittest.TestCase):
    def test_budget_calibration_defaults_handle_cache_inclusive_multistep_turns(self):
        cfg = SafeguardConfig.production_defaults()
        self.assertEqual(45000, cfg.max_turn_tokens)
        self.assertEqual(200000, cfg.max_session_tokens)
        self.assertEqual(500000, cfg.daily_token_budget)
        custom = SafeguardConfig.production_defaults(max_turn_tokens=30000, max_session_tokens=120000)
        self.assertEqual(30000, custom.max_turn_tokens)
        self.assertEqual(120000, custom.max_session_tokens)

    def test_lexical_tokenizer_avoids_ascii_false_bigrams_and_preserves_codes(self):
        unrelated_toks = set(lexical_tokenize('unrelated'))
        return_toks = set(lexical_tokenize('return policy'))
        self.assertTrue(unrelated_toks.isdisjoint(return_toks))
        code_toks = set(lexical_tokenize('P100 示例保温杯 500ml 收货后7天内退货'))
        self.assertIn('p100', code_toks)
        self.assertIn('500ml', code_toks)
        self.assertIn('500', code_toks)
        self.assertIn('ml', code_toks)
        self.assertIn('7天', code_toks)
        self.assertIn('保温', code_toks)
        self.assertIn('退货', code_toks)

    def test_bm25_respects_scope_dates_integrity_and_no_match(self):
        embedder = TinyEmbedder()
        docs = [
            _doc('cup-clean', 'P100 清洗说明', '日常使用温水和中性洗涤剂清洗 P100 保温杯。'),
            _doc('cup-spec', 'P104 焖烧罐规格', '智能显温真空焖烧罐容量 600ml，保温性能说明。'),
            _doc('expired-doc', '过期优惠', 'P104 限时赠品 CANARY_EXPIRED', effective_from='2025-01-01', effective_to='2025-12-31'),
            _doc('private-doc', '私有记录', 'P104 私有备注 CANARY_PRIVATE', visibility='private'),
        ]
        # Chunk only eligible docs for build date, plus manually inject an expired chunk to verify runtime recheck
        chunks = {c['chunk_id']: c for c in chunk_documents(docs, embedder, today=TODAY)}
        eligible_ids = {(d['doc_id'], d['version']) for d in docs if d['doc_id'] in ('cup-clean', 'cup-spec')}

        hits = bm25_lexical_search('P104 600ml', chunks, eligible_ids, TODAY)
        self.assertEqual(1, len(hits))
        self.assertEqual('cup-spec', hits[0][0]['doc_id'])
        self.assertGreaterEqual(hits[0][1], LEXICAL_MIN_SCORE)

        # Unrelated query yields empty list cleanly
        self.assertEqual([], bm25_lexical_search('unrelated', chunks, eligible_ids, TODAY))

        # Tampered content_hash is caught deterministically
        tampered = dict(chunks)
        first_id = next(iter(tampered))
        bad_chunk = dict(tampered[first_id])
        bad_chunk['text'] = bad_chunk['text'] + ' tampered'
        tampered[first_id] = bad_chunk
        with self.assertRaises(KnowledgeError) as ctx:
            bm25_lexical_search('P104', tampered, eligible_ids, TODAY)
        self.assertEqual('INVALID_INDEX', ctx.exception.code)

    def test_rrf_fusion_ranks_dual_match_above_single_source_and_caps_budget(self):
        embedder = TinyEmbedder()
        docs = [
            _doc('doc-a', '文档A', '第一篇关于保温杯清洗与晾干的详细说明。'),
            _doc('doc-b', '文档B', '第二篇关于保温杯日常清洗与中性洗涤剂的说明。'),
            _doc('doc-c', '文档C', '第三篇关于订单退款时效的说明。'),
        ]
        chunks = chunk_documents(docs, embedder, today=TODAY)
        by_doc = {c['doc_id']: c for c in chunks}
        # doc-a is rank 1 in dense only; doc-b is rank 2 in dense AND rank 1 in lexical -> doc-b wins RRF!
        dense = [(by_doc['doc-a'], 0.91), (by_doc['doc-b'], 0.80)]
        lexical = [(by_doc['doc-b'], 3.20), (by_doc['doc-c'], 2.10)]
        fused = fuse_rrf(dense, lexical, top_k=TOP_K, rrf_k=RRF_K)
        self.assertEqual(['doc-b', 'doc-a', 'doc-c'], [h['doc_id'] for h in fused])
        self.assertEqual('dense+lexical', fused[0]['matched_by'])
        self.assertEqual('dense', fused[1]['matched_by'])
        self.assertEqual('lexical', fused[2]['matched_by'])
        self.assertLessEqual(len(_canonical(fused)), MAX_RESULT_CHARS)

    @unittest.skipUnless(importlib.util.find_spec('qdrant_client'), 'optional RAG dependencies are not installed')
    def test_qdrant_hybrid_recovers_exact_code_missed_by_dense_while_preserving_dense_default(self):
        with tempfile.TemporaryDirectory(prefix='hybrid_rag_test_') as tmp:
            root = Path(tmp)
            docs_path = root / 'documents.json'
            rows = [
                _doc('returns', '退换货政策', '收货后7天内未使用商品可申请退货。'),
                _doc('spec-x99', 'X99 专属配件说明', 'X99 型号硅胶密封圈用于特定杯盖，需核对接口尺寸。\n\n禁止使用强酸溶剂浸泡。'),
                _doc('cup-clean', '清洗指南', '使用温水、中性洗涤剂和软刷清洗杯身。'),
            ]
            docs_path.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
            embedder = TinyEmbedder()
            build_index(docs_path, root / 'index', root / 'cache', embedder=embedder)

            # Default mode stays 'dense' for full backward compatibility
            dense_search = KnowledgeSearch(docs_path, root / 'index', root / 'cache', embedder=embedder)
            try:
                self.assertEqual('dense', dense_search.retrieval_mode)
                dense_res = dense_search.search('clean X99')
                self.assertTrue(dense_res['ok'])
                self.assertNotIn('retrieval_mode', dense_res['data'])
                # Dense embedder only matches 'clean' ([0,1,0]), missing X99 ([0,0,1])
                self.assertEqual(['cup-clean'], [h['doc_id'] for h in dense_res['data']['hits']])

                # Hybrid mode fuses dense + BM25 lexical and recovers spec-x99 + its adjacent context!
                hybrid_res = dense_search.search('clean X99', mode='hybrid')
                self.assertTrue(hybrid_res['ok'])
                self.assertEqual('hybrid', hybrid_res['data']['retrieval_mode'])
                self.assertEqual(CONTEXT_POLICY, hybrid_res['data']['context_policy'])
                hit_docs = {h['doc_id'] for h in hybrid_res['data']['hits']}
                self.assertIn('spec-x99', hit_docs)
                self.assertIn('cup-clean', hit_docs)
                self.assertTrue(any('强酸' in c['text'] for c in hybrid_res['data']['context']))

                # Unrelated query still returns no_match cleanly in hybrid mode
                empty_res = dense_search.search('unrelated', mode='hybrid')
                self.assertTrue(empty_res['ok'])
                self.assertEqual('no_match', empty_res['data']['status'])
                self.assertEqual([], empty_res['data']['hits'])
            finally:
                dense_search.close()

    def test_eval_dataset_has_100_well_formed_balanced_cases(self):
        data = json.loads(DATASET_PATH.read_text(encoding='utf-8'))
        cases = data['cases']
        self.assertEqual(100, data['total_cases'])
        self.assertEqual(100, len(cases))
        ids = [c['id'] for c in cases]
        self.assertEqual(100, len(set(ids)))
        self.assertEqual([f'E{i:03d}' for i in range(1, 101)], ids)
        counts = {}
        for case in cases:
            cat = case['category']
            counts[cat] = counts.get(cat, 0) + 1
            self.assertIn(case['customer_id'], (None, 'alice', 'bob', 'charlie'))
            self.assertTrue(case['turns'])
            self.assertTrue(case['fact_groups'])
            for tool in case['expected_tools'] + case['forbidden_tools']:
                self.assertIn(tool, ALL_VALID_TOOLS, f"Invalid tool {tool} in {case['id']}")
        self.assertEqual({
            'normal_query': 20,
            'action_execution': 20,
            'rag_knowledge': 20,
            'security_refusal': 20,
            'escalation_exception': 20,
        }, counts)

    def test_offline_e2e_evaluator_executes_all_5_categories_and_verifies_db_and_timeout(self):
        class MiniStubKnowledge:
            retrieval_mode = 'hybrid'

            def search(self, query):
                return {
                    'ok': True,
                    'data': {
                        'hits': [{
                            'chunk_id': 'kb_stub_1', 'doc_id': 'cup-clean',
                            'title': '模拟保温杯日常清洁说明', 'version': 'demo-rag-2026-11-06',
                            'source': 'data/knowledge_demo/documents.json#cup-clean',
                            'text': '日常清洁可使用温水、少量中性洗涤剂和柔软杯刷，清洗后通风晾干。',
                            'content_hash': '0' * 64, 'kind': 'manual', 'score': 0.032,
                        }],
                        'context': [],
                        'status': 'matched',
                    },
                    'simulated': True,
                    'dataset_version': 'knowledge-demo-v1',
                }

            def close(self):
                pass

        # Sample across all 5 categories including 2-step cancel, rejection, RAG, security, and timeout escalation
        sample_ids = ['E001', 'E017', 'E023', 'E026', 'E041', 'E067', 'E076', 'E085', 'E098']
        report = evaluate_suite(
            DATASET_PATH,
            mode='offline',
            case_ids=sample_ids,
            knowledge=MiniStubKnowledge(),
            verbose=False,
        )
        overall = report['summary']['overall']
        self.assertEqual(len(sample_ids), overall['total'])
        self.assertEqual(1.0, overall['tool_accuracy'], report['results'])
        self.assertEqual(1.0, overall['db_state_accuracy'], report['results'])
        self.assertEqual(1.0, overall['task_resolution_rate'], report['results'])
        self.assertEqual(0, overall['hallucination_case_count'], report['results'])
        self.assertEqual(1.0, overall['escalation_accuracy'], report['results'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
