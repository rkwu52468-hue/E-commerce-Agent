"""Local knowledge administration; never imports a chat client or reads API keys."""
import argparse
import os
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get('FASTEMBED_CACHE_PATH') or (ROOT / '.cache' / 'fastembed'))


def lexical_rank(query, chunks):
    """Character-bigram cosine baseline over exactly the same indexed chunks."""
    def features(text):
        text = ''.join(re.findall(r'[\w]', text.lower()))
        return Counter(text[i:i + 2] for i in range(max(0, len(text) - 1)))
    q = features(query)
    qnorm = math.sqrt(sum(v * v for v in q.values()))
    ranked = []
    today = datetime.now(timezone(timedelta(hours=8))).date().toordinal()
    for chunk in chunks.values():
        if not chunk['effective_from_day'] <= today <= chunk['effective_to_day']:
            continue
        values = features(chunk['title'] + '\n' + chunk['text'])
        norm = math.sqrt(sum(v * v for v in values.values()))
        score = sum(v * values[k] for k, v in q.items()) / (qnorm * norm) if qnorm and norm else 0
        if score > 0:
            ranked.append((score, chunk['chunk_id'], chunk['doc_id']))
    ranked.sort(key=lambda r: (-r[0], r[1]))
    return list(dict.fromkeys(r[2] for r in ranked[:3]))


def evaluate(search, questions_path, output):
    questions = json.loads(Path(questions_path).read_text(encoding='utf-8'))
    rows = []
    for question in questions:
        started = perf_counter()
        result = search.search(question['query'])
        elapsed = (perf_counter() - started) * 1000
        if not result.get('ok'):
            raise RuntimeError('Evaluation query failed: ' + json.dumps(result, ensure_ascii=False))
        hits = result['data']['hits']
        context = result['data'].get('context', [])
        ranked = list(dict.fromkeys(h['doc_id'] for h in hits))
        expected = set(question['expected_doc_ids'])
        found = expected.intersection(ranked)
        reciprocal = next((1 / (i + 1) for i, doc in enumerate(ranked) if doc in expected), 0)
        baseline = lexical_rank(question['query'], search.chunks)
        rows.append({**question, 'retrieved_doc_ids': ranked, 'hits': hits,
                     'doc_recall_at_3': len(found) / len(expected) if expected else None,
                     'reciprocal_rank': reciprocal if expected else None,
                     'lexical_doc_ids': baseline,
                     'lexical_doc_recall_at_3': len(expected.intersection(baseline)) / len(expected) if expected else None,
                     'retrieval_ms': round(elapsed, 3),
                     'returned_any_context': bool(hits or context),
                     'context': context,
                     'main_and_context_json_chars': len(json.dumps(hits + context, ensure_ascii=False, sort_keys=True, separators=(',', ':'))),
                     'retrieval_context_policy': result['data'].get('context_policy')})
    groups = {}
    for split in ('dev', 'holdout'):
        group = [r for r in rows if r['split'] == split]
        labelled = [r for r in group if r['expected_doc_ids']]
        unsupported = [r for r in group if r['answerability'] == 'out_of_scope']
        groups[split] = {
            'questions': len(group), 'labelled_retrieval_questions': len(labelled),
            'mean_doc_recall_at_3': mean(r['doc_recall_at_3'] for r in labelled) if labelled else None,
            'document_mrr_at_3': mean(r['reciprocal_rank'] for r in labelled) if labelled else None,
            'lexical_mean_doc_recall_at_3': mean(r['lexical_doc_recall_at_3'] for r in labelled) if labelled else None,
            'out_of_scope_questions': len(unsupported),
            'out_of_scope_with_context': sum(r['returned_any_context'] for r in unsupported),
            'mean_retrieval_ms': mean(r['retrieval_ms'] for r in group) if group else None,
        }
    report = {
        'recorded_at': datetime.now(timezone.utc).isoformat(),
        'mode': 'real-local-chinese-embedding-qdrant', 'chat_api_calls': 0,
        'index_manifest': search.manifest,
        'context_metrics_note': 'Doc Recall@3 still scores dense hits only; use tools/evaluate_knowledge_context.py to measure added context evidence.',
        'keyword_baseline': 'character-bigram cosine, same chunks/top3; vector threshold and lexical nonzero rule differ',
        'question_sha256': hashlib.sha256(Path(questions_path).read_bytes()).hexdigest(),
        'question_categories': dict(Counter(r['category'] for r in rows)),
        'summary': groups, 'rows': rows,
        'limits': ['Synthetic small corpus; no production or external benchmark claim.',
                   'Recall counts expected documents among at most three returned chunks.',
                   'Context similarity does not determine answerability or answer correctness.',
                   'No chat model, answer quality, citation generation, token cost or speedup measured.',
                   'Fixed threshold/top-k; no tuning using the held-out questions.'],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'search', 'evaluate'])
    parser.add_argument('query', nargs='?')
    args = parser.parse_args()
    from .knowledge import KnowledgeSearch, build_index
    documents = ROOT / 'data/knowledge_demo/documents.json'
    indexes = ROOT / 'runtime/knowledge_demo'
    try:
        if args.action == 'build':
            manifest = build_index(documents, indexes, CACHE)
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
            return
        search = KnowledgeSearch(documents, indexes, CACHE)
        try:
            if args.action == 'search':
                if not args.query:
                    parser.error('search requires a quoted question')
                print(json.dumps(search.search(args.query), ensure_ascii=False, indent=2))
            else:
                output = ROOT / 'runtime/observations/rag-vector-retrieval-context-2026-11-06.json'
                report = evaluate(search, ROOT / 'data/knowledge_demo/evaluation.json', output)
                print(json.dumps({'summary': report['summary'], 'report': str(output),
                                  'chat_api_calls': 0}, ensure_ascii=False, indent=2))
        finally:
            search.close()
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        parser.exit(1, 'Knowledge operation failed: ' + str(error) + '\n')


if __name__ == '__main__':
    main()
