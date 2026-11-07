"""Measure delivery of frozen evidence chunks; never calls a chat model."""
import argparse
import os
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def evaluate_context(search, dataset_path, output):
    dataset_path = Path(dataset_path)
    raw = dataset_path.read_bytes()
    dataset = json.loads(raw.decode('utf-8'))
    if search.manifest['corpus_hash'] != dataset['corpus_sha256']:
        raise ValueError('The corpus differs from the frozen evaluation set')
    if search.manifest['chunks_hash'] != dataset['source_chunks_sha256']:
        raise ValueError('The chunks differ from the frozen evaluation set')
    questions = dataset['cases'] + dataset['probes'] + [dataset['known_regression']]
    for question in questions:
        for evidence in question['expected_evidence']:
            expected = search.chunks[evidence['chunk_id']]
            if any(evidence[k] != expected[k] for k in ('text','content_hash','source','version')):
                raise ValueError('Evidence no longer matches the indexed source')
    rows = []
    for question in questions:
        result = search.search(question['query'])
        if not result.get('ok'):
            raise RuntimeError('Retrieval failed: ' + canonical(result))
        hits = result['data']['hits']
        context = result['data'].get('context', [])
        expected = set(question['expected_chunk_ids'])
        dense_ids = {x['chunk_id'] for x in hits}
        all_ids = dense_ids | {x['chunk_id'] for x in context}
        rows.append({**question, 'result': result,
                     'dense_evidence_coverage': len(expected & dense_ids)/len(expected) if expected else None,
                     'expanded_evidence_coverage': len(expected & all_ids)/len(expected) if expected else None,
                     'dense_all_expected': expected <= dense_ids if expected else None,
                     'expanded_all_expected': expected <= all_ids if expected else None,
                     'dense_chunk_count': len(hits), 'expanded_chunk_count': len(hits)+len(context),
                     'dense_json_chars': len(canonical(hits)),
                     'expanded_json_chars': len(canonical(hits+context)),
                     'unlabelled_context_ids': sorted(all_ids-expected)})
    measured = rows[:len(dataset['cases'])]
    summary = {
        'new_frozen_cases': len(measured),
        'dense_mean_expected_chunk_coverage': mean(x['dense_evidence_coverage'] for x in measured),
        'expanded_mean_expected_chunk_coverage': mean(x['expanded_evidence_coverage'] for x in measured),
        'dense_all_expected_count': sum(x['dense_all_expected'] for x in measured),
        'expanded_all_expected_count': sum(x['expanded_all_expected'] for x in measured),
        'mean_dense_chunks': mean(x['dense_chunk_count'] for x in measured),
        'mean_expanded_chunks': mean(x['expanded_chunk_count'] for x in measured),
        'mean_dense_json_chars': mean(x['dense_json_chars'] for x in measured),
        'mean_expanded_json_chars': mean(x['expanded_json_chars'] for x in measured),
        'max_expanded_json_chars': max(x['expanded_json_chars'] for x in rows),
    }
    report = {'recorded_at': datetime.now(timezone.utc).isoformat(),
              'mode': 'real-local-embedding-and-qdrant-evidence-delivery',
              'question_sha256': hashlib.sha256(raw).hexdigest(),
              'dataset_frozen_at': dataset['frozen_at'], 'index_manifest': search.manifest,
              'summary': summary, 'rows': rows, 'chat_api_calls': 0,
              'limits': [
                  'Strict annotated passage delivery is not generated-answer correctness.',
                  'Expansion returns more passages under the same maximum character budget; no free or same-top-k gain claimed.',
                  'New questions were independently frozen from synthetic source before retrieval; not an external benchmark.',
                  'Unlabelled context may include useful alternative evidence or noise; not automatically false.',
                  'No-result and nonempty-result do not determine answerability for probes.',
                  'Known cleaning regression is separate from the 24 new cases.',
                  'After first scoring this dataset is regression material, not a new untouched holdout.']}
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-root', type=Path, default=ROOT/'runtime/knowledge_demo')
    parser.add_argument('--output', type=Path, default=ROOT/'runtime/observations/rag-adjacent-context-evaluation-2026-11-06.json')
    args = parser.parse_args()
    from customer_service.knowledge import KnowledgeSearch
    search = KnowledgeSearch(ROOT/'data/knowledge_demo/documents.json',args.index_root,Path(os.environ.get('FASTEMBED_CACHE_PATH') or (ROOT / '.cache' / 'fastembed')))
    try:
        report = evaluate_context(search,ROOT/'data/knowledge_demo/evaluation-context-v1.json',args.output)
        print(json.dumps({'summary':report['summary'],'report':str(args.output),'chat_api_calls':0},ensure_ascii=False,indent=2))
    finally:
        search.close()


if __name__ == '__main__':
    main()
