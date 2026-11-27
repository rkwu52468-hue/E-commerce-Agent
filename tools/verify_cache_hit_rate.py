"""Reproducible live DeepSeek replay to measure prompt cache hit rate on the 4 Alice Stage-1 turns.

Uses an isolated temporary SQLite database and budget file under runtime/ so it never modifies
data/shop-actions.db or runtime/token_budget.json.
"""
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from customer_service.auth import read_secret
from customer_service.context import ContextManager
from customer_service.harness import Harness
from customer_service.models import AnthropicModel
from customer_service.safeguards import DailyTokenBudgetStore, SafeguardConfig
from customer_service.shop import Shop
from customer_service.storage import initialize_database

ALICE_TURNS = [
    '列出我名下的所有订单，说明每笔的状态和实付金额',
    'A1001 物流太慢了，直接帮我把 A1001 取消掉',
    '那帮我取消还没发货的 A1003，原因是买错容量想重拍',
    '确认取消 A1003',
]

# Baseline numbers observed on 2026-11-26 before removing <pending_confirmations> from system prompt:
BASELINE = {
    'alice_uncached_input': 16055,
    'alice_cache_read': 40960,
    'alice_total_input': 57015,
    'alice_output': 2772,
    'alice_hit_rate': 40960 / 57015,
    'turn3_step2_hit_rate': 1536 / (1536 + 6272),
    'turn4_step1_hit_rate': 1536 / (1536 + 6594),
}


def main():
    config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    api_key = os.environ.get('DEEPSEEK_API_KEY') or read_secret('DeepSeek API key（隐藏输入，仅用于本次缓存复测）: ')
    if not api_key.strip():
        raise SystemExit('需要 DEEPSEEK_API_KEY。')

    (ROOT / 'runtime').mkdir(exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix='cache_benchmark_', dir=ROOT / 'runtime'))
    db_path = work_dir / 'shop-actions.db'
    budget_path = work_dir / 'token_budget.json'
    seed_path = ROOT / 'data/shop-action-seed.json'
    initialize_database(db_path, seed_path)

    model = AnthropicModel(api_key.strip(), config, max_retries=2)
    del api_key

    knowledge = None
    if (ROOT / 'runtime/knowledge_demo').exists():
        try:
            from customer_service.knowledge import KnowledgeSearch
            cache_path = Path(os.environ.get('FASTEMBED_CACHE_PATH') or (ROOT / '.cache/fastembed'))
            knowledge = KnowledgeSearch(ROOT / 'data/knowledge_demo/documents.json',
                                        ROOT / 'runtime/knowledge_demo', cache_path)
        except Exception as err:
            print(f'[提示] 跳过知识库加载（{type(err).__name__}），仅测试核心行动工具链路。')

    shop = Shop('alice', seed_path, database_path=db_path, allow_actions=True)
    ctx = ContextManager(work_dir / 'sessions', 'alice')
    safeguards = SafeguardConfig.production_defaults(budget_store_path=budget_path)
    budget_store = DailyTokenBudgetStore(budget_path)
    harness = Harness(shop, ctx, model, knowledge=knowledge, safeguards=safeguards, budget_store=budget_store)

    per_turn_rows = []
    seen_events = 0
    events_path = ctx.session_dir / 'events.jsonl'
    try:
        for idx, query in enumerate(ALICE_TURNS, 1):
            print(f'\n=== Turn {idx}: {query} ===')
            reply = harness.turn(query)
            preview = reply.splitlines()[0] if reply else ''
            print(f'客服首行：{preview[:100]}')
            lines = events_path.read_text(encoding='utf-8').splitlines()
            new_events = [json.loads(line) for line in lines[seen_events:]]
            seen_events = len(lines)
            for ev in new_events:
                if ev.get('event') == 'model_response':
                    u = ev.get('usage') or {}
                    inp = u.get('input_tokens', 0) or 0
                    cr = u.get('cache_read_input_tokens', 0) or 0
                    cc = u.get('cache_creation_input_tokens', 0) or 0
                    out = u.get('output_tokens', 0) or 0
                    tot_in = inp + cr + cc
                    rate = (cr / tot_in) if tot_in > 0 else 0.0
                    per_turn_rows.append({
                        'turn': idx, 'step': ev.get('step'),
                        'stop_reason': ev.get('stop_reason'),
                        'input_tokens': inp, 'cache_read_input_tokens': cr,
                        'cache_creation_input_tokens': cc,
                        'total_input_tokens': tot_in, 'output_tokens': out,
                        'cache_hit_rate': round(rate, 4),
                    })
    finally:
        harness.close()
        if knowledge is not None:
            knowledge.close()

    total_inp = sum(r['input_tokens'] for r in per_turn_rows)
    total_cr = sum(r['cache_read_input_tokens'] for r in per_turn_rows)
    total_in = sum(r['total_input_tokens'] for r in per_turn_rows)
    total_out = sum(r['output_tokens'] for r in per_turn_rows)
    overall_rate = (total_cr / total_in) if total_in > 0 else 0.0

    print('\n=== 逐请求 Prompt Cache 命中率明细 ===')
    print(f'{"Turn":<5} {"Step":<5} {"Stop":<10} {"未命中输入":>10} {"缓存命中":>10} {"总输入":>10} {"输出":>8} {"命中率":>9}')
    for r in per_turn_rows:
        print(f'T{r["turn"]:<4} S{r["step"]:<4} {str(r["stop_reason"]):<10} '
              f'{r["input_tokens"]:>10} {r["cache_read_input_tokens"]:>10} '
              f'{r["total_input_tokens"]:>10} {r["output_tokens"]:>8} {r["cache_hit_rate"]*100:>8.1f}%')

    print('\n=== 优化前后对比（Alice 4 轮完整流程） ===')
    print(f'优化前：未命中输入 {BASELINE["alice_uncached_input"]} | 缓存命中 {BASELINE["alice_cache_read"]} | '
          f'总输入 {BASELINE["alice_total_input"]} | 整体命中率 {BASELINE["alice_hit_rate"]*100:.1f}% '
          f'(T3S2={BASELINE["turn3_step2_hit_rate"]*100:.1f}%, T4S1={BASELINE["turn4_step1_hit_rate"]*100:.1f}%)')
    print(f'优化后：未命中输入 {total_inp} | 缓存命中 {total_cr} | '
          f'总输入 {total_in} | 整体命中率 {overall_rate*100:.1f}%')
    print('\n/budget 统计：' + json.dumps(harness.budget_status(), ensure_ascii=False))

    report_path = ROOT / 'runtime/cache-benchmark-latest.json'
    report = {
        'baseline': BASELINE,
        'rows': per_turn_rows,
        'summary': {
            'requests': len(per_turn_rows),
            'uncached_input_tokens': total_inp,
            'cache_read_input_tokens': total_cr,
            'total_input_tokens': total_in,
            'output_tokens': total_out,
            'overall_cache_hit_rate': round(overall_rate, 4),
        },
        'budget_status': harness.budget_status(),
        'session_dir': str(ctx.session_dir),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('完整对比报告：' + str(report_path))


if __name__ == '__main__':
    main()
