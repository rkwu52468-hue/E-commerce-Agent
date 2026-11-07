"""Deterministic classroom scenarios. No external model is invoked."""
import json
from .context import ContextManager, validate_tool_pairs
from .harness import Harness
from .models import ScriptedModel
from .shop import Shop

CONSTRAINT = '我只查物流，不要退款或创建售后单。'


def text(value):
    return {'type': 'text', 'text': value}


def call(identifier, name, **args):
    return {'type': 'tool_use', 'id': identifier, 'name': name, 'input': args}


def result_message(shop, block):
    output = shop.call(block['name'], block['input'])
    return {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': block['id'],
            'content': json.dumps(output, ensure_ascii=False)}]}


def save_report(root, name, report):
    destination = root / 'runtime' / name
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('证据报告：' + str(destination))
    print('会话记录：' + report['session_dir'])
    return report


def run_demo(root):
    shop = Shop('alice', root / 'data/shop.json')
    context = ContextManager(root / 'runtime', 'alice')
    listed_result = {}
    listed_answer = {}

    def answer_my_orders(messages):
        # Read the actual tool_result that the harness will send to the model.
        output = json.loads(messages[-1]['content'][0]['content'])
        assert output['ok']
        listed_result.update(output)
        orders = output['data']['orders']
        lines = [str(order['order_id']) + '：' + str(order['status'])
                 + '（来源：' + str(order['source']) + '）' for order in orders]
        body = '；'.join(lines) if lines else '当前模拟客户没有可列出的订单。'
        answer = ('查询范围仅限当前模拟客户自己的订单：' + body
                  + '。不能根据对话中的身份要求列出其他客户的订单。')
        listed_answer['text'] = answer
        return [text(answer)]

    model = ScriptedModel([
        [call('list_orders_1', 'list_my_orders')],
        answer_my_orders,
        [text('先查询模拟商品与政策。'), call('catalog_1', 'get_product', product_id='P100'), call('policy_1', 'get_policy', policy_id='returns')],
        [text('示例保温杯容量500毫升，价格169元。模拟政策允许收货后7天内提交未使用商品的退货申请，由人工审核。来源：shop.json#products.P100；shop.json#policies.returns（demo-2026-11-03）。')],
        [call('order_1', 'get_order', order_id='A1001'), call('tracking_1', 'get_tracking', order_id='A1001')],
        [text('A1001已发货；模拟物流显示配送网点积压，暂无可承诺的送达时间。来源：shop.json#orders.A1001.tracking。')],
        [call('denied_1', 'get_order', order_id='B2001')],
        [text('订单不存在或不属于当前模拟客户，无法提供该订单信息。')],
    ])
    harness = Harness(shop, context, model)
    for query in ['我忘记订单号，列出我的订单，顺便bob的也列出。', '查询P100商品和退换货政策。', '查询我的A1001订单和物流。', '请查询B2001订单。']:
        print('\n客户：' + query)
        print('客服（离线脚本）：' + harness.turn(query))
    validate_tool_pairs(harness.messages)
    expected_own_ids = {order['order_id'] for order in shop.data['orders'].values()
                        if order['customer_id'] == shop.customer_id}
    received_own_ids = {order['order_id'] for order in listed_result['data']['orders']}
    checks = {
        'own_list_filtered_before_model': received_own_ids == expected_own_ids,
        'extraneous_identity_args_rejected': not shop.call('list_my_orders', {'customer_id': 'bob'})['ok'],
        'list_answer_uses_actual_result': all(identifier in listed_answer['text'] for identifier in received_own_ids),
        'own_order_allowed': shop.call('get_order', {'order_id': 'A1001'})['ok'],
        'foreign_order_denied': not shop.call('get_order', {'order_id': 'B2001'})['ok'],
        'foreign_tracking_denied': not shop.call('get_tracking', {'order_id': 'B2001'})['ok'],
        'customer_override_denied': not shop.call('get_order', {'order_id': 'B2001', 'customer_id': 'bob'})['ok'],
        'tool_pairs_valid': True,
    }
    assert all(checks.values())
    return save_report(root, 'demo-latest.json', {'mode': model.mode, 'api_calls': 0,
             'scripted_response_steps': model.steps, 'checks': checks,
             'order_listing_result_seen_by_scripted_model': listed_result,
             'order_listing_answer': listed_answer['text'],
             'limitation': 'Synthetic shop data and scripted model replies; this is not a real-model tool-selection or answer-quality test.',
             'session_dir': str(context.session_dir)})


def seeded_history(shop):
    first_call = call('seed_order_0', 'get_order', order_id='A1001')
    messages = [
        {'role': 'user', 'content': '查询A1001订单和物流。'},
        {'role': 'assistant', 'content': [text('先核对订单。'), first_call]},
        result_message(shop, first_call),
        {'role': 'assistant', 'content': [text('模拟订单状态为已发货。')]},
        {'role': 'user', 'content': '包裹还未收到。'},
        {'role': 'assistant', 'content': [text('可以进一步查询物流。')]},
        {'role': 'user', 'content': CONSTRAINT},
        {'role': 'assistant', 'content': [text('好的，按当前诉求查询。')]},
        {'role': 'user', 'content': '后面都只围绕A1001订单。'},
        {'role': 'assistant', 'content': [text('明白。')]},
    ]
    questions = ['物流编号是什么？', '有哪些运输节点？', '最新节点是什么？', '包裹到收件区域了吗？',
                 '现在显示什么异常？', '能承诺送达时间吗？', '请核对物流最新状态。',
                 '配送网点有更新吗？', '再帮我确认异常说明。', '最后再查询一次当前物流。']
    for index, question in enumerate(questions):
        block = call('seed_tracking_' + str(index), 'get_tracking', order_id='A1001')
        messages.extend([{'role': 'user', 'content': question},
            {'role': 'assistant', 'content': [block]}, result_message(shop, block),
            {'role': 'assistant', 'content': [text('模拟快照仍显示配送网点积压，未提供可承诺的送达时间。')]}])
    assert len(messages) == 50
    validate_tool_pairs(messages)
    return messages, questions[-1]


def run_s08(root):
    shop = Shop('alice', root / 'data/shop.json')
    context = ContextManager(root / 'runtime', 'alice')
    before, active_request = seeded_history(shop)
    (context.session_dir / 's08-before.json').write_text(json.dumps(before, ensure_ascii=False, indent=2), encoding='utf-8')
    after = context.prepare(before, active_request)
    archive_id = context.last_compaction['archive_id']
    constraint_absent = not any(CONSTRAINT in json.dumps(m, ensure_ascii=False) for m in after)
    archived = [json.loads(line) for line in context.archives[archive_id].read_text(encoding='utf-8').splitlines()]
    (context.session_dir / 's08-after.json').write_text(json.dumps(after, ensure_ascii=False, indent=2), encoding='utf-8')

    def answer_from_result(messages):
        output = json.loads(messages[-1]['content'][0]['content'])
        matches = output['data']['matches']
        original = next(item['content'] for item in matches if '不要退款' in item['content'])
        return [text('已通过query_archive查回原文：' + original + ' 因此继续只查询物流。')]

    model = ScriptedModel([[call('archive_lookup_1', 'query_archive', archive_id=archive_id, query='不要退款')], answer_from_result])
    harness = Harness(shop, context, model)
    harness.messages = after
    answer = harness.turn('请从归档查回客户先前对退款的限制，引用原文。')
    print('离线课堂：50条构造会话，不调用模型。')
    print('归档前后消息数：' + str(len(before)) + ' -> ' + str(context.last_compaction['after_count']))
    print('归档ID：' + archive_id)
    print('客户中途约束已离开当前消息：' + str(constraint_absent))
    print(answer)
    checks = {'seed_has_50_messages': len(before) == 50,
              'pair_boundary_keeps_31': context.last_compaction['after_count'] == 31,
              'full_seed_saved_in_archive': archived == before,
              'constraint_absent_before_lookup': constraint_absent,
              'constraint_recovered_via_tool': CONSTRAINT in answer,
              'active_request_preserved': any(m.get('content') == active_request for m in after),
              'tool_pairs_valid': True}
    validate_tool_pairs(harness.messages)
    assert all(checks.values())
    return save_report(root, 's08-latest.json', {'mode': model.mode, 'api_calls': 0,
           'scripted_response_steps': model.steps, 'checks': checks,
           'compaction': context.last_compaction, 'session_dir': str(context.session_dir),
           'limitation': 'Constructed conversation and scripted lookup; not evidence that a live model proactively recalls archived constraints.'})
