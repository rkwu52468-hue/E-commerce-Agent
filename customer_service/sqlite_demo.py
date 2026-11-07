"""Scripted business flow over a real SQLite engine and synthetic records."""
import copy
import json
import sqlite3
from pathlib import Path
from .context import ContextManager, validate_tool_pairs
from .demos import call, save_report, text
from .harness import Harness
from .models import ScriptedModel
from .shop import Shop
from .storage import initialize_database


def run_sqlite_demo(root):
    root = Path(root)
    context = ContextManager(root / 'runtime', 'alice')
    database = context.session_dir / 'shop.db'
    initialized = initialize_database(database, root / 'data/shop-sqlite-seed.json')
    shop = Shop('alice', root / 'data/shop.json', database_path=database)
    observed = {}

    def explain_prices(messages):
        results = {block['tool_use_id']: json.loads(block['content'])
                   for block in messages[-1]['content'] if block.get('type') == 'tool_result'}
        assert all(item['ok'] for item in results.values())
        observed['business_results'] = copy.deepcopy(results)
        product = results['sqlite_product_1']['data']
        order = results['sqlite_order_1']['data']
        listing = results['sqlite_list_1']['data']['orders']
        answer = ('商品当前售价' + format(product['price_cents'] / 100, '.2f')
                  + '元（来源：' + product['source'] + '）；'
                  + order['order_id'] + '历史订单成交总额'
                  + format(order['total_cents'] / 100, '.2f')
                  + '元（来源：' + order['source'] + '）。'
                  + '当前商品价不会改写历史订单金额。本人订单：'
                  + ('、'.join(item['order_id'] for item in listing) if listing else '无')
                  + '；查询范围由程序绑定的模拟客户身份限定。')
        observed['price_answer'] = answer
        return [text(answer)]

    def explain_denial(messages):
        output = json.loads(messages[-1]['content'][0]['content'])
        observed['foreign_order_result'] = copy.deepcopy(output)
        assert not output['ok']
        return [text('订单不存在或不属于当前模拟客户，无法提供该订单信息。')]

    model = ScriptedModel([
        [call('sqlite_product_1', 'get_product', product_id='P100'),
         call('sqlite_order_1', 'get_order', order_id='A1001'),
         call('sqlite_list_1', 'list_my_orders')],
        explain_prices,
        [call('sqlite_foreign_1', 'get_order', order_id='B2001')],
        explain_denial,
    ])
    harness = Harness(shop, context, model)
    for query in ['查询P100当前价格、我的A1001订单金额，并列出我的订单。', '请再查询B2001订单。']:
        print('客户：' + query)
        print('客服（离线脚本）：' + harness.turn(query))
    validate_tool_pairs(harness.messages)

    # Open a separate connection after tool execution, so this is persistent SQLite data.
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        engine_version = connection.execute('SELECT sqlite_version()').fetchone()[0]
    reopened_shop = Shop('alice', root / 'data/shop.json', database_path=database)
    reopened_product = reopened_shop.call('get_product', {'product_id': 'P100'})
    reopened_order = reopened_shop.call('get_order', {'order_id': 'A1001'})
    results = observed['business_results']
    product = results['sqlite_product_1']['data']
    order = results['sqlite_order_1']['data']
    listing = results['sqlite_list_1']['data']['orders']
    checks = {
        'real_sqlite_file_header': database.read_bytes().startswith(b'SQLite format 3\x00'),
        'real_sqlite_engine_reconnected': bool(engine_version),
        'product_current_price_19900': product['price_cents'] == 19900,
        'historical_order_total_16900': order['total_cents'] == 16900,
        'current_product_and_historical_order_prices_are_separate': product['price_cents'] != order['total_cents'],
        'own_list_filtered_before_model': {item['order_id'] for item in listing} == {'A1001'},
        'foreign_order_denied': observed['foreign_order_result'].get('error', {}).get('code') == 'ACCESS_DENIED_OR_NOT_FOUND',
        'reconnection_preserves_product': reopened_product == results['sqlite_product_1'],
        'reconnection_preserves_order': reopened_order == results['sqlite_order_1'],
        'complete_tool_batches_valid': True,
        'four_scripted_model_responses': model.steps == 4,
    }
    assert all(checks.values()), checks
    report = {
        'mode': model.mode, 'api_calls': 0, 'scripted_response_steps': model.steps,
        'database_path': str(database), 'database_initialization': initialized,
        'sqlite_engine_version': engine_version, 'observed': observed, 'checks': checks,
        'session_dir': str(context.session_dir),
        'limitation': 'Actual SQLite engine and synthetic business records with deterministic scripted replies. No live model/API call, independent student coding, production authentication or refund submission.'}
    (context.session_dir / 'sqlite-report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return save_report(root, 'sqlite-latest.json', report)
