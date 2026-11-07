"""Read-only business tools over explicitly synthetic shop data."""
import copy
import json
from pathlib import Path
from .storage import SQLiteStore, StoreError

# s02 adaptation: define tool inputs separately from business execution.
ARGUMENTS = {
    'get_product': ('product_id',), 'get_policy': ('policy_id',),
    'get_order': ('order_id',), 'get_tracking': ('order_id',),
    'list_my_orders': (),
}
DESCRIPTIONS = {
    'get_product': '查询模拟商品资料，回答时保留来源。price_cents单位是分；19900分显示为199.00元（CNY）。',
    'get_policy': '查询模拟店铺政策及版本，不编造政策。',
    'get_order': '查询已登录模拟客户自己的订单；未登录需在终端/login。身份由程序绑定。total_cents单位是分；16900分显示为169.00元（CNY）。',
    'get_tracking': '查询已登录模拟客户自己的订单物流；未登录需在终端/login。',
    'list_my_orders': '无需订单号，列出已登录模拟客户本人的订单摘要；未登录需在终端/login。身份由程序绑定，不接受客户身份参数，不返回他人订单。total_cents单位是分，显示元须除以100。',
}
TOOL_DEFINITIONS = [
    {'name': name, 'description': DESCRIPTIONS[name], 'input_schema': {
        'type': 'object', 'properties': {argument: {'type': 'string'} for argument in arguments},
        'required': list(arguments), 'additionalProperties': False,
    }} for name, arguments in ARGUMENTS.items()
]


MONEY_SYSTEM = ('金额字段total_cents和price_cents的单位是分，不是元；显示CNY元金额时除以100并保留两位小数。'
                '例如total_cents=16900、currency=CNY应写169.00元（169.00 CNY），不能写16900 CNY。'
                '引用工具原始整数时须明确标注分；父模型也须按实际业务证据核对子报告的金额、币种和单位。'
                '订单历史实付与商品当前标价是不同时间的业务记录，不能用现价改写历史实付。'
                '仅凭两者金额不同，不得认定或预设差异来自优惠券、促销折扣、运费、价格变动等任何具体原因。'
                '仅当业务工具明确提供与该订单对应的原因及来源时，才可按证据解释；'
                '用户猜测、知识库一般规则、数值相减和子报告推测都不能替代该订单的事实依据。'
                '比较后若原因缺失，请表述“当前标价与历史实付不同，现有资料未提供差异原因，无法判断”，'
                '不要说“具体来自哪种优惠”等预设原因的句子。'
                '用户只问历史实付时，不为解释差异额外查询或比较现价；需要商品其他资料时仍可查询。')


PRIVATE_TOOLS = frozenset({'get_order', 'get_tracking', 'list_my_orders'})


def authentication_required():
    return failure('AUTHENTICATION_REQUIRED', '请先在终端输入/login完成本地身份验证；不要在聊天中提供密码或验证码。')


def failure(code, message):
    return {'ok': False, 'error': {'code': code, 'message': message}}


class Shop:
    def __init__(self, customer_id: str | None, data_path: Path, database_path: Path | None = None):
        self.data = json.loads(Path(data_path).read_text(encoding='utf-8'))
        self.store = SQLiteStore(database_path) if database_path is not None else None
        # None is the anonymous principal. A customer ID must be bound by the
        # trusted caller (LiveSession after verification; old lessons simulate it).
        if customer_id is not None:
            if not isinstance(customer_id, str):
                raise ValueError('Unknown simulated customer')
            known = self.store.has_customer(customer_id) if self.store is not None else customer_id in self.data['customers']
            if not known:
                raise ValueError('Unknown simulated customer')
        self.customer_id = customer_id

    def tool_definitions(self):
        # The model chooses IDs from this Shop's actual public catalog, not guesses.
        # Copy templates so per-session fixtures never change another session.
        tools = copy.deepcopy(TOOL_DEFINITIONS)
        policy = next(tool for tool in tools if tool['name'] == 'get_policy')
        records = self.data['policies']
        identifiers = sorted(records)
        catalog = [{'policy_id': key, 'title': records[key].get('title', '')}
                   for key in identifiers]
        if identifiers:
            policy['input_schema']['properties']['policy_id']['enum'] = identifiers
            policy['description'] += ('当前已登记政策目录（仅用于选择ID，正文须调用工具读取）：'
                                      + json.dumps(catalog, ensure_ascii=False)
                                      + '。仅查询目录内ID，不反复猜测其他编号；目录中没有的政策需说明资料缺口。')
        else:
            # Empty enum is invalid for some model APIs; expose the missing catalog.
            policy['description'] += '当前未登记任何政策，请说明资料缺口，不猜测政策编号。'
        return tools

    def _database_success(self, record):
        return {'ok': True, 'data': record, 'simulated': True,
                'dataset_version': self.store.dataset_version, 'storage': 'sqlite'}

    def _call_database(self, name, identifier):
        try:
            if name == 'get_product':
                record = self.store.get_product(identifier)
                if record is None:
                    return failure('NOT_FOUND', '模拟商品资料中没有对应记录。')
                return self._database_success(record)
            record = self.store.get_order(identifier, self.customer_id)
            if record is None:
                return failure('ACCESS_DENIED_OR_NOT_FOUND', '订单不存在或不属于当前模拟客户。')
            if name == 'get_order':
                return self._database_success(record)
            # DB authorizes the order; logistics facts still come from the JSON fixture.
            json_order = self.data['orders'].get(identifier)
            if json_order is None or 'tracking' not in json_order:
                return failure('NOT_FOUND', '当前JSON模拟资料中没有该订单的物流记录。')
            tracking = {'order_id': identifier, **json_order['tracking'],
                        'source': json_order['source'] + '.tracking'}
            return {'ok': True, 'data': copy.deepcopy(tracking), 'simulated': True,
                    'dataset_version': self.data['dataset_version']}
        except StoreError:
            return failure('DATA_STORE_UNAVAILABLE', '模拟数据库查询失败，尚未取得结果，请排查后重试。')

    def list_my_orders(self):
        if self.customer_id is None:
            return authentication_required()
        if self.store is not None:
            try:
                return self._database_success({'orders': self.store.list_orders(self.customer_id)})
            except StoreError:
                return failure('DATA_STORE_UNAVAILABLE', '模拟数据库查询失败，尚未取得订单列表，请排查后重试。')
        # s03 business adaptation: filter ownership before building tool_result.
        fields = ('order_id', 'status', 'total_cents', 'currency', 'source')
        orders = [
            {field: copy.deepcopy(order[field]) for field in fields}
            for order in sorted(self.data['orders'].values(), key=lambda item: item['order_id'])
            if order['customer_id'] == self.customer_id
        ]
        return {'ok': True, 'data': {'orders': orders}, 'simulated': True,
                'dataset_version': self.data['dataset_version']}

    def call(self, name, args):
        if name not in ARGUMENTS:
            return failure('UNKNOWN_TOOL', '没有这个业务工具。')
        arguments = ARGUMENTS[name]
        if (not isinstance(args, dict) or set(args) != set(arguments)
                or any(not isinstance(args[argument], str)
                       or not args[argument].strip() or len(args[argument]) > 80
                       for argument in arguments)):
            return failure('INVALID_ARGUMENTS', '参数不符合工具定义；身份不可由工具参数覆盖。')
        # s03 adaptation: enforce identity before executing a private query.
        if name in PRIVATE_TOOLS and self.customer_id is None:
            return authentication_required()
        if name == 'list_my_orders':
            return self.list_my_orders()
        identifier = args[arguments[0]]
        if self.store is not None and name in ('get_product', 'get_order', 'get_tracking'):
            return self._call_database(name, identifier)
        if name in ('get_order', 'get_tracking'):
            order = self.data['orders'].get(identifier)
            if order is None or order['customer_id'] != self.customer_id:
                return failure('ACCESS_DENIED_OR_NOT_FOUND', '订单不存在或不属于当前模拟客户。')
            if name == 'get_order':
                record = {key: value for key, value in order.items()
                          if key not in ('customer_id', 'tracking')}
            else:
                record = {'order_id': identifier, **order['tracking'],
                          'source': order['source'] + '.tracking'}
        else:
            collection = 'products' if name == 'get_product' else 'policies'
            record = self.data[collection].get(identifier)
            if record is None:
                return failure('NOT_FOUND', '模拟资料中没有对应记录，请澄清或转人工。')
        return {'ok': True, 'data': copy.deepcopy(record),
                'simulated': True, 'dataset_version': self.data['dataset_version']}
