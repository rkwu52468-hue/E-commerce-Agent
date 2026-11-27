"""Business tools over explicitly synthetic shop data (read-only by default, opt-in actions)."""
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import secrets
from .storage import CANCELLABLE_STATUSES, SQLiteStore, StoreError

SHANGHAI_TZ = timezone(timedelta(hours=8), name='Asia/Shanghai')

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

TICKET_CATEGORIES = (
    'refund_dispute',
    'logistics_issue',
    'order_change',
    'policy_gap',
    'human_escalation',
    'other',
)
TICKET_PRIORITIES = ('low', 'normal', 'high', 'urgent')
MAX_TICKETS_PER_SESSION = 5

ACTION_TOOL_DEFINITIONS = [
    {
        'name': 'check_refund_status',
        'description': (
            '查询已登录模拟客户自己某笔订单的退款进度、退款金额与审核说明；未登录需先/login。'
            '身份由程序绑定，不能查他人订单。amount_cents单位是分，显示CNY元须除以100。'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'order_id': {'type': 'string', 'minLength': 1, 'maxLength': 80},
            },
            'required': ['order_id'],
            'additionalProperties': False,
        },
    },
    {
        'name': 'cancel_order',
        'description': (
            '取消已登录模拟客户本人的未发货订单（仅支持pending_payment与paid_unshipped状态）。'
            '高危写操作必须两步确认：第一步不传confirmation_token调用以校验状态并获取confirmation_id，'
            '向客户说明订单金额及是否会自动生成退款单并请求明确确认；'
            '仅在客户后续明确同意（或使用/confirm）后，第二步再传入该confirmation_token执行取消。'
            '严禁在同一回合内自行伪造或提前确认。已发货/已签收订单不可直接取消。'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'order_id': {'type': 'string', 'minLength': 1, 'maxLength': 80},
                'reason': {'type': 'string', 'minLength': 1, 'maxLength': 200},
                'confirmation_token': {'type': 'string', 'minLength': 1, 'maxLength': 80},
            },
            'required': ['order_id', 'reason'],
            'additionalProperties': False,
        },
    },
    {
        'name': 'create_support_ticket',
        'description': (
            '为已登录模拟客户创建售后工单或触发人工升级（Human Escalation）。'
            '适用于物流延误赔付协商、退款被拒复核、超期售后、政策缺口、工具持续异常或客户要求转人工。'
            '若传入order_id，程序会校验归属本人；相同未关闭工单会自动去重。'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'category': {'type': 'string', 'enum': list(TICKET_CATEGORIES)},
                'summary': {'type': 'string', 'minLength': 5, 'maxLength': 300},
                'priority': {'type': 'string', 'enum': list(TICKET_PRIORITIES)},
                'order_id': {'type': 'string', 'minLength': 1, 'maxLength': 80},
                'escalate_to_human': {'type': 'boolean'},
            },
            'required': ['category', 'summary', 'priority'],
            'additionalProperties': False,
        },
    },
    {
        'name': 'list_my_tickets',
        'description': (
            '无需参数，列出已登录模拟客户本人的售后工单记录；未登录需先/login。'
            '身份由程序绑定，不接受客户身份参数，不返回他人工单。'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {},
            'required': [],
            'additionalProperties': False,
        },
    },
]
ACTION_TOOL_NAMES = frozenset(tool['name'] for tool in ACTION_TOOL_DEFINITIONS)

ACTION_SYSTEM = (
    '当前会话已开启受控售后行动能力，新增工具：check_refund_status（查退款进度）、'
    'cancel_order（取消未发货订单）、create_support_ticket（创建售后工单/转人工）、list_my_tickets（查本人工单）。'
    '涉及取消订单时必须严格遵守两步二次确认：先不传confirmation_token调用cancel_order校验状态机并取得confirmation_id，'
    '向客户写明订单号、金额、当前状态及是否自动生成退款单，等待客户下一轮明确确认或执行/confirm后再传入confirmation_token提交。'
    '严禁在同一回合自批确认码；收到ORDER_NOT_CANCELLABLE时不得宣称已取消，应解释状态并引导退货政策或创建工单；'
    '收到ORDER_ALREADY_CANCELLED时按返回记录如实告知，不重复操作。'
    '遇到退款被拒申诉、物流严重延误赔付协商、政策缺口、工具持续超时/熔断或客户要求转人工时，'
    '应调用create_support_ticket（并按需设置escalate_to_human=true）并向客户汇报真实返回的ticket_id与分配队列。'
    '没有工具返回的ok=true回执前，严禁编造取消成功、退款单号或工单号。'
)


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


PRIVATE_TOOLS = frozenset({
    'get_order', 'get_tracking', 'list_my_orders',
    'check_refund_status', 'cancel_order', 'create_support_ticket', 'list_my_tickets',
})


def authentication_required():
    return failure('AUTHENTICATION_REQUIRED', '请先在终端输入/login完成本地身份验证；不要在聊天中提供密码或验证码。')


def failure(code, message, data=None):
    payload = {'ok': False, 'error': {'code': code, 'message': message}}
    if data is not None:
        payload['data'] = data
    return payload


def _valid_text(value, min_len=1, max_len=200):
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not min_len <= len(stripped) <= max_len or len(value) > max_len:
        return False
    return not any((ord(ch) < 32 and ch not in '\n\t') or ord(ch) == 127 for ch in value)


class Shop:
    def __init__(self, customer_id: str | None, data_path: Path, database_path: Path | None = None,
                 allow_actions: bool = False):
        self.data = json.loads(Path(data_path).read_text(encoding='utf-8'))
        self.allow_actions = bool(allow_actions)
        self.store = SQLiteStore(database_path, allow_writes=self.allow_actions) if database_path is not None else None
        # None is the anonymous principal. A customer ID must be bound by the
        # trusted caller (LiveSession after verification; old lessons simulate it).
        if customer_id is not None:
            if not isinstance(customer_id, str):
                raise ValueError('Unknown simulated customer')
            known = self.store.has_customer(customer_id) if self.store is not None else customer_id in self.data['customers']
            if not known:
                raise ValueError('Unknown simulated customer')
        self.customer_id = customer_id
        self._tickets_created_this_session = 0
        # Ensure optional collections exist in memory for JSON fixture mode.
        self.data.setdefault('order_cancellations', {})
        self.data.setdefault('refunds', {})
        self.data.setdefault('tickets', {})

    def tool_definitions(self, include_actions: bool | None = None):
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
        expose_actions = self.allow_actions if include_actions is None else bool(include_actions)
        if expose_actions:
            tools.extend(copy.deepcopy(ACTION_TOOL_DEFINITIONS))
        return tools

    def _database_success(self, record):
        return {'ok': True, 'data': record, 'simulated': True,
                'dataset_version': self.store.dataset_version, 'storage': 'sqlite'}

    def _json_success(self, record):
        return {'ok': True, 'data': copy.deepcopy(record), 'simulated': True,
                'dataset_version': self.data['dataset_version']}

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

    def _owned_order_and_extras(self, order_id):
        """Return (order, cancellation, refund) for current customer, or (None, None, None)."""
        if self.store is not None:
            order = self.store.get_order(order_id, self.customer_id)
            if order is None:
                return None, None, None
            cancellation = self.store.get_cancellation(order_id, self.customer_id)
            refund = self.store.get_refund_by_order(order_id, self.customer_id)
            return order, cancellation, refund
        raw = self.data['orders'].get(order_id)
        if raw is None or raw['customer_id'] != self.customer_id:
            return None, None, None
        order = {k: copy.deepcopy(v) for k, v in raw.items() if k not in ('customer_id', 'tracking')}
        cancel_raw = self.data['order_cancellations'].get(order_id)
        cancellation = ({k: copy.deepcopy(v) for k, v in cancel_raw.items() if k != 'customer_id'}
                        if cancel_raw and cancel_raw.get('customer_id') == self.customer_id else None)
        refund_raw = next(
            (item for item in self.data['refunds'].values()
             if item['order_id'] == order_id and item['customer_id'] == self.customer_id),
            None,
        )
        refund = ({k: copy.deepcopy(v) for k, v in refund_raw.items() if k != 'customer_id'}
                  if refund_raw is not None else None)
        if refund is not None and 'source' not in refund:
            refund['source'] = 'shop.json#refunds.' + refund['refund_id']
        return order, cancellation, refund

    def check_refund_status(self, order_id):
        if not self.allow_actions:
            return failure('UNKNOWN_TOOL', '没有这个业务工具。')
        if self.customer_id is None:
            return authentication_required()
        if not _valid_text(order_id, 1, 80):
            return failure('INVALID_ARGUMENTS', 'order_id必须为1至80字符字符串。')
        try:
            order, _, refund = self._owned_order_and_extras(order_id)
        except StoreError:
            return failure('DATA_STORE_UNAVAILABLE', '模拟数据库查询退款进度失败，请稍后重试。')
        if order is None:
            return failure('ACCESS_DENIED_OR_NOT_FOUND', '订单不存在或不属于当前模拟客户。')
        payload = {
            'order_id': order_id,
            'order_status': order['status'],
            'has_refund': refund is not None,
            'refund': refund,
            'source': refund['source'] if refund is not None else order['source'],
        }
        return self._database_success(payload) if self.store is not None else self._json_success(payload)

    def preview_cancel_order(self, order_id, reason):
        """Validate ownership and business state machine before creating a confirmation request."""
        if not self.allow_actions:
            return failure('UNKNOWN_TOOL', '没有这个业务工具。')
        if self.customer_id is None:
            return authentication_required()
        if not _valid_text(order_id, 1, 80) or not _valid_text(reason, 1, 200):
            return failure('INVALID_ARGUMENTS', 'cancel_order需要合法的order_id（1-80字符）和reason（1-200字符）。')
        try:
            order, cancellation, refund = self._owned_order_and_extras(order_id)
        except StoreError:
            return failure('DATA_STORE_UNAVAILABLE', '模拟数据库校验订单状态失败，未执行取消。')
        if order is None:
            return failure('ACCESS_DENIED_OR_NOT_FOUND', '订单不存在或不属于当前模拟客户。')
        if order['status'] == 'cancelled':
            return failure(
                'ORDER_ALREADY_CANCELLED',
                '该订单此前已取消，无需重复取消（已做幂等保护）。',
                data={'order': order, 'cancellation': cancellation, 'refund': refund},
            )
        if order['status'] not in CANCELLABLE_STATUSES:
            return failure(
                'ORDER_NOT_CANCELLABLE',
                '订单当前状态为 ' + order['status'] + '，不支持直接取消；请参考退换货政策或创建售后工单转人工处理。',
                data={'order': order, 'cancellable_statuses': sorted(CANCELLABLE_STATUSES)},
            )
        will_refund = order['status'] == 'paid_unshipped'
        preview = {
            'order': order,
            'reason': reason.strip(),
            'will_create_refund': will_refund,
            'refund_amount_cents': order['total_cents'] if will_refund else 0,
            'currency': order['currency'],
        }
        return self._database_success(preview) if self.store is not None else self._json_success(preview)

    def execute_cancel_order(self, order_id, reason, confirmation_id):
        """Execute a verified, user-confirmed order cancellation atomically."""
        if not self.allow_actions:
            return failure('UNKNOWN_TOOL', '没有这个业务工具。')
        if self.customer_id is None:
            return authentication_required()
        if (not _valid_text(order_id, 1, 80) or not _valid_text(reason, 1, 200)
                or not _valid_text(confirmation_id, 1, 80)):
            return failure('INVALID_ARGUMENTS', '取消执行参数不合法。')
        clean_reason = reason.strip()
        if self.store is not None:
            try:
                outcome = self.store.cancel_order(order_id, self.customer_id, clean_reason, confirmation_id)
            except StoreError:
                return failure('DATA_STORE_UNAVAILABLE', '模拟数据库执行订单取消失败，请先核对订单状态再重试。')
            kind = outcome['outcome']
            if kind == 'not_found':
                return failure('ACCESS_DENIED_OR_NOT_FOUND', '订单不存在或不属于当前模拟客户。')
            if kind == 'already_cancelled':
                return failure(
                    'ORDER_ALREADY_CANCELLED',
                    '该订单此前已取消，无需重复取消（已做幂等保护）。',
                    data={
                        'order': outcome['order'],
                        'cancellation': outcome.get('cancellation'),
                        'refund': outcome.get('refund'),
                    },
                )
            if kind == 'not_cancellable':
                return failure(
                    'ORDER_NOT_CANCELLABLE',
                    '订单当前状态为 ' + outcome['order']['status'] + '，不支持直接取消；请走退货退款或创建工单。',
                    data={'order': outcome['order'], 'cancellable_statuses': sorted(CANCELLABLE_STATUSES)},
                )
            return self._database_success({
                'order': outcome['order'],
                'previous_status': outcome['previous_status'],
                'status': 'cancelled',
                'reason': clean_reason,
                'cancelled_at': outcome['cancelled_at'],
                'confirmation_id': confirmation_id,
                'refund_created': outcome['refund'],
                'source': outcome['order']['source'],
            })
        # JSON in-memory fallback (never mutates fixture file on disk).
        raw = self.data['orders'].get(order_id)
        if raw is None or raw['customer_id'] != self.customer_id:
            return failure('ACCESS_DENIED_OR_NOT_FOUND', '订单不存在或不属于当前模拟客户。')
        if raw['status'] == 'cancelled':
            order, cancellation, refund = self._owned_order_and_extras(order_id)
            return failure('ORDER_ALREADY_CANCELLED', '该订单此前已取消，无需重复取消（已做幂等保护）。',
                           data={'order': order, 'cancellation': cancellation, 'refund': refund})
        if raw['status'] not in CANCELLABLE_STATUSES:
            order, _, _ = self._owned_order_and_extras(order_id)
            return failure('ORDER_NOT_CANCELLABLE',
                           '订单当前状态为 ' + raw['status'] + '，不支持直接取消；请走退货退款或创建工单。',
                           data={'order': order, 'cancellable_statuses': sorted(CANCELLABLE_STATUSES)})
        previous = raw['status']
        timestamp = datetime.now(SHANGHAI_TZ).isoformat(timespec='seconds')
        raw['status'] = 'cancelled'
        self.data['order_cancellations'][order_id] = {
            'order_id': order_id, 'customer_id': self.customer_id, 'previous_status': previous,
            'reason': clean_reason, 'cancelled_at': timestamp, 'confirmation_id': confirmation_id,
        }
        refund = None
        if previous == 'paid_unshipped':
            refund_id = 'RF-' + order_id
            refund_record = {
                'refund_id': refund_id, 'order_id': order_id, 'customer_id': self.customer_id,
                'amount_cents': raw['total_cents'], 'currency': raw['currency'],
                'status': 'approved_pending_payout', 'reason': '未发货订单取消自动退款：' + clean_reason,
                'updated_at': timestamp, 'estimated_arrival': '预计1-3个工作日原路退回',
                'reviewer_note': '未发货取消自动审核通过，已进入打款队列',
                'source': 'shop.json#refunds.' + refund_id,
            }
            self.data['refunds'][refund_id] = refund_record
            refund = {k: v for k, v in refund_record.items() if k != 'customer_id'}
        order_view = {k: copy.deepcopy(v) for k, v in raw.items() if k not in ('customer_id', 'tracking')}
        return self._json_success({
            'order': order_view,
            'previous_status': previous,
            'status': 'cancelled',
            'reason': clean_reason,
            'cancelled_at': timestamp,
            'confirmation_id': confirmation_id,
            'refund_created': refund,
            'source': order_view['source'],
        })

    def create_support_ticket(self, category, summary, priority, order_id=None, escalate_to_human=None):
        if not self.allow_actions:
            return failure('UNKNOWN_TOOL', '没有这个业务工具。')
        if self.customer_id is None:
            return authentication_required()
        if (category not in TICKET_CATEGORIES or priority not in TICKET_PRIORITIES
                or not _valid_text(summary, 5, 300)):
            return failure('INVALID_ARGUMENTS', '工单参数不符合要求（需有效category、priority及5至300字符summary）。')
        if order_id is not None and not _valid_text(order_id, 1, 80):
            return failure('INVALID_ARGUMENTS', 'order_id必须为1至80字符字符串。')
        if escalate_to_human is not None and type(escalate_to_human) is not bool:
            return failure('INVALID_ARGUMENTS', 'escalate_to_human必须为布尔值。')
        escalate = (category == 'human_escalation') if escalate_to_human is None else escalate_to_human
        clean_summary = summary.strip()
        clean_order_id = order_id.strip() if isinstance(order_id, str) else None
        if self._tickets_created_this_session >= MAX_TICKETS_PER_SESSION:
            return failure('TICKET_SESSION_LIMIT_REACHED', '本会话创建工单已达上限，请直接通过已有工单跟进或联系人工客服。')
        if self.store is not None:
            try:
                outcome = self.store.create_support_ticket(
                    self.customer_id, category, clean_summary, priority,
                    order_id=clean_order_id, escalate_to_human=escalate,
                )
            except StoreError:
                return failure('DATA_STORE_UNAVAILABLE', '模拟数据库写入工单失败，请先查看本人工单列表再重试。')
            if outcome['outcome'] == 'order_not_owned':
                return failure('ACCESS_DENIED_OR_NOT_FOUND', '关联订单不存在或不属于当前模拟客户，未创建工单。')
            deduplicated = outcome['outcome'] == 'deduplicated'
            if not deduplicated:
                self._tickets_created_this_session += 1
            return self._database_success({
                'ticket': outcome['ticket'],
                'deduplicated': deduplicated,
                'source': outcome['ticket']['source'],
            })
        # JSON fallback mode
        if clean_order_id is not None:
            raw = self.data['orders'].get(clean_order_id)
            if raw is None or raw['customer_id'] != self.customer_id:
                return failure('ACCESS_DENIED_OR_NOT_FOUND', '关联订单不存在或不属于当前模拟客户，未创建工单。')
        for existing in self.data['tickets'].values():
            if (existing['customer_id'] == self.customer_id and existing['status'] == 'open'
                    and existing['category'] == category and existing.get('order_id') == clean_order_id
                    and existing['summary'] == clean_summary):
                view = {k: copy.deepcopy(v) for k, v in existing.items() if k != 'customer_id'}
                view.setdefault('source', 'shop.json#tickets.' + view['ticket_id'])
                return self._json_success({'ticket': view, 'deduplicated': True, 'source': view['source']})
        now = datetime.now(SHANGHAI_TZ)
        ticket_id = 'TK-' + now.strftime('%Y%m%d') + '-' + secrets.token_hex(3).upper()
        record = {
            'ticket_id': ticket_id, 'customer_id': self.customer_id, 'order_id': clean_order_id,
            'category': category, 'priority': priority, 'status': 'open',
            'escalate_to_human': escalate,
            'assigned_queue': 'human_specialist' if escalate else 'after_sales_support',
            'summary': clean_summary, 'created_at': now.isoformat(timespec='seconds'),
            'source': 'shop.json#tickets.' + ticket_id,
        }
        self.data['tickets'][ticket_id] = record
        self._tickets_created_this_session += 1
        view = {k: copy.deepcopy(v) for k, v in record.items() if k != 'customer_id'}
        return self._json_success({'ticket': view, 'deduplicated': False, 'source': view['source']})

    def list_my_tickets(self):
        if not self.allow_actions:
            return failure('UNKNOWN_TOOL', '没有这个业务工具。')
        if self.customer_id is None:
            return authentication_required()
        if self.store is not None:
            try:
                return self._database_success({'tickets': self.store.list_tickets(self.customer_id)})
            except StoreError:
                return failure('DATA_STORE_UNAVAILABLE', '模拟数据库查询工单列表失败，请稍后重试。')
        tickets = []
        for item in sorted(self.data['tickets'].values(),
                           key=lambda row: (row['created_at'], row['ticket_id']), reverse=True):
            if item['customer_id'] == self.customer_id:
                view = {k: copy.deepcopy(v) for k, v in item.items() if k != 'customer_id'}
                view.setdefault('source', 'shop.json#tickets.' + view['ticket_id'])
                tickets.append(view)
        return self._json_success({'tickets': tickets})

    def _call_action(self, name, args):
        if not isinstance(args, dict):
            return failure('INVALID_ARGUMENTS', '参数不符合工具定义；身份不可由工具参数覆盖。')
        if name == 'list_my_tickets':
            if args:
                return failure('INVALID_ARGUMENTS', 'list_my_tickets不接受任何参数。')
            return self.list_my_tickets()
        if name == 'check_refund_status':
            if set(args) != {'order_id'}:
                return failure('INVALID_ARGUMENTS', 'check_refund_status仅接受order_id参数。')
            return self.check_refund_status(args['order_id'])
        if name == 'create_support_ticket':
            allowed = {'category', 'summary', 'priority', 'order_id', 'escalate_to_human'}
            required = {'category', 'summary', 'priority'}
            if not required <= set(args) or not set(args) <= allowed:
                return failure('INVALID_ARGUMENTS', 'create_support_ticket参数字段不符合定义。')
            return self.create_support_ticket(
                args['category'], args['summary'], args['priority'],
                order_id=args.get('order_id'), escalate_to_human=args.get('escalate_to_human'),
            )
        if name == 'cancel_order':
            allowed = {'order_id', 'reason', 'confirmation_token'}
            required = {'order_id', 'reason'}
            if not required <= set(args) or not set(args) <= allowed:
                return failure('INVALID_ARGUMENTS', 'cancel_order参数字段不符合定义。')
            # Direct shop.call('cancel_order', ...) never bypasses the Harness confirmation gate.
            preview = self.preview_cancel_order(args['order_id'], args['reason'])
            if not preview['ok']:
                return preview
            return failure(
                'CONFIRMATION_REQUIRED',
                '取消订单属于高危写操作，必须通过会话二次确认门禁获取并验证确认令牌后方可执行。',
                data=preview['data'],
            )
        return failure('UNKNOWN_TOOL', '没有这个业务工具。')

    def call(self, name, args):
        if self.allow_actions and name in ACTION_TOOL_NAMES:
            return self._call_action(name, args)
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
