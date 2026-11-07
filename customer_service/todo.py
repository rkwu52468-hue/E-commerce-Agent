"""s05 business adaptation: model-reported plans, scoped to one Harness.

Reference: fixed upstream s05 TodoManager.update/render and batch reminder loop.
Strict JSON tool inputs replace upstream coercion/string parsing. Plan statuses
are not automatically verified business facts or authorization. No global TODO
and no cross-process task restore are introduced in this lesson.
"""
import copy
import json
from .shop import failure

STATUSES = ('pending', 'in_progress', 'completed')
TODO_TOOL = {
    'name': 'todo_write',
    'description': ('替换当前会话多步骤任务的整份清单；简单查询无需使用。最多20项，'
                    '最多一项in_progress。状态是模型提交的计划进度，不是业务事实或授权；'
                    '查询失败的项目应保留in_progress并在content说明资料缺口。'),
    'input_schema': {'type': 'object', 'properties': {
        'todos': {'type': 'array', 'maxItems': 20, 'items': {
            'type': 'object', 'properties': {
                'content': {'type': 'string', 'minLength': 1, 'maxLength': 500},
                'status': {'type': 'string', 'enum': list(STATUSES)}},
            'required': ['content', 'status'], 'additionalProperties': False}}},
        'required': ['todos'], 'additionalProperties': False},
}


MAX_PROJECTION_CHARS = 20000


def plan_projection(items):
    state = json.dumps({'todos': items, 'reference_only': True,
                        'status_source': 'model_reported'}, ensure_ascii=False)
    return state.replace('<', r'\u003c').replace('>', r'\u003e').replace('&', r'\u0026')


class TodoManager:
    def __init__(self):
        self._items = []

    @property
    def items(self):
        return copy.deepcopy(self._items)

    def update(self, todos):
        if not isinstance(todos, list) or len(todos) > 20:
            raise ValueError('todos must be an array with at most 20 items')
        validated = []
        for item in todos:
            if not isinstance(item, dict) or set(item) != {'content', 'status'}:
                raise ValueError('Each task requires only content and status')
            content, status = item['content'], item['status']
            if (not isinstance(content, str) or not content.strip()
                    or len(content) > 500
                    or any(ord(char) < 32 or ord(char) == 127 for char in content)):
                raise ValueError('Task content must be a nonempty single-line string up to 500 characters')
            if not isinstance(status, str) or status not in STATUSES:
                raise ValueError('Unknown task status')
            validated.append({'content': content.strip(), 'status': status})
        if sum(item['status'] == 'in_progress' for item in validated) > 1:
            raise ValueError('Only one task can be in_progress')
        if len(plan_projection(validated)) > MAX_PROJECTION_CHARS:
            raise ValueError('Escaped task projection exceeds character budget')
        # Validate the full replacement before committing any state.
        self._items = validated
        return self.render()

    def render(self):
        if not self.items:
            return 'No todos.'
        marks = {'pending': '[ ]', 'in_progress': '[>]', 'completed': '[x]'}
        lines = [marks[item['status']] + ' ' + item['content'] for item in self.items]
        done = sum(item['status'] == 'completed' for item in self.items)
        return '\n'.join(lines) + f'\n\n({done}/{len(self.items)} completed)'

    def call(self, args):
        if not isinstance(args, dict) or set(args) != {'todos'}:
            return failure('INVALID_ARGUMENTS', '任务清单只接受todos数组；不能覆盖客户身份。')
        try:
            rendered = self.update(args['todos'])
        except ValueError:
            return failure('INVALID_ARGUMENTS', '清单需为最多20项的数组，每项仅含1至500字符单行content及合法status，最多一项进行中，转义后计划投影最多20000字符；原清单未改。')
        return {'ok': True, 'data': {'todos': copy.deepcopy(self.items),
                'rendered': rendered, 'completed': sum(x['status'] == 'completed' for x in self.items),
                'total': len(self.items), 'reference_only': True,
                'status_source': 'model_reported'}}
