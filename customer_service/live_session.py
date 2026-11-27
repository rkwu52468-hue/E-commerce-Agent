"""Local identity and explicit preference commands; credentials never become messages."""
from pathlib import Path
from contextlib import contextmanager
from functools import wraps
from threading import RLock
from .schedule_runtime import ScheduleRuntime
from .auth import CredentialError
from .context import ContextManager
from .harness import Harness
from .memory import MemoryStore, MemoryStoreError, PREFERENCE_VALUES
from .memory_records import is_memory_key
from .shop import Shop, failure
from .task_system import CustomerTaskStore, TaskStoreError


def serialized(method):
    @wraps(method)
    def run(self, *args, **kwargs):
        with self._operation_lock:
            return method(self, *args, **kwargs)
    return run


class LiveSession:
    MAX_LOGIN_FAILURES = 3

    def __init__(self, root, model, store='sqlite', summary_client=None,
                 allow_memory=False, memory_selector=None, memory_extractor=None, allow_tasks=False, allow_background=False, background_delay_seconds=0.0, allow_schedules=False, schedule_clock=None, allow_teams=False, knowledge=None, allow_actions=False, safeguards=None, data_path=None, database_path=None, enable_subagents=True):
        if store not in ('sqlite', 'json'):
            raise ValueError('Unknown shop storage mode')
        self.root = Path(root)
        self._operation_lock = RLock()
        self.allow_schedules = bool(allow_schedules)
        self.allow_teams = bool(allow_teams)
        self.allow_actions = bool(allow_actions)
        self.enable_subagents = bool(enable_subagents)
        self.safeguards = safeguards
        self.data_path = Path(data_path) if data_path is not None else None
        self.database_path = Path(database_path) if database_path is not None else None
        self.scheduler = None
        self.model = model
        self.knowledge = knowledge
        self.store = store
        self.summary_client = summary_client
        self.allow_tasks = bool(allow_tasks)
        self.allow_background = bool(allow_background)
        self.background_delay_seconds = background_delay_seconds
        self._closed = False
        self.allow_memory = bool(allow_memory)
        self.memory_selector = memory_selector if self.allow_memory else None
        self.memory_extractor = memory_extractor if self.allow_memory else None
        self.failed_login_attempts = 0
        self._login_locked = False
        self.harness = self._new_harness(None)
        if self.allow_schedules:
            self.scheduler = ScheduleRuntime(self, clock=schedule_clock)

    def _new_memory_store(self, customer_id):
        try:
            project_root = self.root.resolve()
            memory_root = (self.root / 'data/memory').resolve()
            if not memory_root.is_relative_to(project_root):
                raise MemoryStoreError('记忆目录必须位于本项目内，未读取或写入偏好。')
            return MemoryStore(memory_root, customer_id)
        except (OSError, RuntimeError) as error:
            raise MemoryStoreError('无法安全访问当前账号的偏好存储。') from None

    def _new_task_store(self, customer_id):
        try:
            project_root = self.root.resolve()
            task_root = (self.root / 'data/tasks').resolve()
            if not task_root.is_relative_to(project_root):
                raise TaskStoreError('TASK_STORE_UNAVAILABLE', '任务目录必须位于本项目内，未访问外部记录。')
            return CustomerTaskStore(task_root, customer_id)
        except TaskStoreError:
            raise
        except (OSError, RuntimeError):
            raise TaskStoreError('TASK_STORE_UNAVAILABLE', '无法安全访问当前账号的内部任务存储。') from None

    def _new_harness(self, customer_id):
        default_data = self.root / ('data/shop-action-seed.json' if self.allow_actions else 'data/shop.json')
        default_db = self.root / ('data/shop-actions.db' if self.allow_actions else 'data/shop.db')
        data_file = self.data_path if self.data_path is not None else default_data
        database = (self.database_path if self.database_path is not None else default_db) if self.store == 'sqlite' else None
        shop = Shop(customer_id, data_file, database_path=database, allow_actions=self.allow_actions)
        memory_store = None
        selector = None
        if self.allow_memory and customer_id is not None:
            memory_store = self._new_memory_store(customer_id)
            selector = self.memory_selector
        context = ContextManager(self.root / 'runtime', customer_id or 'guest',
                                 summary_client=self.summary_client)
        memory_args = {'memory_store': memory_store, 'memory_selector': selector}
        if self.allow_memory and customer_id is not None:
            memory_args['memory_extractor'] = self.memory_extractor
        if self.allow_tasks:
            memory_args.update(enable_tasks=True, task_store=self._new_task_store(customer_id)
                               if customer_id is not None else None)
        return Harness(shop, context, self.model, enable_subagents=self.enable_subagents,
                       enable_background=self.allow_background,
                       background_delay_seconds=self.background_delay_seconds,
                       schedule_handler=(lambda name, args: self._schedule_tool(customer_id, name, args))
                       if self.allow_schedules else None, enable_teams=self.allow_teams,
                       knowledge=self.knowledge, safeguards=self.safeguards, **memory_args)

    @property
    def context(self):
        return self.harness.context

    @property
    def shop(self):
        return self.harness.shop

    @property
    def customer_id(self):
        return self.harness.shop.customer_id if self.harness is not None else None

    @property
    def is_authenticated(self):
        return not self._closed and self.customer_id is not None

    @property
    def login_locked(self):
        return self._login_locked

    @serialized
    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.scheduler is not None:
                self.scheduler.stop()
        finally:
            try:
                if self.harness is not None:
                    self.harness.close()
            finally:
                # The shared index survives login/logout context replacement.
                # Mark closed first so even a failing cleanup is never retried.
                if self.knowledge is not None:
                    self.knowledge.close()

    @serialized
    def login(self, customer_id, password, auth_store):
        if self._closed:
            raise ValueError('会话已关闭，请重启后登录。')
        if self.login_locked:
            return False
        try:
            verified = auth_store.verify(customer_id, password)
        except CredentialError:
            verified = None
        if verified is None or verified != customer_id:
            self.failed_login_attempts += 1
            self._login_locked = self.failed_login_attempts >= self.MAX_LOGIN_FAILURES
            return False
        if self.scheduler is not None:
            self.scheduler.poll_only()  # Finish this minute under the old verified identity.
        # Construct the full replacement before changing identity, messages or memory.
        replacement = self._new_harness(verified)
        previous = self.harness
        if previous is not None:
            previous.close()
        self.harness = replacement
        if self.scheduler is not None:
            self.scheduler.identity_changed()
        self.failed_login_attempts = 0
        return True

    @serialized
    def logout(self):
        if self._closed:
            raise ValueError('会话已关闭，请重启后继续。')
        # Revoke first: storage failure must never preserve the previous identity.
        previous = self.harness
        self.harness = None
        if previous is not None:
            previous.close()
        self.harness = self._new_harness(None)
        if self.scheduler is not None:
            self.scheduler.identity_changed()
            self.scheduler.poll_only()

    @serialized
    def memory_command(self, command):
        # The command stays local. Only fixed metadata may enter the audit stream.
        parts = command.split() if isinstance(command, str) else []
        names = {'/memories': 'memories', '/remember': 'remember', '/forget': 'forget'}
        action = names.get(parts[0], 'unknown') if parts else 'unknown'
        key = parts[1] if len(parts) > 1 and is_memory_key(parts[1]) else None
        harness = self.harness

        def audit(status, code):
            if harness is not None:
                harness._write_audit('MemoryCommand', harness._metadata(
                    'MemoryCommand', action=action, key=key, status=status, code=code))

        def finish(result):
            try:
                audit('success' if result['ok'] else 'error',
                      'OK' if result['ok'] else result['error']['code'])
            except (OSError, ValueError):
                if harness is not None:
                    harness.memory_context = ''
                return failure('MEMORY_AUDIT_FAILED',
                               '记忆操作审计未完成；操作可能已完成，请先用/memories核对，不要直接重复提交。')
            return result

        if not self.allow_memory:
            return finish(failure('MEMORY_DISABLED', '长期记忆未开启；请退出并以--live --allow-memory重启。'))
        if harness is None or not self.is_authenticated:
            return finish(failure('AUTHENTICATION_REQUIRED', '请先通过本地/login登录，再管理本人记忆。'))
        valid = (isinstance(command, str)
                 and not any(ord(char) < 32 or ord(char) == 127 for char in command)
                 and ((action == 'memories' and len(parts) == 1)
                      or (action == 'forget' and len(parts) == 2 and key is not None)
                      or (action == 'remember' and len(parts) == 3 and key in PREFERENCE_VALUES
                          and parts[2] in PREFERENCE_VALUES[key])))
        if not valid:
            return finish(failure(
                'INVALID_ARGUMENTS',
                '用法：/memories；/remember reply_language zh-CN|en；'
                '/remember reply_detail concise|detailed；/forget <本人记忆key，见/memories>。'))
        try:
            # Stop before local storage changes if audit is unavailable.
            audit('attempted', 'PENDING')
        except (OSError, ValueError):
            return failure('MEMORY_AUDIT_FAILED', '无法写入记忆操作审计，未执行本次存储操作。')
        try:
            current_store = self._new_memory_store(self.customer_id)
            harness.memory_store = current_store
            if action == 'memories':
                data = {'action': action, 'records': current_store.list_records()}
            elif action == 'remember':
                data = dict(current_store.set_preference(key, parts[2]))
                data['action'] = action
            else:
                removed = current_store.forget(key)
                data = {'action': action, 'key': key,
                        'status': 'forgotten' if removed else 'not_found'}
            harness.memory_context = ''
            return finish({'ok': True, 'data': data})
        except (MemoryStoreError, OSError):
            harness.memory_context = ''
            return finish(failure('MEMORY_STORE_ERROR', '当前账号记忆存储未完成，请检查本地记忆文件；未调用模型。'))

    @serialized
    def turn(self, query):
        if self._closed or self.harness is None:
            raise ValueError('会话已撤销，请退出后重新启动。')
        return self.harness.turn(query)

    @serialized
    def approve_confirmation(self, confirmation_id, execute_now=True):
        if self._closed or self.harness is None:
            raise ValueError('会话已撤销，请退出后重新启动。')
        return self.harness.approve_confirmation(confirmation_id, execute_now=execute_now)

    @serialized
    def reject_confirmation(self, confirmation_id):
        if self._closed or self.harness is None:
            raise ValueError('会话已撤销，请退出后重新启动。')
        return self.harness.reject_confirmation(confirmation_id)

    @serialized
    def pending_confirmations(self):
        if self._closed or self.harness is None:
            return []
        return self.harness.pending_confirmations()

    @serialized
    def list_tickets(self):
        if self._closed or self.harness is None:
            raise ValueError('会话已撤销，请退出后重新启动。')
        return self.harness.shop.list_my_tickets()

    @serialized
    def budget_status(self):
        if self._closed or self.harness is None:
            return None
        return self.harness.budget_status()


    @contextmanager
    def local_credentials(self):
        # Polling may queue, but scheduled model work cannot overlap credential input.
        with self._operation_lock:
            yield

    def start_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.start()

    def scheduled_notices(self):
        with self._operation_lock:
            return self.scheduler.take_notices() if self.scheduler is not None else []

    def poll_schedules(self, now=None):
        return self.scheduler.tick(now) if self.scheduler is not None else None

    def _schedule_tool(self, bound_customer, name, args):
        with self._operation_lock:
            if self._closed or self.scheduler is None:
                return failure('SCHEDULES_DISABLED', '定时安排未开启。')
            if bound_customer is None:
                return failure('AUTHENTICATION_REQUIRED', '请在终端/login登录后管理本人的定时安排。')
            if bound_customer != self.customer_id:
                return failure('ACCESS_DENIED_OR_NOT_FOUND', '当前会话无法访问这份安排。')
            if name == 'schedule_tracking':
                if not isinstance(args, dict) or set(args) != {'order_id', 'cron'}:
                    return failure('INVALID_ARGUMENTS', '只接受order_id和cron。')
                authorized = self.shop.call('get_order', {'order_id': args.get('order_id')})
                if not authorized.get('ok'):
                    return authorized
            result = self.scheduler.manager.call(bound_customer, name, args)
            if name == 'list_schedules' and result.get('ok'):
                result['data']['runtime'] = {
                    'automatic_execution_enabled': not self.scheduler._disabled and not self.scheduler._stop.is_set(),
                    'automatic_attempts': self.scheduler._attempts,
                    'process_attempt_limit': self.scheduler.MAX_ATTEMPTS,
                    'remaining_attempts': max(0, self.scheduler.MAX_ATTEMPTS - self.scheduler._attempts)}
            return result
