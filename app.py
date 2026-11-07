"""Local CLI for the course ecommerce customer-service portfolio."""
import argparse
from contextlib import ExitStack
import json
import math
import os
import sqlite3
from pathlib import Path
from customer_service.demos import run_demo, run_s08
from customer_service.micro_demo import run_micro
from customer_service.fit_demo import run_fit
from customer_service.summary_demo import run_summary
from customer_service.compact_demo import run_compact
from customer_service.reactive_demo import run_reactive
from customer_service.budget_demo import run_budget
from customer_service.hooks_demo import run_hooks
from customer_service.todo_demo import run_todo
from customer_service.subagent_demo import run_subagent_demo
from customer_service.summary import SummaryClient
from customer_service.models import AnthropicModel
from customer_service.live_micro import run_trial
from customer_service.live_context import run_trial as run_context_trial
from customer_service.live_background_plan import run_plan_trial
from customer_service.live_summary import run_summary_trial
from customer_service.shop import Shop
from customer_service.auth import AuthStore, CredentialError, read_secret
from customer_service.live_session import LiveSession
from customer_service.team_trial import run_team_trial
from customer_service.memory import MemoryExtractor, MemorySelector, MemoryStoreError
from customer_service.storage import SQLiteStore, initialize_database
from customer_service.sqlite_demo import run_sqlite_demo
from customer_service.task_system import TaskStoreError

ROOT = Path(__file__).resolve().parent


def main():
    with ExitStack() as resources:
        _main(resources)


def _main(resources):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--demo', action='store_true', help='Offline scripted business demo')
    modes.add_argument('--init-auth', action='store_true', help='Provision a local simulated account password without API calls; requires --customer')
    modes.add_argument('--init-db', action='store_true', help='Initialize the persistent SQLite shop without API calls; never overwrite an existing database')
    modes.add_argument('--demo-sqlite', action='store_true', help='Offline scripted flow using a real isolated SQLite database')
    modes.add_argument('--lesson', choices=['s04-hooks', 's05-todo', 's06-subagent', 's08', 's08-micro', 's08-fit', 's08-summary', 's08-compact', 's08-reactive', 's08-budget'], help='Offline course lessons using synthetic shop data')
    modes.add_argument('--live', action='store_true', help='Explicit interactive model API session')
    modes.add_argument('--live-micro', action='store_true', help='Controlled two-request real-model micro trial')
    modes.add_argument('--live-summary', action='store_true', help='Controlled real-model summary, archive lookup and continuation trial')
    modes.add_argument('--live-context', action='store_true', help='Default-50000 local micro+fit with a maximum of two real retrieval requests; no summary')
    modes.add_argument('--live-background-plan', action='store_true', help='Controlled two-request program-assisted query-plan experiment; synthetic Alice fixture, no login credentials, no auxiliary model calls')
    parser.add_argument('--allow-knowledge', action='store_true', help='With --live, enable local public Chinese vector knowledge search; build the index first using run-rag.cmd build; off by default')
    parser.add_argument('--allow-summary', action='store_true', help='With --live, automatic/explicit/context recovery share one summary attempt (max_tokens=2000, SDK retries=0); context overflow may retry the main request once')
    parser.add_argument('--allow-memory', action='store_true', help='With --live, enable user/feedback/project/reference memories, at most one recall (200 output tokens) and one extraction (600) request per authenticated input; off by default')
    parser.add_argument('--allow-tasks', action='store_true', help='With --live, enable authenticated per-customer persistent investigation tasks; no extra auxiliary API calls; off by default')
    parser.add_argument('--allow-teams', action='store_true', help='With --live, enable bounded reusable readonly teammates and session-only step cards; off by default')
    parser.add_argument('--allow-schedules', action='store_true', help='With --live, enable bounded authenticated daily readonly tracking schedules; off by default')
    parser.add_argument('--allow-background', action='store_true', help='With --live, enable parent-only bounded readonly background tracking queries; off by default')
    parser.add_argument('--background-delay-seconds', type=float, default=None, help='With --allow-background, explicitly simulate 0 to 10 seconds of background latency; default 0')
    parser.add_argument('--dry-run', action='store_true', help='Preview --live-micro, --live-summary, --live-context or --live-background-plan without API calls')
    parser.add_argument('--store', choices=['sqlite', 'json'], help='With --live, select shop storage (default: sqlite)')
    parser.add_argument('--customer', choices=['alice', 'bob'], default=None, help='Fixture identity or explicit --init-auth target; never a --live login shortcut')
    args = parser.parse_args()
    if args.live and args.customer is not None:
        parser.error('--live starts as a guest; use local /login instead of --customer.')
    if args.init_auth and args.customer is None:
        parser.error('--init-auth requires an explicit --customer alice or bob.')
    if args.store is not None and not args.live:
        parser.error('--store only applies to --live.')
    if args.allow_knowledge and not args.live:
        parser.error('--allow-knowledge requires --live.')
    if args.allow_summary and not args.live:
        parser.error('--allow-summary requires --live.')
    if args.allow_memory and not args.live:
        parser.error('--allow-memory requires --live.')
    if args.allow_tasks and not args.live:
        parser.error('--allow-tasks requires --live.')
    if args.allow_teams and not args.live:
        parser.error('--allow-teams requires --live.')
    if args.allow_schedules and not args.live:
        parser.error('--allow-schedules requires --live.')
    if args.allow_background and not args.live:
        parser.error('--allow-background requires --live.')
    if args.background_delay_seconds is not None:
        if (not args.allow_background or not math.isfinite(args.background_delay_seconds)
                or not 0 <= args.background_delay_seconds <= 10):
            parser.error('--background-delay-seconds requires --live --allow-background and a finite value from 0 to 10.')
    if args.live_background_plan:
        if (args.customer or 'alice') != 'alice':
            parser.error('The controlled planning fixture uses synthetic alice, not a login identity.')
        run_plan_trial(ROOT, dry_run=args.dry_run)
        return
    if args.live_context:
        if (args.customer or 'alice') != 'alice':
            parser.error('The controlled fixture uses alice.')
        run_context_trial(ROOT, dry_run=args.dry_run)
        return
    if args.live_summary:
        if (args.customer or 'alice') != 'alice':
            parser.error('The controlled fixture uses alice.')
        run_summary_trial(ROOT, dry_run=args.dry_run)
        return
    if args.live_micro:
        if (args.customer or 'alice') != 'alice':
            parser.error('The controlled fixture uses alice.')
        run_trial(ROOT, dry_run=args.dry_run)
        return
    if args.dry_run:
        parser.error('--dry-run requires --live-micro, --live-summary, --live-context or --live-background-plan.')
    if args.init_auth:
        try:
            initialize_database(ROOT / 'data/shop.db', ROOT / 'data/shop-sqlite-seed.json')
            if not SQLiteStore(ROOT / 'data/shop.db').has_customer(args.customer):
                parser.error('业务资料中没有此模拟账号，未读取密码。')
        except (OSError, ValueError, sqlite3.Error) as error:
            parser.error('模拟账号业务资料检查失败（' + type(error).__name__ + '）；未读取密码或调用API。')
        password = confirmation = None
        try:
            print('仅为本地模拟账号配置密码：' + args.customer)
            password = read_secret('设置密码（15至128字符，隐藏输入）: ')
            confirmation = read_secret('再次输入密码（隐藏输入）: ')
            if password != confirmation:
                parser.error('两次密码不一致，未保存或覆盖凭据。')
            AuthStore(ROOT / 'data/auth').create(args.customer, password)
        except CredentialError as error:
            parser.error(str(error))
        except (EOFError, KeyboardInterrupt):
            print('本地凭据配置已取消。')
            return
        finally:
            password = confirmation = None
        print('本地模拟账号凭据已创建；未读取API key，API调用数为0。')
        return
    if args.init_db:
        try:
            info = initialize_database(ROOT / 'data/shop.db', ROOT / 'data/shop-sqlite-seed.json')
        except (OSError, ValueError, sqlite3.Error) as error:
            parser.error('数据库初始化失败（' + type(error).__name__ + '）；未读取密钥或调用API，也未自动覆盖数据库。')
        print('SQLite商店初始化：' + ('新建' if info['created'] else '已存在，保留原数据'))
        print('数据库：' + str(info['database_path']))
        print('全部业务数据为模拟；API调用数为0。')
        return
    if args.demo_sqlite:
        if (args.customer or 'alice') != 'alice':
            parser.error('固定离线场景仅使用alice；普通--live通过本地/login验证身份。')
        print('全部为模拟数据；真实SQLite引擎与离线脚本演示，API调用数为0。')
        run_sqlite_demo(ROOT)
        return
    if not args.live:
        print('全部为模拟数据；离线脚本演示，API调用数为0。')
        if (args.customer or 'alice') != 'alice':
            parser.error('固定离线场景仅使用alice；普通--live通过本地/login验证身份。')
        {'s04-hooks': run_hooks, 's05-todo': run_todo, 's06-subagent': run_subagent_demo, 's08': run_s08, 's08-micro': run_micro, 's08-fit': run_fit, 's08-summary': run_summary, 's08-compact': run_compact, 's08-reactive': run_reactive, 's08-budget': run_budget, None: run_demo}[args.lesson](ROOT)
        return
    # Validate persistent business storage before asking for credentials or creating an SDK client.
    store = args.store or 'sqlite'
    try:
        if store == 'sqlite':
            initialize_database(ROOT / 'data/shop.db', ROOT / 'data/shop-sqlite-seed.json')
            Shop(None, ROOT / 'data/shop.json', database_path=ROOT / 'data/shop.db')
        else:
            Shop(None, ROOT / 'data/shop.json')
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.error('商店存储检查失败（' + type(error).__name__ + '）；未读取密钥或调用API，请检查数据库，程序不会自动重建覆盖。')
    knowledge = None
    if args.allow_knowledge:
        try:
            from customer_service.knowledge import KnowledgeSearch
            knowledge = KnowledgeSearch(ROOT / 'data/knowledge_demo/documents.json',
                                        ROOT / 'runtime/knowledge_demo',
                                        Path(os.environ.get('FASTEMBED_CACHE_PATH') or (ROOT / '.cache' / 'fastembed')))
            resources.callback(knowledge.close)
        except (ImportError, ValueError, RuntimeError, OSError) as error:
            parser.error('本地知识库未就绪：' + str(error) + '；尚未读取聊天密钥或请求聊天模型。请先运行run-rag.cmd build。')
    config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    try:
        key = os.environ.get('DEEPSEEK_API_KEY') or read_secret('DeepSeek API key（隐藏输入，仅用于当前进程）: ')
    except CredentialError as error:
        parser.error(str(error))
    except (EOFError, KeyboardInterrupt):
        print('会话启动已取消。')
        return
    if not key.strip():
        parser.error('需要API key，不能留空。')
    model = AnthropicModel(key.strip(), config)
    del key
    summary_client = SummaryClient(model.client, model.model, mode=model.mode) if args.allow_summary else None
    memory_selector = MemorySelector(model.client, model.model) if args.allow_memory else None
    memory_extractor = MemoryExtractor(model.client, model.model) if args.allow_memory else None
    session = LiveSession(ROOT, model, store=store, summary_client=summary_client,
                          allow_memory=args.allow_memory, memory_selector=memory_selector,
                          memory_extractor=memory_extractor, allow_tasks=args.allow_tasks,
                          allow_background=args.allow_background,
                          background_delay_seconds=args.background_delay_seconds or 0.0,
                          allow_schedules=args.allow_schedules, allow_teams=args.allow_teams, knowledge=knowledge)
    auth_store = AuthStore(ROOT / 'data/auth')
    print('真实模型交互；全部业务数据为模拟，默认以访客身份开始。')
    print('来源：' + config['base_url'] + '；模型：' + config['model'])
    print('商店存储：' + ('订单与商品使用SQLite；政策与物流使用JSON。' if store == 'sqlite' else '全部使用原JSON模拟数据。'))
    if knowledge is not None:
        print('已启用公开模拟知识库：本地中文Embedding与Qdrant检索；相似片段仍须核实是否能支持结论，订单与物流沿用原业务工具。')
    show_session(session)
    print('本地命令：/login 登录、/logout 退出登录、/help 帮助；输入q退出程序。')
    if args.allow_summary:
        print('已启用自动/主动/超长恢复摘要：三者共用每次客户输入最多1次摘要尝试（max_tokens=2000，失败也计）；SDK自动重试0，仅明确上下文超长可整理后重试1次主请求。摘要与主任务用量分别记录。')
    else:
        print('历史摘要默认关闭，不会自动进行超长恢复；需要启用时以--live --allow-summary启动。')
    if args.allow_memory:
        print('四类长期记忆已开启：登录后自然表达偏好user、长期反馈feedback、本人项目背景project或资料线索reference；程序在成功答复及Stop后核对原话再保存。模型不能提前承诺已保存，请核对[memory] extract日志及确定保存回执。')
        print('每次普通输入最多额外2次请求：选择1次（200输出token）＋提取1次（600）；失败占次数、不自动重试。空目录选择0次但提取仍可1次，访客两者0次。主任务加子任务及记忆最多13次，有摘要恢复最坏15次SDK尝试。')
        print('维护记忆：/memories查看本人记录和key；/forget <key>删除本人记录；手动增改仍可用/remember reply_language zh-CN|en或/remember reply_detail concise|detailed。自动合并整理尚未接入。')
    else:
        print('长期记忆默认关闭，无额外记忆API；需要时用--live --allow-memory重启。')
    if args.allow_tasks:
        print('持久内部调查任务已开启：登录后按本人范围保存，主客服可创建/查看/认领/完成并设置依赖。认领成功才进入进行中，解锁不自动执行；completed不证明业务成功或退款授权。')
        print('新会话可自然要求继续上次调查；先读取已有任务再继续。任务工具不单独调用模型，但可能增加主回合工具轮数，仍遵守8步上限；简单查询无需建任务。')
    else:
        print('持久任务默认关闭；需要时以--live --allow-tasks启动，可同时开启--allow-memory。')
    if args.allow_background:
        print('受控后台物流查询已开启：每输入最多1次启动尝试，最多2项未消费作业；BACKGROUND_STARTED只表示已提交，结果在后续主循环请求前收取。不会自动唤醒模型、外发消息或完成持久任务。')
        print('显式模拟后台延迟：' + str(args.background_delay_seconds or 0.0) + '秒；后台退出不持久化，登录切换会丢弃旧会话迟到结果。')
    if args.allow_teams:
        print('只读团队已开启：登录后由主客服按自然需求安排物流/政策队友；同一验证父会话内复用独立上下文。')
        print('每输入最多两项、同角色一次；每会话最多6次受理含重试，每项最多3次1024输出token模型请求，输入上限24000字符、无SDK重试。')
        print('团队报告在本轮有限等待后交回；成功保存父messages并确认后才完成步骤卡，不自动完成客户整体调查。失败需后续输入显式重试，退出/重登隔离旧队友。')
        print('开发验证：/team-trial 每次程序固定两角色，不测试自主组队；最多两轮，每轮至多7次SDK尝试，仍需/login Alice。')
    if args.allow_schedules:
        print('每日物流安排已开启：Asia/Shanghai固定时分，身份不匹配跳过本次、不补跑；每进程最多3次自动处理，每次最多1次1024输出token，无自动重试。')
        print('程序关闭不执行。定时结果在下一次终端输入后显示给同一已验证会话，不外发；凭据输入期间不执行定时模型请求。')
        session.start_scheduler()
    try:
        while True:
            try:
                query = input('客户 >> ')
                for notice in session.scheduled_notices():
                    print('[schedule] ' + notice)
                if query.strip().lower() in ('q', 'exit'):
                    break
                command = query.strip()
                if command.startswith('/'):
                    if command == '/help':
                        print('本地命令：/login（单独输入后按提示输入账号和隐藏密码）、/logout、/help；q退出。命令不能附带密码。')
                        print('开启--allow-memory且登录后，可自然表达四类长期记忆；当前临时要求不会覆盖持久偏好。每普通输入最多额外选择1次200输出＋提取1次600输出，失败也计且无自动重试。')
                        print('维护命令：/memories、/forget <本人记忆key>；手动偏好增改：/remember reply_language zh-CN|en、/remember reply_detail concise|detailed。命令本身不调用模型。')
                        print('团队需要--allow-teams：可自然要求核对订单物流与规则、继续某项调查；步骤与持久case分开，仍按身份和预算执行。')
                        print('开发验证/team-trial：仅--live --allow-teams、登录Alice，程序固定分工；最多两轮，每轮至多7次SDK，失败停止，不代表模型自主组队。')
                        print('每日物流安排需要--allow-schedules且登录：例如每天9点查询本人订单物流；自然语言可查看或取消本人安排。')
                        print('持久调查需要--allow-tasks：用自然语言要求建立或继续本人调查，程序不会把任务记录当成退款/售后单。')
                    elif command == '/team-trial':
                        result = run_team_trial(session)
                        print('[team trial] ' + result['status'] + ': ' + result['message'])
                        if result.get('answer'):
                            print('客服：' + result['answer'])
                        if result.get('metrics'):
                            print('[team trial usage] ' + json.dumps(result['metrics'], ensure_ascii=False))
                        if result.get('report_path'):
                            print('验证报告：' + result['report_path'])
                    elif command == '/login':
                        if session.login_locked:
                            print('本进程登录尝试已锁定。未登录时可继续公开查询，或退出程序后重启。')
                            continue
                        with session.local_credentials():
                            customer_id = input('本地模拟账号: ').strip()
                            password = None
                            try:
                                password = read_secret('账号密码（隐藏输入，不发送给模型）: ')
                                accepted = session.login(customer_id, password, auth_store)
                            finally:
                                password = None
                        if accepted:
                            print('登录验证成功，已切换到新的会话上下文。')
                        else:
                            remaining = max(0, session.MAX_LOGIN_FAILURES - session.failed_login_attempts)
                            print('登录未成功，当前身份和会话保持不变；本进程剩余登录尝试：' + str(remaining))
                        show_session(session)
                    elif command == '/logout':
                        try:
                            session.logout()
                        except (OSError, ValueError, sqlite3.Error, TaskStoreError):
                            print('已撤销当前登录，但无法创建访客会话；程序结束，请检查数据存储。')
                            break
                        print('已退出登录，已创建新的访客会话。')
                        show_session(session)
                    elif command.split()[0] in ('/remember', '/memories', '/forget'):
                        show_memory_result(session.memory_command(command))
                    else:
                        print('未知本地命令或命令包含参数，未发送给模型。请输入/help查看用法。')
                    continue
                print('客服：' + session.turn(query))
            except (EOFError, KeyboardInterrupt):
                print('\n会话结束。')
                break
            except TaskStoreError as error:
                print('内部任务存储初始化未完成（' + error.code + '）；本次登录未切换身份，请检查后重试。')
            except MemoryStoreError:
                print('当前账号偏好存储不可用；当前身份与会话未切换，请检查本地记忆文件。')
            except (RuntimeError, ValueError) as error:
                print('本轮未完成：' + str(error))
    finally:
        session.close()




def show_memory_result(result):
    if not result['ok']:
        print('记忆操作未完成：' + result['error']['message'])
        return
    data = result['data']
    labels = {'reply_language': '回复语言', 'reply_detail': '回复详细程度'}
    values = {'zh-CN': '中文', 'en': '英语', 'concise': '简洁', 'detailed': '详细'}
    if data['action'] == 'memories':
        records = data['records']
        if not records:
            print('当前已登录账号尚无保存的记忆。')
        else:
            print('当前已登录账号的记忆：')
            for record in records:
                if record['key'] in labels:
                    print('  ' + labels[record['key']] + '：' + values.get(record['value'], '未知'))
                else:
                    print('  [' + record['type'] + '] ' + record['key'] + '：' + record['body'])
    elif data['action'] == 'remember':
        status = '偏好未变化' if data['status'] == 'unchanged' else '偏好已保存'
        print(status + '：' + labels[data['key']] + '＝' + values[data['value']])
    else:
        print('记忆已删除。' if data['status'] == 'forgotten' else '该项记忆原本未保存。')


def show_session(session):
    if session.harness is None:
        print('会话已撤销，请退出后重新启动。')
        return
    identity = ('已验证的本地模拟账号：' + session.customer_id) if session.is_authenticated else '访客（未登录）'
    print('当前身份：' + identity)
    print('记录：' + str(session.context.session_dir))


if __name__ == '__main__':
    main()
