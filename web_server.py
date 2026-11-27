"""Launch the Stage 3/4 E-Commerce AI Support Web Workbench (FastAPI + SSE).

Supports both local CLI flags and cloud/container environment variables
(HOST, PORT, APP_MODE, DEEPSEEK_API_KEY, RETRIEVAL_MODE, RATE_LIMIT_RPM,
DAILY_TOKEN_BUDGET, STRICT_AUTH, AUTO_BUILD_INDEX, FASTEMBED_CACHE_PATH).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from customer_service.auth import CredentialError, read_secret
from customer_service.models import AnthropicModel
from customer_service.web_app import create_app
from customer_service.web_sessions import DemoInteractiveModel


def _env_bool(environ: dict, name: str, default: bool = False) -> bool:
    raw = str(environ.get(name, '')).strip().lower()
    if not raw:
        return default
    return raw in ('1', 'true', 'yes', 'on')


def _env_int(environ: dict, name: str, default: int) -> int:
    raw = str(environ.get(name, '')).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def resolve_runtime_settings(argv: list[str] | None = None,
                             environ: dict | None = None) -> argparse.Namespace:
    env = os.environ if environ is None else environ
    default_mode = env.get('APP_MODE', 'auto').strip().lower() or 'auto'
    if default_mode not in ('auto', 'live', 'demo'):
        default_mode = 'auto'
    default_retrieval = env.get('RETRIEVAL_MODE', 'hybrid').strip().lower() or 'hybrid'
    if default_retrieval not in ('hybrid', 'dense'):
        default_retrieval = 'hybrid'

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--mode',
        choices=['auto', 'live', 'demo'],
        default=default_mode,
        help='auto = live when DEEPSEEK_API_KEY is set else demo; live = real DeepSeek API; demo = zero-API preview',
    )
    parser.add_argument('--host', default=env.get('HOST', '127.0.0.1'))
    parser.add_argument('--port', type=int, default=_env_int(env, 'PORT', 8000))
    parser.add_argument('--retrieval-mode', choices=['hybrid', 'dense'], default=default_retrieval)
    parser.add_argument('--no-knowledge', action='store_true', default=_env_bool(env, 'NO_KNOWLEDGE', False),
                        help='Start without loading local Qdrant knowledge index')
    parser.add_argument('--auto-build-index', action='store_true', default=_env_bool(env, 'AUTO_BUILD_INDEX', False),
                        help='Automatically build runtime/knowledge_demo if missing on startup')
    parser.add_argument('--rate-limit', type=int, default=_env_int(env, 'RATE_LIMIT_RPM', 30),
                        help='Max chat requests per minute per client+session')
    parser.add_argument('--daily-token-budget', type=int, default=_env_int(env, 'DAILY_TOKEN_BUDGET', 500000))
    parser.add_argument('--strict-auth', action='store_true', default=_env_bool(env, 'STRICT_AUTH', False),
                        help='Disable 1-click demo account switching and require --init-auth passwords')
    args = parser.parse_args(argv)

    has_key = bool(str(env.get('DEEPSEEK_API_KEY', '')).strip())
    if args.mode == 'auto':
        args.effective_mode = 'live' if has_key else 'demo'
    else:
        args.effective_mode = args.mode
    return args


def main(argv: list[str] | None = None):
    args = resolve_runtime_settings(argv)

    knowledge = None
    if not args.no_knowledge:
        index_root = ROOT / 'runtime/knowledge_demo'
        cache_dir = Path(os.environ.get('FASTEMBED_CACHE_PATH') or (ROOT / '.cache' / 'fastembed'))
        docs_path = ROOT / 'data/knowledge_demo/documents.json'
        if not (index_root / 'CURRENT').exists() and args.auto_build_index:
            try:
                from customer_service.knowledge import build_index
                print(f'[web] Building local Qdrant knowledge index into {index_root}...')
                build_index(docs_path, index_root, cache_dir)
            except Exception as exc:
                print(f'[web] Warning: auto-build knowledge index failed ({exc}); continuing without RAG.')
        if (index_root / 'CURRENT').exists():
            try:
                from customer_service.knowledge import KnowledgeSearch
                knowledge = KnowledgeSearch(
                    docs_path,
                    index_root,
                    cache_dir,
                    retrieval_mode=args.retrieval_mode,
                )
                print(f'[web] Local Hybrid RAG loaded ({args.retrieval_mode}) from {index_root}')
            except Exception as exc:
                print(f'[web] Warning: local knowledge index not loaded ({exc}); continuing without RAG.')
        else:
            print('[web] Note: runtime/knowledge_demo not built; continuing without RAG.')

    if args.effective_mode == 'live':
        config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
        key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
        if not key:
            try:
                key = read_secret('DeepSeek API key（隐藏输入，仅用于当前 Web 进程）: ').strip()
            except (CredentialError, EOFError, KeyboardInterrupt):
                print('已取消启动。')
                if knowledge is not None:
                    knowledge.close()
                return
        if not key:
            print('[web] Error: live mode requires DEEPSEEK_API_KEY (or use --mode demo).', file=sys.stderr)
            if knowledge is not None:
                knowledge.close()
            sys.exit(2)
        api_key = key

        def model_factory():
            return AnthropicModel(api_key, config, max_retries=2)
    else:
        if args.mode == 'auto':
            print('[web] APP_MODE=auto and DEEPSEEK_API_KEY is unset -> starting in zero-API demo mode.')

        def model_factory():
            return DemoInteractiveModel()

    app = create_app(
        root=ROOT,
        model_factory=model_factory,
        knowledge=knowledge,
        allow_demo_switch=not args.strict_auth,
        rate_limit_rpm=args.rate_limit,
        daily_token_budget=args.daily_token_budget,
    )

    import uvicorn
    print(f'[web] Starting E-Commerce Support Workbench ({args.effective_mode}) at http://{args.host}:{args.port}')
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level='info')
    finally:
        if knowledge is not None:
            knowledge.close()


if __name__ == '__main__':
    main()
