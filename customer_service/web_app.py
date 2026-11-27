"""FastAPI + SSE Web Workbench backend for the E-Commerce AI Customer Support Agent."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import queue
import threading

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .observability import LangfuseReporter
from .web_sessions import (
    DemoInteractiveModel,
    SlidingWindowRateLimiter,
    WebSessionStore,
)

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / 'web'


class LoginRequest(BaseModel):
    customer_id: str = Field(..., min_length=1, max_length=40)
    password: str | None = Field(default=None, max_length=128)
    demo_switch: bool = Field(default=False)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


def _sse_frame(event_name: str, payload: dict) -> str:
    return f'event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n'


def _chunk_reply_for_stream(text: str, chunk_size: int = 24) -> list[str]:
    if not text:
        return []
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def create_app(*, root: Path = ROOT, model_factory=None, knowledge=None,
               seed_path: Path | None = None, database_path: Path | None = None,
               allow_demo_switch: bool = True, rate_limit_rpm: int = 30,
               daily_token_budget: int = 500000, tool_timeout_seconds: float = 10.0,
               langfuse: LangfuseReporter | None = None,
               web_dir: Path | None = None) -> FastAPI:
    root = Path(root)
    static_dir = Path(web_dir) if web_dir is not None else (root / 'web')
    seed = Path(seed_path) if seed_path is not None else (root / 'data/shop-action-seed.json')
    db = Path(database_path) if database_path is not None else (root / 'runtime/web-shop-actions.db')
    factory = model_factory if model_factory is not None else DemoInteractiveModel
    lf = langfuse if langfuse is not None else LangfuseReporter()

    store = WebSessionStore(
        root,
        factory,
        seed_path=seed,
        database_path=db,
        knowledge=knowledge,
        allow_demo_switch=allow_demo_switch,
        daily_token_budget=daily_token_budget,
        tool_timeout_seconds=tool_timeout_seconds,
        langfuse=lf,
    )
    limiter = SlidingWindowRateLimiter(max_requests=rate_limit_rpm, window_seconds=60.0)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            store.close_all()

    app = FastAPI(
        title='E-Commerce Support Agent Workbench',
        version='3.0.0',
        lifespan=lifespan,
    )
    app.state.session_store = store
    app.state.rate_limiter = limiter

    if static_dir.exists():
        app.mount('/static', StaticFiles(directory=str(static_dir)), name='static')

    def _require_session(session_id: str):
        managed = store.get(session_id)
        if managed is None:
            raise HTTPException(status_code=404, detail='Session not found or expired')
        return managed

    def _enforce_rate_limit(request: Request, session_id: str) -> int:
        client_host = request.client.host if request.client else 'local'
        allowed, remaining, retry_after = limiter.check(f'{client_host}:{session_id}')
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f'请求过于频繁，触发速率限制（{limiter.max_requests} 次/分钟），请 {retry_after} 秒后再试。',
                headers={'Retry-After': str(max(1, int(round(retry_after))))},
            )
        return remaining

    @app.get('/')
    async def index():
        index_file = static_dir / 'index.html'
        if not index_file.exists():
            return JSONResponse({'status': 'ok', 'message': 'Web UI index.html not found'})
        return FileResponse(str(index_file), media_type='text/html; charset=utf-8')

    @app.get('/api/health')
    async def health():
        probe_model = factory()
        return {
            'status': 'ok',
            'model_mode': getattr(probe_model, 'mode', 'unknown'),
            'retrieval_mode': getattr(knowledge, 'retrieval_mode', None) if knowledge is not None else None,
            'knowledge_enabled': knowledge is not None,
            'allow_demo_switch': allow_demo_switch,
            'active_sessions': store.active_count(),
            'rate_limit_rpm': limiter.max_requests,
            'langfuse_enabled': lf.enabled,
            'simulated': True,
        }

    @app.post('/api/sessions')
    async def create_session():
        managed = store.create_session()
        return {'session_id': managed.session_id, 'state': store.snapshot_state(managed)}

    @app.get('/api/sessions/{session_id}/state')
    async def get_state(session_id: str):
        managed = _require_session(session_id)
        return store.snapshot_state(managed)

    @app.post('/api/sessions/{session_id}/login')
    async def login(session_id: str, body: LoginRequest):
        managed = _require_session(session_id)
        ok, message = store.login(managed, body.customer_id, body.password, demo_switch=body.demo_switch)
        if not ok:
            raise HTTPException(status_code=401, detail=message)
        return {'ok': True, 'message': message, 'state': store.snapshot_state(managed)}

    @app.post('/api/sessions/{session_id}/logout')
    async def logout(session_id: str):
        managed = _require_session(session_id)
        store.logout(managed)
        return {'ok': True, 'state': store.snapshot_state(managed)}

    @app.post('/api/sessions/{session_id}/confirmations/{confirmation_id}/approve')
    async def approve_confirmation(session_id: str, confirmation_id: str):
        managed = _require_session(session_id)
        with managed.lock:
            result = managed.live_session.approve_confirmation(confirmation_id, execute_now=True)
        return {'result': result, 'state': store.snapshot_state(managed)}

    @app.post('/api/sessions/{session_id}/confirmations/{confirmation_id}/reject')
    async def reject_confirmation(session_id: str, confirmation_id: str):
        managed = _require_session(session_id)
        with managed.lock:
            result = managed.live_session.reject_confirmation(confirmation_id)
        return {'result': result, 'state': store.snapshot_state(managed)}

    @app.post('/api/sessions/{session_id}/reset-demo-db')
    async def reset_demo_db(session_id: str):
        managed = _require_session(session_id)
        try:
            info = store.reset_demo_db()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return {'ok': True, 'info': info, 'state': store.snapshot_state(managed)}

    @app.post('/api/sessions/{session_id}/chat')
    async def chat_sync(session_id: str, body: ChatRequest, request: Request):
        managed = _require_session(session_id)
        remaining = _enforce_rate_limit(request, session_id)
        message = body.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail='Message cannot be empty')
        with managed.lock:
            reply = await asyncio.to_thread(managed.live_session.turn, message)
        return JSONResponse(
            {'reply': reply, 'state': store.snapshot_state(managed)},
            headers={'X-RateLimit-Remaining': str(remaining)},
        )

    @app.post('/api/sessions/{session_id}/chat/stream')
    async def chat_stream(session_id: str, body: ChatRequest, request: Request):
        managed = _require_session(session_id)
        remaining = _enforce_rate_limit(request, session_id)
        message = body.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail='Message cannot be empty')

        async def event_generator():
            event_queue: queue.Queue[tuple[str, dict]] = queue.Queue()

            def on_hook(entry: dict):
                ev = entry.get('event')
                if ev == 'PreToolUse':
                    event_queue.put(('tool_start', entry))
                elif ev == 'PostToolUse':
                    event_queue.put(('tool_end', entry))

            def worker():
                unsubscribe = managed.observer.subscribe(on_hook)
                try:
                    with managed.lock:
                        reply = managed.live_session.turn(message)
                    event_queue.put(('reply_ready', {'reply': reply}))
                except Exception as exc:
                    event_queue.put(('error', {'code': type(exc).__name__, 'message': str(exc)}))
                finally:
                    unsubscribe()
                    event_queue.put(('worker_done', {}))

            yield _sse_frame('start', {
                'session_id': session_id,
                'customer_id': managed.live_session.customer_id,
                'rate_limit_remaining': remaining,
            })
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()

            final_reply = None
            while True:
                if await request.is_disconnected():
                    break
                try:
                    kind, payload = await asyncio.to_thread(event_queue.get, True, 0.25)
                except queue.Empty:
                    yield ': keepalive\n\n'
                    continue
                if kind in ('tool_start', 'tool_end'):
                    yield _sse_frame(kind, payload)
                elif kind == 'reply_ready':
                    final_reply = payload['reply']
                    for piece in _chunk_reply_for_stream(final_reply):
                        yield _sse_frame('delta', {'text': piece})
                        await asyncio.sleep(0.012)
                elif kind == 'error':
                    yield _sse_frame('error', payload)
                elif kind == 'worker_done':
                    break

            state = store.snapshot_state(managed)
            yield _sse_frame('state', state)
            yield _sse_frame('done', {'ok': final_reply is not None, 'reply_chars': len(final_reply or '')})

        return StreamingResponse(
            event_generator(),
            media_type='text/event-stream; charset=utf-8',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
                'X-RateLimit-Remaining': str(remaining),
            },
        )

    return app
