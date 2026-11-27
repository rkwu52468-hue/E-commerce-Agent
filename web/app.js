/* desk.cs interactive workbench client:
   - Procedural exe.dev paper craft: punched binder holes, right-edge type ticks, hand-drawn SVG pen underlines, mascot eyes
   - Real-time SSE streaming over POST /api/sessions/{sid}/chat/stream
   - Live SQLite order / refund / ticket / trace workbench with mutation flash animations
   - Visual 2-step confirmation approval / rejection gate */

const state = {
  sessionId: null,
  snapshot: null,
  busy: false,
  prevOrderStatuses: new Map(),
  prevRefundIds: new Set(),
  prevTicketIds: new Set(),
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

/* ============================================================
   1. exe.dev Paper Craft: Punched Holes, Ticks, Hand-Drawn Pen
   ============================================================ */
function seededRng(text) {
  let a = 2166136261;
  for (const ch of text) a = Math.imul(a ^ ch.charCodeAt(0), 16777619) >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const smoothstep = (a, b, x) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

function generatePenPath(len, H, rnd) {
  const rise = 0.6 + rnd() * 1.1;
  const lambda = 70 + rnd() * 65;
  const phase = rnd() * 6.28;
  const wave = 0.16 + rnd() * 0.2;
  const press = 1.2 + rnd() * 0.25;
  const flick = 0.55 + rnd() * 0.7;
  const knots = Array.from({ length: Math.ceil((len * 1.7 + 60) / 9) + 3 }, () => rnd() - 0.5);
  const jitter = (x) => {
    const t = x / 9, j = Math.floor(t), f = t - j, s = f * f * (3 - 2 * f);
    return knots[j] * (1 - s) + knots[j + 1] * s;
  };
  const top = [], bot = [];
  const n = Math.max(12, Math.ceil(len / 2));
  for (let i = 0; i <= n; i++) {
    const t = i / n, x = 1 + (len - 2) * t;
    const y = H * 0.56 - rise * t + wave * Math.sin((x / lambda) * Math.PI * 2 + phase)
      + jitter(x) * 0.22 - flick * smoothstep(0.86, 1, t);
    const w = (press + 0.7 * Math.exp(-x / 5) + jitter(x * 1.7 + 40) * 0.16) * (1 - 0.52 * smoothstep(0.8, 1, t));
    top.push([x, y - w / 2]);
    bot.push([x, y + w / 2]);
  }
  const pts = [...top, ...bot.reverse()];
  const d = 'M' + pts.map(([x, y]) => `${x.toFixed(2)} ${y.toFixed(2)}`).join('L') + 'Z';
  return `<path d="${d}"/>`;
}

function drawPenUnderlines() {
  for (const el of $$('.pen-nav, .cta-cmd .cmd')) {
    const w = el.getBoundingClientRect().width;
    if (!w) continue;
    const rnd = seededRng(el.textContent.trim());
    const before = 2 + rnd() * 2.5;
    const past = 3 + rnd() * 5;
    const len = w + before + past;
    const H = 9;
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${len.toFixed(1)} ${H}" width="${len.toFixed(1)}" height="${H}">${generatePenPath(len, H, rnd)}</svg>`;
    const url = `url("data:image/svg+xml,${encodeURIComponent(svg)}")`;
    el.style.setProperty('--pen', url);
    el.style.setProperty('--pen-left', `${(-before).toFixed(1)}px`);
    el.style.setProperty('--pen-width', `${len.toFixed(1)}px`);
  }
}

function punchBinderHoles(sheet) {
  const h = sheet.offsetHeight;
  if (!h) return;
  const PITCH = 68, TOP = 34, SIZE = 11, INSET = 20;
  const span = Math.max(1, h - TOP * 2);
  const n = Math.max(2, Math.min(20, Math.round(span / PITCH) + 1));
  const r = SIZE / 2;
  const layers = Array.from({ length: n }, (_, i) => TOP + (span * i) / (n - 1))
    .map((y) => `radial-gradient(circle ${r}px at ${INSET}px ${y.toFixed(2)}px, transparent ${(r - 0.4).toFixed(2)}px, #000 ${(r + 0.4).toFixed(2)}px)`)
    .join(', ');
  sheet.style.webkitMaskImage = layers;
  sheet.style.maskImage = layers;
  sheet.style.webkitMaskRepeat = 'no-repeat';
  sheet.style.maskRepeat = 'no-repeat';
  sheet.style.webkitMaskComposite = 'source-in';
  sheet.style.maskComposite = 'intersect';
}

function placeSheetTicks(sheet) {
  const sr = sheet.getBoundingClientRect();
  if (!sr.width) return;
  sheet.querySelectorAll(':scope > .tk-y').forEach((n) => n.remove());
  const seen = new Set();
  for (const el of sheet.querySelectorAll('[data-tick]')) {
    if (el.closest('.sheet') !== sheet) continue;
    const r = el.getBoundingClientRect();
    if (!r.height) continue;
    const kinds = el.dataset.tick.split(/\s+/);
    const positions = [];
    if (kinds.includes('cap')) positions.push(r.top - sr.top + 4);
    if (kinds.includes('base') || kinds.includes('bottom')) positions.push(r.bottom - sr.top - 4);
    for (const pos of positions) {
      const k = Math.round(pos);
      if (seen.has(k)) continue;
      seen.add(k);
      const tick = document.createElement('i');
      tick.className = 'tk-y';
      tick.setAttribute('aria-hidden', 'true');
      tick.style.top = `${pos.toFixed(1)}px`;
      sheet.append(tick);
    }
  }
}

function initPaperCraft() {
  const sheets = $$('[data-holes]');
  const update = () => {
    sheets.forEach((s) => {
      punchBinderHoles(s);
      placeSheetTicks(s);
    });
    drawPenUnderlines();
  };
  update();
  const ro = new ResizeObserver((entries) => {
    for (const e of entries) {
      punchBinderHoles(e.target);
      placeSheetTicks(e.target);
    }
  });
  sheets.forEach((s) => ro.observe(s));
  window.addEventListener('resize', () => requestAnimationFrame(drawPenUnderlines));
  document.fonts?.ready.then(update);

  // Subtle mascot eye tracking (exe.dev eyes.js homage)
  if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
    const svg = $('#mascot-svg');
    const leftEye = $('#eye-left');
    const rightEye = $('#eye-right');
    if (svg && leftEye && rightEye) {
      window.addEventListener('pointermove', (ev) => {
        const rect = svg.getBoundingClientRect();
        const cx = rect.left + rect.width * 0.22;
        const cy = rect.top + rect.height * 0.5;
        const angle = Math.atan2(ev.clientY - cy, ev.clientX - cx);
        const dist = Math.min(2.1, Math.hypot(ev.clientX - cx, ev.clientY - cy) / 140);
        const dx = Math.cos(angle) * dist;
        const dy = Math.sin(angle) * dist;
        leftEye.style.transform = `translate(${dx.toFixed(2)}px, ${dy.toFixed(2)}px)`;
        rightEye.style.transform = `translate(${dx.toFixed(2)}px, ${dy.toFixed(2)}px)`;
      }, { passive: true });
    }
  }
}

/* ============================================================
   2. Safe Markdown Rendering (XSS-proof)
   ============================================================ */
function escapeHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function renderInline(text) {
  let s = escapeHtml(text);
  s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  return s;
}

function renderMarkdown(raw) {
  const lines = String(raw ?? '').replace(/\r\n/g, '\n').split('\n');
  let html = '';
  let inList = false;
  let inTable = false;
  let tableHeaderDone = false;

  const closeBlocks = () => {
    if (inList) { html += '</ul>'; inList = false; }
    if (inTable) { html += '</tbody></table>'; inTable = false; tableHeaderDone = false; }
  };

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      closeBlocks();
      continue;
    }
    // Markdown table row
    if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
      const cells = trimmed.slice(1, -1).split('|').map((c) => c.trim());
      if (cells.every((c) => /^:?-{2,}:?$/.test(c))) continue;
      if (inList) { html += '</ul>'; inList = false; }
      if (!inTable) {
        html += '<table><thead><tr>' + cells.map((c) => `<th>${renderInline(c)}</th>`).join('') + '</tr></thead><tbody>';
        inTable = true;
        tableHeaderDone = true;
      } else {
        html += '<tr>' + cells.map((c) => `<td>${renderInline(c)}</td>`).join('') + '</tr>';
      }
      continue;
    }
    if (inTable) { html += '</tbody></table>'; inTable = false; tableHeaderDone = false; }

    // Bullet item
    if (/^[-*]\s+/.test(trimmed)) {
      if (!inList) { html += '<ul>'; inList = true; }
      html += `<li>${renderInline(trimmed.replace(/^[-*]\s+/, ''))}</li>`;
      continue;
    }
    if (inList) { html += '</ul>'; inList = false; }

    // Heading -> crisp bold paragraph
    if (/^#{1,4}\s+/.test(trimmed)) {
      html += `<p><strong>${renderInline(trimmed.replace(/^#{1,4}\s+/, ''))}</strong></p>`;
      continue;
    }
    html += `<p>${renderInline(trimmed)}</p>`;
  }
  closeBlocks();
  return html || '<p></p>';
}

/* ============================================================
   3. Toast & Status Helpers
   ============================================================ */
let toastTimer = null;
function showToast(msg) {
  const el = $('#toast');
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 3200);
}

function fmtMoney(cents) {
  if (typeof cents !== 'number') return '—';
  return `¥${(cents / 100).toFixed(2)}`;
}

function fmtK(n) {
  if (!n) return '0';
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

function setStreamStatus(text, mode = 'ready') {
  $('#stream-status').textContent = text;
  const dot = $('#stream-dot');
  dot.classList.remove('is-live', 'is-busy');
  if (mode === 'live') dot.classList.add('is-live');
  if (mode === 'busy') dot.classList.add('is-busy');
}

/* ============================================================
   4. State Rendering (Orders / Refunds / Tickets / Budget / Confirm)
   ============================================================ */
function applySnapshot(snap, { flashMutations = true } = {}) {
  state.snapshot = snap;
  const cid = snap.customer_id;

  // Identity rail + prompt prefix
  $('#identity-name').textContent = cid ? `${cid.toUpperCase()}（已验证）` : 'GUEST（访客）';
  $('#identity-meta').textContent = cid
    ? `绑定身份：${cid} · 可读写本人订单与工单`
    : '未登录状态 · 仅可查公开商品、政策与知识库';
  $('#identity-dot').classList.toggle('is-live', Boolean(cid));
  $('#mobile-id').textContent = cid ? cid.toUpperCase() : 'GUEST';
  $('#prompt-prefix').textContent = `${cid || 'guest'}@desk >>`;
  $('#session-id-label').textContent = snap.session_id;
  $('#copy-session-btn').dataset.copy = snap.session_id;

  // Active nav pen highlight
  for (const btn of $$('[data-switch-customer]')) {
    btn.setAttribute('aria-current', String(btn.dataset.switchCustomer === cid));
  }
  $('#nav-logout').setAttribute('aria-current', String(!cid));

  // Hero badges
  const modelMode = (snap.model_mode || 'unknown').toUpperCase();
  $('#badge-model').textContent = `MODE · ${modelMode}`;
  $('#badge-rag').textContent = snap.retrieval_mode
    ? `RAG · ${snap.retrieval_mode.toUpperCase()} RRF`
    : 'RAG · OFF';
  $('#foot-dataset').textContent = `DATASET · ${snap.dataset_version || 'simulated'}`;

  // Tab counts
  $('#cnt-orders').textContent = String((snap.orders || []).length);
  $('#cnt-aftersales').textContent = String((snap.refunds || []).length + (snap.tickets || []).length);
  $('#cnt-catalog').textContent = String((snap.products || []).length);
  $('#cnt-traces').textContent = String((snap.traces || []).length);

  renderOrders(snap, flashMutations);
  renderAftersales(snap, flashMutations);
  renderCatalog(snap.products || []);
  renderTraces(snap.traces || []);
  renderConfirmations(snap.pending_confirmations || []);
  renderBudget(snap.budget);
  requestAnimationFrame(drawPenUnderlines);
}

function renderOrders(snap, flashMutations) {
  const list = $('#orders-list');
  const orders = snap.orders || [];
  $('#orders-owner-label').textContent = snap.customer_id
    ? `客户 ${snap.customer_id} · 共 ${orders.length} 笔`
    : '请点击左侧切换模拟客户（alice/bob/charlie）';

  if (!orders.length) {
    list.innerHTML = `<li class="empty-row">${
      snap.customer_id
        ? '当前账号暂无订单记录。'
        : '当前为访客状态。点击左侧 <code>alice</code>、<code>bob</code> 或 <code>charlie</code> 即可实时加载其 SQLite 订单账本。'
    }</li>`;
    state.prevOrderStatuses.clear();
    return;
  }

  list.innerHTML = '';
  const nextMap = new Map();
  for (const o of orders) {
    nextMap.set(o.order_id, o.status);
    const changed = flashMutations && state.prevOrderStatuses.size > 0
      && state.prevOrderStatuses.get(o.order_id) !== o.status;
    const cancellable = o.status === 'pending_payment' || o.status === 'paid_unshipped';
    const li = document.createElement('li');
    if (changed) li.classList.add('is-flash');
    li.innerHTML = `
      <span class="sw sw-${escapeHtml(o.status)}"></span>
      <div class="row-main">
        <div class="row-title">
          <span>${escapeHtml(o.order_id)}</span>
          <span class="status-tag ${cancellable ? 'tag-cancellable' : o.status === 'delivered' ? 'tag-ok' : ''}">${escapeHtml(o.status)}</span>
        </div>
        <div class="row-desc">SKU ${escapeHtml(o.product_id)} × ${o.quantity} · ${escapeHtml(o.source || 'sqlite')}</div>
      </div>
      <div class="row-end">
        <span>${fmtMoney(o.total_cents)}</span>
        <button class="row-action-btn" type="button" data-quick-order="${escapeHtml(o.order_id)}" data-cancellable="${cancellable ? '1' : '0'}">${cancellable ? '申请取消' : '查详情'}</button>
      </div>
    `;
    list.append(li);
  }
  state.prevOrderStatuses = nextMap;
}

function renderAftersales(snap, flashMutations) {
  const rfList = $('#refunds-list');
  const tkList = $('#tickets-list');
  const refunds = snap.refunds || [];
  const tickets = snap.tickets || [];

  if (!refunds.length) {
    rfList.innerHTML = `<li class="empty-row">${snap.customer_id ? '暂无退款单；取消已付款未发货订单（如 A1003 / B2002）将自动创建原路退款单。' : '登录后展示本人退款单。'}</li>`;
  } else {
    rfList.innerHTML = '';
    for (const r of refunds) {
      const isNew = flashMutations && state.prevRefundIds.size > 0 && !state.prevRefundIds.has(r.refund_id);
      const li = document.createElement('li');
      if (isNew) li.classList.add('is-flash');
      li.innerHTML = `
        <span class="sw sw-${escapeHtml(r.status)}"></span>
        <div class="row-main">
          <div class="row-title">
            <span>${escapeHtml(r.refund_id)}</span>
            <span class="status-tag ${r.status === 'rejected' ? 'tag-escalated' : 'tag-ok'}">${escapeHtml(r.status)}</span>
          </div>
          <div class="row-desc">订单 ${escapeHtml(r.order_id)} · ${escapeHtml(r.reviewer_note || r.reason)}</div>
        </div>
        <div class="row-end">
          <span>${fmtMoney(r.amount_cents)}</span>
          <span>${escapeHtml((r.updated_at || '').slice(5, 16))}</span>
        </div>
      `;
      rfList.append(li);
    }
  }
  state.prevRefundIds = new Set(refunds.map((r) => r.refund_id));

  if (!tickets.length) {
    tkList.innerHTML = `<li class="empty-row">${snap.customer_id ? '暂无售后工单；遇到物流延误、拒退申诉或要求转人工时会自动建单。' : '登录后展示本人工单。'}</li>`;
  } else {
    tkList.innerHTML = '';
    for (const t of tickets) {
      const isNew = flashMutations && state.prevTicketIds.size > 0 && !state.prevTicketIds.has(t.ticket_id);
      const li = document.createElement('li');
      if (isNew) li.classList.add('is-flash');
      li.innerHTML = `
        <span class="sw ${t.escalate_to_human ? 'sw-escalated' : 'sw-blue'}"></span>
        <div class="row-main">
          <div class="row-title">
            <span>${escapeHtml(t.ticket_id)}</span>
            <span class="status-tag ${t.escalate_to_human ? 'tag-escalated' : ''}">${t.escalate_to_human ? 'HUMAN ESCALATION' : escapeHtml(t.assigned_queue)}</span>
          </div>
          <div class="row-desc">${t.order_id ? `订单 ${escapeHtml(t.order_id)} · ` : ''}${escapeHtml(t.summary)}</div>
        </div>
        <div class="row-end">
          <span>${escapeHtml(t.priority.toUpperCase())}</span>
          <span>${escapeHtml(t.status)}</span>
        </div>
      `;
      tkList.append(li);
    }
  }
  state.prevTicketIds = new Set(tickets.map((t) => t.ticket_id));
}

function renderCatalog(products) {
  const list = $('#catalog-list');
  list.innerHTML = '';
  for (const p of products) {
    const li = document.createElement('li');
    li.innerHTML = `
      <span class="sw sw-blue"></span>
      <div class="row-main">
        <div class="row-title"><span>${escapeHtml(p.product_id)}</span><span>${escapeHtml(p.name)}</span></div>
        <div class="row-desc">标称容量 ${p.capacity_ml} ml · ${escapeHtml(p.source || '')}</div>
      </div>
      <div class="row-end">
        <span>${fmtMoney(p.price_cents)}</span>
        <button class="row-action-btn" type="button" data-quick-sku="${escapeHtml(p.product_id)}">问价</button>
      </div>
    `;
    list.append(li);
  }
}

function renderTraces(traces) {
  const list = $('#traces-list');
  if (!traces.length) {
    list.innerHTML = '<li class="empty-row">尚无工具审计轨迹；发送消息后将实时记录 Hook 生命周期与耗时。</li>';
    return;
  }
  list.innerHTML = '';
  for (const tr of [...traces].reverse()) {
    const isErr = tr.ok === false;
    const li = document.createElement('li');
    li.innerHTML = `
      <span class="sw ${isErr ? 'sw-orange' : tr.event === 'Stop' ? 'sw-green' : 'sw-blue'}"></span>
      <div class="row-main">
        <div class="row-title">
          <span>${escapeHtml(tr.event)}</span>
          ${tr.tool ? `<span class="status-tag">${escapeHtml(tr.tool)}</span>` : ''}
          ${tr.error_code ? `<span class="status-tag tag-cancellable">${escapeHtml(tr.error_code)}</span>` : ''}
        </div>
        <div class="row-desc">turn=${tr.turn ?? '-'} · customer=${escapeHtml(tr.customer_id || 'guest')}${tr.resource_id ? ` · resource=${escapeHtml(tr.resource_id)}` : ''}</div>
      </div>
      <div class="row-end">
        <span>${tr.duration_ms != null ? `${tr.duration_ms} ms` : (tr.status || '')}</span>
      </div>
    `;
    list.append(li);
  }
}

function renderConfirmations(pending) {
  const dock = $('#confirm-dock');
  if (!pending.length) {
    dock.hidden = true;
    dock.innerHTML = '';
    return;
  }
  dock.hidden = false;
  dock.innerHTML = pending.map((item) => {
    const willRefund = item.preview?.will_create_refund;
    const amt = item.preview?.order?.total_cents;
    return `
      <div class="confirm-card" data-cid="${escapeHtml(item.confirmation_id)}">
        <div class="confirm-head">
          <span class="confirm-title">⚠ 高危写操作 · 两步二次确认门控</span>
          <span class="confirm-code">${escapeHtml(item.confirmation_id)}</span>
        </div>
        <div class="confirm-meta">
          拟取消订单 <strong>${escapeHtml(item.order_id)}</strong>（${fmtMoney(amt)}，当前状态 <code>${escapeHtml(item.preview?.order?.status || '')}</code>）·
          ${willRefund ? '取消后将<strong>自动创建原路退款单</strong>' : '待支付订单取消后<strong>不产生退款单</strong>'}。
        </div>
        <div class="confirm-actions">
          <button class="ui-d-btn btn-accent" type="button" data-approve-cid="${escapeHtml(item.confirmation_id)}">✓ 确认取消并执行</button>
          <button class="ui-d-btn btn-danger-outline" type="button" data-reject-cid="${escapeHtml(item.confirmation_id)}">✕ 放弃取消</button>
        </div>
      </div>
    `;
  }).join('');
}

function renderBudget(budget) {
  if (!budget) return;
  const cap = Math.max(1, budget.max_session_tokens || 200000);
  const cached = budget.session_cache_read_tokens || 0;
  const uncached = budget.session_input_tokens || 0;
  const output = budget.session_output_tokens || 0;
  const used = budget.session_tokens || (cached + uncached + output);
  const free = Math.max(0, cap - used);
  const hitRate = budget.session_cache_hit_rate;
  const pctEl = $('#cache-hit-pct');
  const nextPct = hitRate != null ? `HIT ${(hitRate * 100).toFixed(1)}%` : 'HIT 0.0%';
  if (pctEl.textContent !== nextPct) {
    pctEl.textContent = nextPct;
    pctEl.classList.remove('is-jump');
    void pctEl.offsetWidth;
    pctEl.classList.add('is-jump');
  }
  $('#seg-cached').style.flexGrow = String(cached);
  $('#seg-uncached').style.flexGrow = String(uncached);
  $('#seg-output').style.flexGrow = String(output);
  $('#seg-free').style.flexGrow = String(Math.max(free, used === 0 ? 1 : 0));

  $('#stat-cached').textContent = fmtK(cached);
  $('#stat-uncached').textContent = fmtK(uncached);
  $('#stat-output').textContent = fmtK(output);
  $('#stat-session').textContent = fmtK(used);
  $('#stat-session-cap').textContent = fmtK(cap);
}

/* ============================================================
   5. Chat Transcript & SSE Streaming Engine
   ============================================================ */
function scrollChatBottom() {
  const feed = $('#chat-feed');
  feed.scrollTop = feed.scrollHeight;
}

function appendMessage(role, content, { meta = '', streaming = false } = {}) {
  const feed = $('#chat-feed');
  const article = document.createElement('article');
  article.className = `msg msg-${role}`;
  article.innerHTML = `
    <div class="msg-meta">
      <span>${escapeHtml(meta || (role === 'user' ? 'CUSTOMER' : 'AGENT'))}</span>
      <span>${new Date().toLocaleTimeString('zh-CN', { hour12: false })}</span>
    </div>
    <div class="msg-tools" hidden></div>
    <div class="msg-content">${streaming ? '<span class="caret"></span>' : renderMarkdown(content)}</div>
  `;
  feed.append(article);
  scrollChatBottom();
  return article;
}

function addToolChip(article, payload) {
  const container = article.querySelector('.msg-tools');
  container.hidden = false;
  const chip = document.createElement('span');
  const code = payload.error_code;
  const isDanger = code === 'ACCESS_DENIED_OR_NOT_FOUND' || code === 'AUTHENTICATION_REQUIRED' || code === 'TOOL_TIMEOUT';
  chip.className = 'tool-chip' + (isDanger ? ' is-danger' : payload.ok === false ? ' is-err' : '');
  const statusTxt = payload.ok ? 'OK' : (code || 'ERR');
  const durTxt = payload.duration_ms != null ? ` ${payload.duration_ms}ms` : '';
  chip.textContent = `${payload.tool}${payload.resource_id ? `(${payload.resource_id})` : ''} → ${statusTxt}${durTxt}`;
  container.append(chip);
  scrollChatBottom();
}

async function sendChat(text) {
  const message = text.trim();
  if (!message || state.busy || !state.sessionId) return;
  state.busy = true;
  $('#send-btn').disabled = true;
  setStreamStatus('STREAMING', 'busy');

  const cid = state.snapshot?.customer_id || 'guest';
  appendMessage('user', message, { meta: `客户 · ${cid}` });
  const assistantCard = appendMessage('assistant', '', {
    meta: `AGENT · ${(state.snapshot?.model_mode || '').toUpperCase()}`,
    streaming: true,
  });
  const contentEl = assistantCard.querySelector('.msg-content');
  const transitBar = $('#transit-bar');

  let accumulated = '';
  try {
    const resp = await fetch(`/api/sessions/${state.sessionId}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });
    const rem = resp.headers.get('X-RateLimit-Remaining');
    if (rem !== null) $('#rate-limit-meta').textContent = `RPM ${rem} LEFT`;
    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
      throw new Error(errData.detail || `HTTP ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf('\n\n')) !== -1) {
        const rawFrame = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        if (!rawFrame.trim() || rawFrame.startsWith(':')) continue;
        let eventName = 'message';
        let dataStr = '';
        for (const line of rawFrame.split('\n')) {
          if (line.startsWith('event:')) eventName = line.slice(6).trim();
          else if (line.startsWith('data:')) dataStr += line.slice(5).trim();
        }
        if (!dataStr) continue;
        const payload = JSON.parse(dataStr);

        if (eventName === 'start' && payload.rate_limit_remaining != null) {
          $('#rate-limit-meta').textContent = `RPM ${payload.rate_limit_remaining} LEFT`;
        } else if (eventName === 'tool_start') {
          transitBar.hidden = false;
          $('#transit-tool-name').textContent = `TOOL · ${payload.tool}`;
          $('#transit-tool-detail').textContent = payload.resource_id ? `target=${payload.resource_id}` : 'executing...';
        } else if (eventName === 'tool_end') {
          transitBar.hidden = true;
          addToolChip(assistantCard, payload);
        } else if (eventName === 'delta') {
          transitBar.hidden = true;
          accumulated += payload.text || '';
          contentEl.innerHTML = renderMarkdown(accumulated) + '<span class="caret"></span>';
          scrollChatBottom();
        } else if (eventName === 'error') {
          transitBar.hidden = true;
          accumulated += `\n\n**执行中止（${payload.code || 'ERROR'}）**：${payload.message || '未知错误'}`;
          contentEl.innerHTML = renderMarkdown(accumulated);
        } else if (eventName === 'state') {
          applySnapshot(payload, { flashMutations: true });
        } else if (eventName === 'done') {
          transitBar.hidden = true;
          contentEl.innerHTML = renderMarkdown(accumulated || '（本轮未返回文本）');
        }
      }
    }
    contentEl.innerHTML = renderMarkdown(accumulated || '（本轮未返回文本）');
  } catch (err) {
    transitBar.hidden = true;
    contentEl.innerHTML = renderMarkdown(`**请求未完成**：${err.message}`);
    showToast(err.message);
  } finally {
    state.busy = false;
    $('#send-btn').disabled = false;
    setStreamStatus('READY', 'live');
    scrollChatBottom();
  }
}

/* ============================================================
   6. Session & Confirmation Actions
   ============================================================ */
async function switchAccount(customerId) {
  if (!state.sessionId || state.busy) return;
  const resp = await fetch(`/api/sessions/${state.sessionId}/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ customer_id: customerId, demo_switch: true }),
  });
  const data = await resp.json();
  if (!resp.ok) {
    showToast(data.detail || '切换失败');
    return;
  }
  state.prevOrderStatuses.clear();
  state.prevRefundIds.clear();
  state.prevTicketIds.clear();
  applySnapshot(data.state, { flashMutations: false });
  appendMessage('system', `身份已切换为已验证模拟账号 \`${customerId}\`，已开启隔离上下文并同步右侧订单账本。`, { meta: 'AUTH · SWITCHED' });
  showToast(`已切换为 ${customerId}`);
}

async function logoutAccount() {
  if (!state.sessionId || state.busy) return;
  const resp = await fetch(`/api/sessions/${state.sessionId}/logout`, { method: 'POST' });
  const data = await resp.json();
  if (resp.ok) {
    state.prevOrderStatuses.clear();
    applySnapshot(data.state, { flashMutations: false });
    appendMessage('system', '已退出登录，回退至访客（guest）隔离会话，待确认令牌已清空。', { meta: 'AUTH · LOGOUT' });
    showToast('已切换为访客');
  }
}

async function handleConfirmation(cid, action) {
  if (!state.sessionId || state.busy) return;
  state.busy = true;
  setStreamStatus(action === 'approve' ? 'COMMITTING' : 'REJECTING', 'busy');
  try {
    const resp = await fetch(`/api/sessions/${state.sessionId}/confirmations/${cid}/${action}`, { method: 'POST' });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || '确认操作失败');
    applySnapshot(data.state, { flashMutations: true });
    const res = data.result || {};
    if (action === 'approve' && res.ok) {
      const d = res.data;
      let msg = `已通过可视化门控确认并取消订单 **\`${d.order.order_id}\`**（\`${d.previous_status}\` → \`cancelled\`）。`;
      if (d.refund_created) {
        msg += `\n已自动创建原路退款单 **\`${d.refund_created.refund_id}\`**（${fmtMoney(d.refund_created.amount_cents)}，状态 \`${d.refund_created.status}\`），可在右侧 REFUND/TK 标签查看。`;
      }
      appendMessage('assistant', msg, { meta: '2-STEP GATE · COMMITTED' });
      showToast(`订单 ${d.order.order_id} 已取消`);
    } else if (action === 'reject' && res.ok) {
      appendMessage('assistant', `已拒绝并销毁待确认令牌 \`${cid}\`，订单状态未变更。`, { meta: '2-STEP GATE · REJECTED' });
      showToast('已放弃取消操作');
    } else {
      appendMessage('system', `确认操作未完成：${res.error?.message || '未知状态'}`, { meta: '2-STEP GATE' });
    }
  } catch (err) {
    showToast(err.message);
  } finally {
    state.busy = false;
    setStreamStatus('READY', 'live');
  }
}

async function bootstrap() {
  initPaperCraft();

  // Theme toggle
  const savedTheme = localStorage.getItem('desk_cs_theme');
  if (savedTheme === 'dark' || savedTheme === 'light') {
    document.documentElement.setAttribute('data-theme', savedTheme);
  }
  const syncThemeLabel = () => {
    const cur = document.documentElement.getAttribute('data-theme');
    $('#theme-label').textContent = cur === 'dark' ? 'PAPER' : 'DARK';
  };
  syncThemeLabel();
  $('#theme-toggle').addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('desk_cs_theme', next);
    syncThemeLabel();
  });

  // Fetch health + create session
  try {
    const [healthRes, sessRes] = await Promise.all([
      fetch('/api/health').then((r) => r.json()),
      fetch('/api/sessions', { method: 'POST' }).then((r) => r.json()),
    ]);
    state.sessionId = sessRes.session_id;
    $('#rate-limit-meta').textContent = `RPM ${healthRes.rate_limit_rpm}/${healthRes.rate_limit_rpm}`;
    $('#badge-lf').textContent = healthRes.langfuse_enabled ? 'TRACE · LANGFUSE ON' : 'TRACE · LOCAL HOOKS';
    $('#trace-lf-label').textContent = healthRes.langfuse_enabled ? 'LANGFUSE EXPORT ON' : 'IN-PROCESS HOOKS';
    applySnapshot(sessRes.state, { flashMutations: false });
    setStreamStatus('READY', 'live');
  } catch (err) {
    setStreamStatus('OFFLINE', 'ready');
    showToast('后端连接失败：' + err.message);
  }

  // Ledger Tabs
  for (const tab of $$('.ledger-card [role="tab"]')) {
    tab.addEventListener('click', () => {
      for (const t of $$('.ledger-card [role="tab"]')) t.setAttribute('aria-selected', 'false');
      tab.setAttribute('aria-selected', 'true');
      const target = tab.dataset.tab;
      for (const pane of $$('.ledger-pane')) pane.hidden = pane.id !== `pane-${target}`;
    });
  }

  // Account Switch & Logout
  for (const btn of $$('[data-switch-customer]')) {
    btn.addEventListener('click', () => switchAccount(btn.dataset.switchCustomer));
  }
  $('#nav-logout').addEventListener('click', logoutAccount);
  $('#mobile-alice-btn')?.addEventListener('click', () => switchAccount('alice'));

  // Password login form
  $('#pw-login-form').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    if (!state.sessionId) return;
    const customer_id = $('#pw-customer').value;
    const password = $('#pw-password').value;
    $('#pw-msg').textContent = '验证中...';
    const resp = await fetch(`/api/sessions/${state.sessionId}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ customer_id, password, demo_switch: false }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      $('#pw-msg').textContent = data.detail || '校验失败';
      return;
    }
    $('#pw-password').value = '';
    $('#pw-msg').textContent = '';
    $('#pw-login-box').open = false;
    applySnapshot(data.state, { flashMutations: false });
    appendMessage('system', `已通过本地密码验证登录模拟账号 \`${customer_id}\`。`, { meta: 'AUTH · VERIFIED' });
  });

  // Copy command button
  $('#copy-session-btn').addEventListener('click', () => {
    const btn = $('#copy-session-btn');
    navigator.clipboard?.writeText(btn.dataset.copy).catch(() => {});
    btn.classList.add('is-copied');
    setTimeout(() => btn.classList.remove('is-copied'), 1500);
  });

  // Reset demo SQLite DB
  $('#reset-db-btn').addEventListener('click', async () => {
    if (!state.sessionId || state.busy) return;
    const resp = await fetch(`/api/sessions/${state.sessionId}/reset-demo-db`, { method: 'POST' });
    const data = await resp.json();
    if (resp.ok) {
      state.prevOrderStatuses.clear();
      state.prevRefundIds.clear();
      state.prevTicketIds.clear();
      applySnapshot(data.state, { flashMutations: false });
      appendMessage('system', '演示 SQLite 数据库已重置为初始种子状态，可重新演练取消订单与自动退款流程。', { meta: 'DB · RESET' });
      showToast('演示数据库已重置');
    } else {
      showToast(data.detail || '重置失败');
    }
  });

  // Clear chat display
  $('#clear-chat-btn').addEventListener('click', () => {
    $('#chat-feed').innerHTML = '';
    appendMessage('system', '已清空当前屏幕显示（服务端会话上下文与 SQLite 状态保持不变）。', { meta: 'CONSOLE · CLEARED' });
  });

  // Chat submit (Enter to send, Shift+Enter newline)
  const input = $('#chat-input');
  $('#chat-form').addEventListener('submit', (ev) => {
    ev.preventDefault();
    const text = input.value;
    if (!text.trim()) return;
    input.value = '';
    sendChat(text);
  });
  input.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter' && !ev.shiftKey && !ev.isComposing) {
      ev.preventDefault();
      $('#chat-form').requestSubmit();
    }
  });

  // Delegated clicks: scenario chips, confirmation approve/reject, order/SKU quick actions
  document.addEventListener('click', (ev) => {
    const sc = ev.target.closest('[data-prompt]');
    if (sc) { sendChat(sc.dataset.prompt); return; }

    const approve = ev.target.closest('[data-approve-cid]');
    if (approve) { handleConfirmation(approve.dataset.approveCid, 'approve'); return; }

    const reject = ev.target.closest('[data-reject-cid]');
    if (reject) { handleConfirmation(reject.dataset.rejectCid, 'reject'); return; }

    const qOrder = ev.target.closest('[data-quick-order]');
    if (qOrder) {
      const oid = qOrder.dataset.quickOrder;
      const can = qOrder.dataset.cancellable === '1';
      sendChat(can ? `帮我取消订单 ${oid}，原因是暂不需要。` : `帮我查一下订单 ${oid} 的详情和物流状态。`);
      return;
    }

    const qSku = ev.target.closest('[data-quick-sku]');
    if (qSku) {
      sendChat(`查询一下商品 ${qSku.dataset.quickSku} 的售价和规格容量。`);
    }
  });
}

bootstrap();
