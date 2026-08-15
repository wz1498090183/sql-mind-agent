// api.js — 请求后端接口、SSE 流式通信
// 前端经 Nginx 反向代理访问后端：/api/** -> backend:8000/**
// 所有请求同源走 /api 前缀，天然解决跨域，后端无需开启 CORS。

const API_BASE = '/api';

// 健康检查：探测后端是否可用
export async function health() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`健康检查失败: HTTP ${res.status}`);
  return res.json();
}

// 同步查询（非流式，备用接口）
export async function query(question, dbId) {
  const res = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, db_id: dbId }),
  });
  if (!res.ok) throw new Error(`查询失败: HTTP ${res.status}`);
  return res.json();
}

// SSE 流式查询：返回 EventSource 实例，事件回调由 handlers 提供
// 事件: start / node_done / done / error
export function queryStream(question, dbId, handlers = {}) {
  const url =
    `${API_BASE}/query/stream` +
    `?question=${encodeURIComponent(question)}` +
    `&db_id=${encodeURIComponent(dbId)}`;

  const es = new EventSource(url);

  es.addEventListener('start', (e) => {
    handlers.onStart?.(JSON.parse(e.data));
  });
  es.addEventListener('node_done', (e) => {
    handlers.onNodeDone?.(JSON.parse(e.data));
  });
  es.addEventListener('done', (e) => {
    handlers.onDone?.(JSON.parse(e.data));
  });
  es.addEventListener('error', (e) => {
    // 服务端推送的 error 事件（带 data）或连接层错误（无 data）
    let data = null;
    try { data = e.data ? JSON.parse(e.data) : null; } catch (_) { /* 非 JSON */ }
    handlers.onError?.(data, es.readyState);
  });
  es.onerror = () => {
    // 连接层错误兜底
    if (es.readyState === EventSource.CLOSED) {
      handlers.onClose?.(es.readyState);
    }
  };

  return es;
}
