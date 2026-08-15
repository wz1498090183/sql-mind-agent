// main.js — 入口文件：绑定交互事件，编排 api / chatStore / uiRender

import { health, queryStream } from './api.js';
import store from './chatStore.js';
import {
  setStatus, addStep, addRoundSeparator, renderAnswer, renderError,
  clearBoard, setLoading, toast, closeModal,
} from './uiRender.js';

const $ = (id) => document.getElementById(id);
const questionInput = $('questionInput');
const dbSelect = $('dbSelect');
const submitBtn = $('submitBtn');
const clearBtn = $('clearBtn');

// 启动时探测后端健康状态
async function bootstrap() {
  try {
    await health();
    setStatus('idle');
  } catch (_) {
    toast('后端服务不可用，请确认已执行 docker compose up -d', 'error');
    setStatus('error');
  }
}

// 处理节点事件
function handleNodeDone(data) {
  const node = data.node;
  const label = data.label || node;
  const payload = data.payload || {};

  const needSeparator = store.recordNode(node);
  if (needSeparator) addRoundSeparator(store.currentRound + 1);

  let detail = '';
  if (node === 'plan' && payload.subtask_count !== undefined) {
    detail = `拆解出 ${payload.subtask_count} 个子任务`;
    if (payload.subtask_ids) detail += ` (${payload.subtask_ids.join(', ')})`;
  } else if (node === 'dispatch') {
    const total = payload.total || 0;
    const ok = payload.success_count || 0;
    const fail = payload.failed_count || 0;
    detail = `子任务执行完成: 共 ${total} 个`;
    if (ok > 0) detail += `, ✅ ${ok} 成功`;
    if (fail > 0) detail += `, ❌ ${fail} 失败`;
  } else if (node === 'reflect') {
    detail = payload.passed === true
      ? '审查通过 ✓'
      : (payload.passed === false ? '审查不通过，将重规划…' : '审查中…');
  } else if (node === 'aggregate') {
    detail = '正在汇总各子任务结果，生成自然语言答案';
  } else if (node === 'finalize') {
    detail = '反思通过，答案已确认为最终结果';
  } else if (node === 'degrade') {
    detail = '已达最大重试次数，生成降级回答';
  }

  addStep(node, label, detail);
}

// 开始查询
function startQuery() {
  const question = questionInput.value.trim();
  if (!question) return;

  store.closeStream();
  clearBoard();
  store.reset();
  store.status = 'running';
  setStatus('running');
  setLoading(true);

  store.eventSource = queryStream(question, dbSelect.value, {
    onStart: (d) => {
      store.traceId = d.trace_id;
      addStep('plan', '已连接', `trace_id: ${d.trace_id}`);
    },
    onNodeDone: handleNodeDone,
    onDone: (d) => {
      store.status = d.status || 'done';
      renderAnswer(d);
      setLoading(false);
      store.closeStream();
    },
    onError: (d, readyState) => {
      store.status = 'error';
      if (d) {
        // 服务端推送的 error 事件
        renderError(d.error || d.detail || '未知错误', d.trace_id);
      } else if (readyState === EventSource.CLOSED) {
        // 连接层错误
        addStep('error', '连接断开', 'SSE 连接已关闭，可能是服务端或网络问题');
        setStatus('error');
      }
      setLoading(false);
      store.closeStream();
    },
    onClose: () => {
      // 兜底：异常关闭且仍在运行态
      if (store.status === 'running') {
        addStep('error', '连接中断', 'SSE 连接意外关闭');
        setStatus('error');
      }
      setLoading(false);
    },
  });
}

// 清空
function clearAll() {
  store.closeStream();
  clearBoard();
  store.reset();
  setLoading(false);
  setStatus('idle');
  questionInput.focus();
}

questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') startQuery();
});
submitBtn.addEventListener('click', startQuery);
clearBtn.addEventListener('click', clearAll);
$('modalClose').addEventListener('click', closeModal);

bootstrap();
