// uiRender.js — 页面渲染、loading、弹窗

// 节点图标（FontAwesome）与配色映射
const NODE_ICONS = {
  plan: 'fa-solid fa-list-check',
  dispatch: 'fa-solid fa-gears',
  aggregate: 'fa-solid fa-puzzle-piece',
  reflect: 'fa-solid fa-magnifying-glass-chart',
  finalize: 'fa-solid fa-circle-check',
  degrade: 'fa-solid fa-triangle-exclamation',
  done: 'fa-solid fa-champagne-glasses',
  error: 'fa-solid fa-circle-xmark',
};

const NODE_COLORS = {
  plan: 'plan', dispatch: 'dispatch', aggregate: 'aggregate',
  reflect: 'reflect', finalize: 'finalize', degrade: 'degrade',
  error: 'error', done: 'done',
};

const STATUS_LABELS = {
  idle: '就绪', running: '执行中…', done: '完成',
  degraded: '降级完成', error: '出错',
};

const $ = (id) => document.getElementById(id);

// HTML 转义，防 XSS
export function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text ?? '';
  return div.innerHTML;
}

// 更新状态栏
export function setStatus(state) {
  $('statusDot').className = `status-dot ${state}`;
  $('statusText').textContent = STATUS_LABELS[state] || state;
}

// 追加步骤卡片到时间线
export function addStep(node, label, detail = '', detailHtml = '') {
  const timeline = $('timeline');
  const icon = NODE_ICONS[node] || 'fa-solid fa-circle';
  const colorClass = NODE_COLORS[node] || 'plan';

  let detailContent = '';
  if (detailHtml) detailContent = detailHtml;
  else if (detail) detailContent = escapeHtml(detail);

  const div = document.createElement('div');
  div.className = 'step';
  div.innerHTML = `
    <div class="step-icon ${colorClass}"><i class="${icon}"></i></div>
    <div class="step-body">
      <div class="step-header">
        <span class="step-label">${escapeHtml(label)}</span>
      </div>
      ${detailContent ? `<div class="step-detail">${detailContent}</div>` : ''}
    </div>
  `;
  timeline.appendChild(div);
  div.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  return div;
}

// 轮次分隔线
export function addRoundSeparator(roundNum) {
  const timeline = $('timeline');
  const sep = document.createElement('div');
  sep.className = 'round-separator';
  sep.innerHTML = `
    <span class="sep-line"></span>
    <i class="fa-solid fa-rotate"></i> 第 ${roundNum} 轮重规划
    <span class="sep-line"></span>
  `;
  timeline.appendChild(sep);
}

// 渲染最终答案
export function renderAnswer({ status, final_answer, trace_id, iterations, plan }) {
  const answer = final_answer || '(无答案)';
  const iters = iterations || 0;
  const planLen = (plan || []).length;

  let cardClass = '';
  let title = '';
  if (status === 'done') {
    cardClass = ''; title = '📝 最终答案'; setStatus('done');
  } else if (status === 'degraded') {
    cardClass = 'degraded'; title = '⚠️ 降级回答（部分结果仅供参考）'; setStatus('degraded');
  } else {
    cardClass = 'error'; title = '❌ 执行异常'; setStatus('error');
  }

  $('answerArea').innerHTML = `
    <div class="answer-card ${cardClass}">
      <h3>${title}</h3>
      <div class="answer-text">${escapeHtml(answer)}</div>
    </div>
  `;
  $('metaRow').innerHTML =
    `trace_id: ${escapeHtml(trace_id)} &nbsp;|&nbsp; 状态: ${status} &nbsp;|&nbsp; ` +
    `迭代轮次: ${iters} &nbsp;|&nbsp; 子任务数: ${planLen}`;
}

// 渲染错误
export function renderError(msg, traceId) {
  addStep('error', '请求出错', msg);
  setStatus('error');
  $('answerArea').innerHTML = `
    <div class="answer-card error">
      <h3>❌ 错误</h3>
      <div class="answer-text">${escapeHtml(msg)}</div>
    </div>
  `;
  if (traceId) $('metaRow').innerHTML = `trace_id: ${escapeHtml(traceId)}`;
}

// 清空时间线 / 答案 / 元信息
export function clearBoard() {
  $('timeline').innerHTML = '';
  $('answerArea').innerHTML = '';
  $('metaRow').innerHTML = '';
}

// loading：切换提交按钮可用态 + 旋转图标
export function setLoading(loading) {
  const btn = $('submitBtn');
  btn.disabled = loading;
  const icon = btn.querySelector('i');
  if (icon) {
    icon.className = loading ? 'fa-solid fa-spinner fa-spin' : 'fa-solid fa-bolt';
  }
}

// 轻提示 toast
export function toast(message, type = 'info') {
  const box = $('toastBox');
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.textContent = message;
  box.appendChild(el);
  setTimeout(() => el.remove(), 3000);
}

// 通用弹窗
export function showModal(title, bodyHtml) {
  $('modalTitle').textContent = title;
  $('modalBody').innerHTML = bodyHtml;
  $('modalOverlay').classList.remove('hidden');
}

export function closeModal() {
  $('modalOverlay').classList.add('hidden');
}
