// chatStore.js — 会话状态管理（一次查询的运行时状态集中管理）

// 单一状态对象，供 main.js / uiRender.js 共享读写
const store = {
  traceId: null,
  status: 'idle',        // idle | running | done | degraded | error
  currentRound: 0,       // 当前轮次（反思重规划时 +1）
  nodeCountInRound: 0,
  seenPlanInRound: false,
  eventSource: null,     // 当前 SSE 连接

  // 重置为初始状态
  reset() {
    this.traceId = null;
    this.status = 'idle';
    this.currentRound = 0;
    this.nodeCountInRound = 0;
    this.seenPlanInRound = false;
  },

  // 终止当前 SSE 连接
  closeStream() {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
  },

  // 记录一个节点完成事件，返回是否需要插入轮次分隔线
  recordNode(node) {
    let needSeparator = false;
    if (node === 'plan') {
      if (this.seenPlanInRound) {
        this.currentRound += 1;
        this.nodeCountInRound = 0;
        needSeparator = true;
      }
      this.seenPlanInRound = true;
    }
    this.nodeCountInRound += 1;
    return needSeparator;
  },
};

export default store;
