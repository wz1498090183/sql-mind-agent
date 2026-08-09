# SQL Mind Agent — Text2SQL 多步骤智能问答 Agent

基于 LangGraph 构建的 Text2SQL 智能问答系统，支持将复杂自然语言问题拆解为多个 SQL 子任务，执行查询后聚合反思，最终生成可信的自然语言答案。

## 架构

```
用户问题 → plan(任务拆解) → dispatch(并行执行子任务) → aggregate(聚合结果) → reflect(审查反思)
                                          ↓                                ├── passed → finalize → 最终答案
                              每个子任务: generate→validate→execute→retry  ├── 未达上限 → retry(重规划)
                                                                         └── 达上限 → degrade(降级回答)
```

## 快速开始

### 1. 环境要求

- Python 3.12
- SQLite 数据库（Spider 格式）

### 2. 安装依赖

```bash
pip install fastapi uvicorn sse-starlette langgraph langchain-openai loguru sqlparse
```

### 3. 配置 .env

在项目根目录创建 `.env` 文件：

```env
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-pro
SPIDER_DB_ROOT=./spider/database
```

### 4. 启动 Web 服务

```bash
# 方式一: 直接运行
python api.py

# 方式二: 使用 uvicorn
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

## API 接口

### `GET /health`

健康检查。

```bash
curl http://127.0.0.1:8000/health
```

**响应:** `{"status": "ok"}`

---

### `POST /query` (同步)

运行完整多步骤 Text2SQL 流程，返回最终答案。

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Customers 表有多少条记录？", "db_id": "department_store"}'
```

**请求体:**

| 字段 | 类型 | 说明 |
|------|------|------|
| `question` | `str` | 用户自然语言问题 |
| `db_id` | `str` | 目标数据库标识符 |

**响应体:**

| 字段 | 类型 | 说明 |
|------|------|------|
| `trace_id` | `str` | 链路追踪 ID |
| `final_answer` | `str\|null` | 最终自然语言答案 |
| `status` | `str` | 状态: `done` / `degraded` / `error` |
| `plan` | `list[dict]` | 执行计划（子任务列表） |
| `iterations` | `int` | 实际迭代轮次（含重试） |

---

### `GET /query/stream` (SSE 流式)

实时 SSE 流式推送每一步进度，前端可据此展示进度条/步骤指示器。

```bash
curl -N "http://127.0.0.1:8000/query/stream?question=Customers表有多少条记录&db_id=department_store"
```

**查询参数:**

| 参数 | 类型 | 说明 |
|------|------|------|
| `question` | `str` | 用户自然语言问题 |
| `db_id` | `str` | 目标数据库标识符 |

**事件流:**

| event | payload | 说明 |
|-------|---------|------|
| `start` | `{"trace_id": "..."}` | 连接建立，返回追踪 ID |
| `node_done` | `{"node": "plan", "label": "正在规划", "status": "completed", "payload": {...}}` | 单个节点完成 |
| `done` | `{"trace_id": "...", "final_answer": "...", "status": "done", ...}` | 全部流程完成 |
| `error` | `{"trace_id": "...", "error": "..."}` | 执行异常 |

`node_done` 事件中的 `node` 可能取值: `plan` / `dispatch` / `aggregate` / `reflect` / `finalize` / `degrade`

---

## CLI 使用

```bash
# 命令行直接查询
python main.py --question "Customers 表有多少条记录？" --db_id department_store

# 运行内置演示 Demo
python main.py --demo
```

## ⚠ 安全警告

**当前未开启鉴权，仅限本地演示使用，切勿暴露到公网。**

启动服务时会在终端打印以下警告：

```
============================================================
  ⚠ 未开启鉴权，仅限本地演示，勿暴露公网
============================================================
```

## 项目结构

```
sql-mind-agent/
├── api.py                  # FastAPI Web 服务（同步 + SSE 流式）
├── main.py                 # CLI 命令行入口
├── run_soul_case.py        # Demo 演示脚本（3 个 Soul Case）
├── app/
│   ├── graph.py            # 主图编排（LangGraph StateGraph）
│   ├── nodes.py            # 6 个主图节点实现
│   ├── sql_subgraph.py     # SQL 求解子图（生成→校验→执行→重试）
│   ├── state.py            # TypedDict 状态定义
│   ├── llm_client.py       # LLM 客户端封装
│   ├── db_utils.py         # SQLite 数据库操作工具
│   ├── log_utils.py        # 日志模块（loguru）
│   └── test/               # 测试目录
├── spider/database/        # SQLite 数据库文件（Spider 格式）
└── .env                    # 环境配置
```
