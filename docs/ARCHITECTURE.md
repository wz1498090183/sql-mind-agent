# 架构与设计决策

本文记录 SQL Mind Agent 的核心架构和关键设计权衡，作为理解代码与面试问答的参考。核心实现位于 [backend/app](../backend/app/)。

## 一、整体架构

项目采用 **Plan-Execute-Reflect** 多智能体范式，用 LangGraph 组织为**双层图**：

```
用户问题 → plan(拆解) → dispatch(拓扑分层并行执行) → aggregate(聚合) → reflect(反思)
                                          ↓                                ├── passed → finalize → 答案
                              子任务: generate→validate→execute→retry      ├── 未达上限 → retry(重规划)
                                                                          └── 达上限 → degrade(降级)
```

- **主图**（`graph.py` + `nodes.py`）：负责任务规划、调度、聚合与反思。
- **子图**（`sql_subgraph.py`）：负责单个子任务的 SQL 生成→校验→执行→重试闭环。

主图与子图通过 LangGraph 的 `operator.add` reducer（`_completed_tasks`）自动合并并行子任务的返回结果。

## 二、关键设计决策（为什么这么做）

### 1. 为什么用双层图（主图 + 子图）而非单层？

**问题**：复杂问题需要拆解，但每个子任务又需要独立的 SQL 生成+自修复循环。单层图会把「拆解」和「SQL 自修复」两类不同粒度的逻辑耦合在一起。

**方案**：主图只做规划/调度/反思，每个子任务进入可复用的 SQL 求解子图。子图内部 `generate→validate→execute→retry` 有独立的循环语义，不污染主图状态机。

**权衡**：双层图状态传递更复杂（需 reducer 合并），但换来职责清晰、子图可独立测试与复用。

### 2. 为什么用 DAG + 拓扑排序而非顺序执行？

**问题**：同比/环比/对比类问题可拆成多个互不依赖的取数子任务，顺序执行会浪费大量时间（LLM 调用是主要耗时）。

**方案**：`_topological_levels` 用 Kahn 算法分层，层内并行（ThreadPoolExecutor）、层间串行，并检测依赖环。

**权衡**：增加了环检测与依赖校验的复杂度，但并行取数让复杂问题的端到端耗时接近单次查询。

### 3. 上游结果如何传给下游子任务？

**方案**：不落库、不拼 SQL 字符串，而是把上游 `{columns, rows}` 渲染成 Markdown 表格注入下游 prompt，并要求下游用 `WITH t1 AS (...) SELECT ...` 的 CTE 形式引用。同时加了一层「上游 id 误用校验」——检测 LLM 是否把 `t1` 直接当表名引用而没定义同名 CTE（这是依赖型 SQL 最高频的错误）。

**权衡**：CTE 内联数据对大结果集有限制（截断 20 行），但对 Spider 规模足够。

### 4. 为什么反思（reflect）是三维度审查而非单一判断？

**方案**：反思节点从**语义**（是否答非所问/遗漏）、**逻辑**（占比>100%、同比负无穷等业务常识）、**可执行性**（是否有 failed 子任务）三个维度审查，输出 `{passed, reason, fix_hint}`。不通过时 `fix_hint` 回传 planning 节点作为重规划依据。

**权衡**：比单靠「执行成功与否」判断更贴近业务正确性，但多一次 LLM 调用。反思失败时降级为 passed（不阻塞主流程）。

### 5. 为什么评估用三级兜底（而非直接比 SQL 字符串）？

**问题**：SQL 语义等价不等于字符串相等——列名拼写、列序、行序都可能导致 SQL 不同但结果一致。

**方案**（`evaluate.py:compare_sql_results`）：
1. 字段一致 → 按列名对齐后精确比较值（消除列序）；
2. 字段不一致 → 忽略列名/列序/行序，比较单元格值的多重集合；
3. 仍不一致 → 调 LLM 判断语义等价，并给出自然语言诊断。

**权衡**：三级兜底把确定性比较放在前、昂贵的 LLM 判断放在最后，兼顾准确率与成本。

### 6. 为什么用 GRPO 强化学习而非 SFT 微调？

**问题**：SFT（监督微调）只拟合标准答案，无法感知「SQL 能否实际执行」这类可验证的反馈信号。

**方案**：GRPO 把「SQL 执行成功 + 格式合规」作为 reward（见 `grpo-train`），无需 value 模型，采样多条候选按组内相对奖励优化。微调出的 LoRA 再合并回基座供 vLLM 部署。

**权衡**：GRPO 训练流程比 SFT 复杂，但 reward 信号直接对齐「可执行性」这一业务目标。

### 7. 为什么做多层超时与只读安全？

**方案**：请求级（120s）、子任务级（30s）、LLM 调用级（60s）、SQLite 执行级（5s + 看门狗线程 `interrupt()`）四层超时；SQL 只允许 `SELECT`/`WITH` 开头，`DROP/DELETE/UPDATE/INSERT` 词边界拦截。

**权衡**：只读约束牺牲了灵活性，但对「把生成的 SQL 交给真实数据库执行」的 Agent 而言，安全是不可妥协的底线。

### 8. 为什么引入值检索与检索式 few-shot？

**问题**：LLM 写 `WHERE product_type_code='Electronics'` 时会凭空猜枚举值，而库里实际是 `'Hardware'`；静态 few-shot 无法覆盖当前问题的具体模式。

**方案**：`get_value_samples` 抽取低基数类别列（`*_code`/`*_type`/`*_status`/`*_name` 等）的真实样本值注入 prompt；`fewshot.py` 按关键词 Jaccard 相似度从示例池检索最相似的 (问题, SQL) 对动态注入，未命中时退回静态 few-shot。

**权衡**：值检索增加若干轻量 `SELECT DISTINCT` 查询，检索式 few-shot 增加一次本地匹配，都以可忽略的延迟换取实体匹配与同类问题的准确率。

## 三、数据流

1. `init_main_state` 初始化 `MainState`（`state.py`，全部 TypedDict）。
2. `plan_node` 拉取 schema + 值样本，LLM 拆解为 `SubTask` 列表（含 `depends_on`）。
3. `dispatch_node` 拓扑分层，逐层并行调用子图求解。
4. 子图 `solve_subtask`：生成 SQL（含值检索+动态 few-shot）→ 四层校验 → 执行 → 失败重试。
5. `aggregate_node` 预计算摘要（环比/差值），LLM 生成自然语言答案。
6. `reflect_node` 三维度审查 → 通过则 `finalize`，否则 `retry` 或 `degrade`。
7. 请求结束 `save_trace` 落库 `traces.db`，供 badcase 排查。

## 四、可扩展方向

- Schema Linking 增强：表/列选择与相关性排序（当前全量 DDL 注入）。
- 多轮对话与追问澄清（当前每次请求独立）。
- 多数据库方言（当前仅 SQLite）。
