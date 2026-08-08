"""
FastAPI 接口模块 — Text2SQL 多步骤智能问答 API。

提供同步 /query 和 SSE 流式 /query/stream 两个接口。
基于 LangGraph 主图 build_main_graph() 做包装，不改动 graph/nodes 核心逻辑。

启动方式:
    python api.py
    或:
    uvicorn api:app --host 127.0.0.1 --port 8000 --reload

⚠ 安全提示: 当前未开启鉴权，仅限本地演示，切勿暴露到公网。
"""

import asyncio
import json
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.graph import build_main_graph
from app.log_utils import get_logger, new_trace_id
from app.state import SubTask, init_main_state
from trace_store import init_trace_db, save_trace

# ============================================================
# 常量
# ============================================================
_WARNING: str = "⚠ 未开启鉴权，仅限本地演示，勿暴露公网"
_REQUEST_TIMEOUT: float = float(
    __import__("os").environ.get("REQUEST_TIMEOUT_SEC", "120")
)
_SUBTASK_TIMEOUT: float = float(
    __import__("os").environ.get("SUBTASK_TIMEOUT_SEC", "30")
)

# 节点名称 → 中文描述映射（用于 SSE 事件 label 字段）
_NODE_LABELS: dict[str, str] = {
    "plan": "正在规划",
    "dispatch": "正在执行子任务",
    "aggregate": "正在聚合结果",
    "reflect": "正在反思审查",
    "finalize": "正在生成最终答案",
    "degrade": "正在生成降级回答",
}


# ============================================================
# FastAPI 应用初始化
# ============================================================
app = FastAPI(
    title="SQL Mind Agent",
    description="Text2SQL 多步骤智能问答 API — 基于 LangGraph",
    version="0.1.0",
)


@app.on_event("startup")
async def _startup_warning() -> None:
    """启动时打印安全警告并初始化 traces 表。"""
    print()
    print("=" * 60)
    print(f"  {_WARNING}")
    print("=" * 60)
    print()
    # 初始化 trace 落库表
    db_path = init_trace_db()
    print(f"  trace_store 初始化完成: {db_path}")
    print()


# ============================================================
# 请求 / 响应模型
# ============================================================
class QueryRequest(BaseModel):
    """同步查询请求体。"""

    question: str = Field(
        ...,
        description="用户自然语言问题",
        examples=["Customers 表有多少条记录？"],
    )
    db_id: str = Field(
        ...,
        description="目标数据库标识符",
        examples=["concert_singer"],
    )


class QueryResponse(BaseModel):
    """同步查询响应体。"""

    trace_id: str = Field(..., description="唯一链路追踪 ID")
    final_answer: str | None = Field(None, description="最终自然语言答案")
    status: str = Field(..., description="执行状态: done / degraded / error")
    plan: list[dict] = Field(default_factory=list, description="执行计划（子任务列表）")
    iterations: int = Field(..., description="实际迭代轮次（含重试）")


class ErrorResponse(BaseModel):
    """统一错误响应体。"""

    trace_id: str
    error: str
    detail: str | None = None


# ============================================================
# 内部工具函数
# ============================================================
def _serialize_plan(plan: list[SubTask]) -> list[dict]:
    """将 SubTask 列表序列化为 API 友好的 dict 列表。

    提取核心字段，避免在响应中暴露完整的内部状态。

    Args:
        plan: 子任务列表。

    Returns:
        list[dict]: 序列化后的计划列表。
    """
    result: list[dict] = []
    for s in plan:
        item: dict[str, Any] = {
            "id": s.get("id", ""),
            "description": s.get("description", ""),
            "depends_on": s.get("depends_on", []),
            "status": s.get("status", "unknown"),
        }
        if s.get("sql"):
            item["sql"] = s["sql"]
        if s.get("error"):
            item["error"] = s["error"]
        result.append(item)
    return result


def _extract_payload(node_name: str, node_output: dict) -> dict[str, Any]:
    """从节点输出中提取关键信息作为 SSE 事件的 payload。

    Args:
        node_name: 节点名称（如 "plan"、"dispatch"）。
        node_output: 节点返回的状态更新 dict。

    Returns:
        dict: 精简后的 payload，可直接序列化为 JSON。
    """
    payload: dict[str, Any] = {}

    if node_name == "plan" and "plan" in node_output:
        plan_list: list[dict] = node_output["plan"]
        payload["subtask_count"] = len(plan_list)
        payload["subtask_ids"] = [s["id"] for s in plan_list]

    elif node_name == "dispatch" and "completed" in node_output:
        completed: dict = node_output["completed"]
        success_count = sum(
            1 for t in completed.values() if t.get("status") == "success"
        )
        payload["total"] = len(completed)
        payload["success_count"] = success_count
        payload["failed_count"] = len(completed) - success_count

    elif node_name == "reflect" and "reflection" in node_output:
        refl: dict = node_output.get("reflection", {})
        payload["passed"] = refl.get("passed")

    return payload


# ============================================================
# 接口 1: POST /query — 同步查询
# ============================================================
@app.post(
    "/query",
    response_model=QueryResponse,
    responses={500: {"model": ErrorResponse}},
    summary="同步查询",
    description="运行完整多步骤 Text2SQL 流程，返回最终答案和完整执行计划。",
)
async def query(req: QueryRequest) -> QueryResponse | JSONResponse:
    """同步查询接口：运行完整多步骤 Text2SQL 流程，返回最终答案。

    流程: START → plan → dispatch → aggregate → reflect
            → [finalize(通过) | retry(重规划) | degrade(降级)]

    Args:
        req: 查询请求，包含 question 和 db_id。

    Returns:
        QueryResponse: 包含 trace_id、final_answer、status、plan、iterations。
    """
    trace_id: str = new_trace_id()
    log = get_logger(trace_id)
    log.info(
        f"POST /query  问题长度={len(req.question)}  db_id={req.db_id}  "
        f"request_timeout={_REQUEST_TIMEOUT}s  subtask_timeout={_SUBTASK_TIMEOUT}s"
    )

    final_state: dict[str, Any] = {}
    try:
        # 构建图并执行同步 invoke（整体请求级超时用 asyncio.wait_for 包裹）
        graph = build_main_graph()
        state = init_main_state(
            question=req.question,
            db_id=req.db_id,
            trace_id=trace_id,
        )

        async def _run_graph() -> dict[str, Any]:
            """在线程池外执行同步 graph.invoke，避免阻塞事件循环。"""
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, graph.invoke, state)

        final_state = await asyncio.wait_for(
            _run_graph(),
            timeout=_REQUEST_TIMEOUT,
        )
    except asyncio.TimeoutError:
        log.error(
            f"POST /query 请求超时  timeout={_REQUEST_TIMEOUT}s  "
            f"已等待 {_REQUEST_TIMEOUT}s 仍未完成，返回降级响应"
        )
        # 超时降级：尝试从已有状态构造部分响应
        error_response = JSONResponse(
            status_code=504,
            content={
                "trace_id": trace_id,
                "error": "RequestTimeout",
                "detail": (
                    f"请求超时（>{_REQUEST_TIMEOUT}s），"
                    f"已等待 {_REQUEST_TIMEOUT}s 仍未完成。"
                    f"建议简化问题或检查数据库/LLM 服务状态。"
                ),
            },
        )
        # 超时降级也写入 trace 记录（状态标记为 timeout）
        try:
            save_trace({
                "trace_id": trace_id,
                "question": req.question,
                "db_id": req.db_id,
                "final_answer": None,
                "status": "timeout",
                "iteration": 0,
                "plan": [],
                "reflection": {
                    "passed": False,
                    "reason": f"请求超时（>{_REQUEST_TIMEOUT}s）",
                },
            })
        except Exception as save_exc:
            log.error(f"save_trace 超时落库异常: {save_exc}")
        return error_response
    except Exception as exc:
        log.error(f"POST /query 异常: {type(exc).__name__}: {exc}")
        error_response = JSONResponse(
            status_code=500,
            content={
                "trace_id": trace_id,
                "error": type(exc).__name__,
                "detail": str(exc),
            },
        )
        # 异常也落库
        try:
            save_trace({
                "trace_id": trace_id,
                "question": req.question,
                "db_id": req.db_id,
                "final_answer": None,
                "status": "error",
                "iteration": 0,
                "plan": [],
                "reflection": {
                    "passed": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                },
            })
        except Exception as save_exc:
            log.error(f"save_trace 异常落库异常: {save_exc}")
        return error_response

    final_answer: str | None = final_state.get("final_answer")
    status: str = final_state.get("status", "error")
    plan: list[SubTask] = final_state.get("plan", [])
    iteration: int = final_state.get("iteration", 0)

    log.info(
        f"POST /query 完成  status={status}  "
        f"plan子任务数={len(plan)}  iterations={iteration + 1}"
    )

    # 请求结束落库 trace 记录，便于事后 badcase 排查
    try:
        save_trace(final_state)
    except Exception as save_exc:
        log.error(f"save_trace 落库异常: {save_exc}")

    return QueryResponse(
        trace_id=trace_id,
        final_answer=final_answer,
        status=status,
        plan=_serialize_plan(plan),
        iterations=iteration + 1,
    )


# ============================================================
# 接口 2: GET /query/stream — SSE 流式查询
# ============================================================
@app.get(
    "/query/stream",
    summary="SSE 流式查询",
    description=(
        "实时推送每一步进度。事件: start → node_done(×N) → done | error。"
        "前端可根据 node 字段展示'正在规划/正在执行子任务/正在反思…'。"
    ),
)
async def query_stream(
    question: str = Query(..., description="用户自然语言问题"),
    db_id: str = Query(..., description="目标数据库标识符"),
) -> EventSourceResponse:
    """SSE 流式查询接口：实时推送每一步的进度。

    事件格式:
        event: start       data: {"trace_id": "..."}
        event: node_done   data: {"node": "plan", "label": "正在规划", "status": "completed", "payload": {...}}
        event: loop        data: {"node": "plan", "iteration": 2}（反思不通过重试时）
        event: done        data: {"trace_id": "...", "final_answer": "...", ...}
        event: error       data: {"trace_id": "...", "error": "..."}

    前端可据此实时展示当前进度条/步骤指示器。

    Args:
        question: 用户自然语言问题。
        db_id: 目标数据库标识符。

    Returns:
        EventSourceResponse: SSE 事件流。
    """
    trace_id: str = new_trace_id()
    log = get_logger(trace_id)

    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        """SSE 事件生成器，逐节点推送进度。"""
        # 用于 finally 块中落库的 accumulated_state 引用
        accumulated_state: dict[str, Any] = {}
        try:
            log.info(
                f"GET /query/stream 开始  问题长度={len(question)}  db_id={db_id}"
            )

            # 推送初始事件
            yield {
                "event": "start",
                "data": json.dumps({"trace_id": trace_id}, ensure_ascii=False),
            }

            # 构建图与初始状态
            graph = build_main_graph()
            state = init_main_state(
                question=question,
                db_id=db_id,
                trace_id=trace_id,
            )

            # 用 astream 流式执行，每个节点完成后 yield 一次
            # accumulated_state 合并所有 update，循环结束后即为最终状态
            accumulated_state = dict(state)
            last_node: str | None = None

            # 对每个 astream chunk 加超时检测，整体超时后走降级而非崩溃
            astream_iter = graph.astream(state, stream_mode="updates")
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        astream_iter.__anext__(),
                        timeout=_REQUEST_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    log.error(
                        f"GET /query/stream 请求超时  "
                        f"timeout={_REQUEST_TIMEOUT}s，返回降级响应"
                    )
                    yield {
                        "event": "error",
                        "data": json.dumps(
                            {
                                "trace_id": trace_id,
                                "error": "RequestTimeout",
                                "detail": (
                                    f"请求超时（>{_REQUEST_TIMEOUT}s），"
                                    f"建议简化问题或检查数据库/LLM 服务状态。"
                                ),
                            },
                            ensure_ascii=False,
                        ),
                    }
                    # 超时落库
                    accumulated_state["status"] = "timeout"
                    return
                except StopAsyncIteration:
                    break  # 正常结束

                for node_name, node_output in chunk.items():
                    # 合并状态更新
                    accumulated_state.update(node_output)

                    label = _NODE_LABELS.get(node_name, node_name)
                    payload = _extract_payload(node_name, node_output)

                    yield {
                        "event": "node_done",
                        "data": json.dumps(
                            {
                                "node": node_name,
                                "label": label,
                                "status": "completed",
                                "payload": payload,
                            },
                            ensure_ascii=False,
                        ),
                    }

                    last_node = node_name

            # 从累积状态中提取最终字段
            final_answer: str | None = accumulated_state.get("final_answer")
            final_status: str = accumulated_state.get("status", "error")
            final_iteration: int = accumulated_state.get("iteration", 0)
            final_plan: list[SubTask] = accumulated_state.get("plan", [])

            yield {
                "event": "done",
                "data": json.dumps(
                    {
                        "trace_id": trace_id,
                        "final_answer": final_answer,
                        "status": final_status,
                        "plan": _serialize_plan(final_plan),
                        "iterations": final_iteration + 1,
                    },
                    ensure_ascii=False,
                ),
            }
            log.info(f"GET /query/stream 完成  status={final_status}")

            # 正常结束落库
            try:
                save_trace(accumulated_state)
            except Exception as save_exc:
                log.error(f"save_trace 落库异常: {save_exc}")

        except Exception as exc:
            log.error(f"GET /query/stream 异常: {type(exc).__name__}: {exc}")
            yield {
                "event": "error",
                "data": json.dumps(
                    {"trace_id": trace_id, "error": str(exc)},
                    ensure_ascii=False,
                ),
            }
            # 异常落库
            accumulated_state["status"] = "error"
            try:
                save_trace(accumulated_state)
            except Exception as save_exc:
                log.error(f"save_trace 异常落库异常: {save_exc}")

    return EventSourceResponse(event_generator())


# ============================================================
# 接口 3: GET /health — 健康检查
# ============================================================
@app.get("/health", summary="健康检查")
async def health() -> dict[str, str]:
    """健康检查接口。

    Returns:
        dict: {"status": "ok"}。
    """
    return {"status": "ok"}


# ============================================================
# 全局异常处理 — 避免裸奔 500
# ============================================================
@app.exception_handler(Exception)
async def _global_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """全局异常兜底：未捕获异常统一返回结构化错误，不裸奔 500。

    Args:
        request: 触发出错的请求对象。
        exc: 未捕获的异常。

    Returns:
        JSONResponse: 结构化错误响应。
    """
    return JSONResponse(
        status_code=500,
        content={
            "trace_id": "-",
            "error": type(exc).__name__,
            "detail": str(exc),
        },
    )


# ============================================================
# 入口 — 直接启动 uvicorn
# ============================================================
if __name__ == "__main__":
    import uvicorn

    print(f"  {_WARNING}")
    print()
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
