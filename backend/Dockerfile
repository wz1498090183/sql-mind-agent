# Agent 在线推理后端镜像 — 仅含推理依赖，不拷贝 grpo-train 训练代码
FROM python:3.12-slim

WORKDIR /app

# 先装依赖，利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 仅拷贝推理服务代码（api.py / main.py / app/），杜绝训练代码进入镜像
COPY api.py main.py ./
COPY app ./app

# 日志目录
RUN mkdir -p /app/logs

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
