# GRPO 训练镜像 — 独立于推理后端，含 GPU 训练重型依赖
# 基镜像自带 CUDA + PyTorch，避免手动装 CUDA
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 仅拷贝训练代码（自带 db_utils，不依赖 backend 的 app 包）
COPY train ./train

CMD ["python", "train/grpo_sql.py"]
