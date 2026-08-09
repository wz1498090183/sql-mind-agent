"""
GRPO 强化学习训练脚本 — SQL 生成任务。
使用 TRL 库的 GRPOTrainer 对 Qwen2.5-Coder-3B 进行 GRPO 微调。

奖励函数（简化版）：
    1. reward_exec — 执行正确性奖励：结果集一致=1.0，可执行=0.2，失败=-0.5
    2. reward_format — 格式奖励：包含SELECT=0.2，否则=-0.2

数据格式：spider_complex.json（Alpaca 格式，含 instruction/output/db_id）
数据库：通过 SPIDER_DB_ROOT 环境变量配置（默认 ./spider/database）

用法:
    python train/grpo_sql.py
"""

import json
import os
import re
import sys
from pathlib import Path

# ============================================================
# 项目路径 & 环境配置
# ============================================================
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 加载 .env 文件中的环境变量（SPIDER_DB_ROOT 等）
_env_path = _project_root / ".env"
if _env_path.is_file():
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _key, _, _value = _line.partition("=")
            os.environ.setdefault(_key.strip(), _value.strip())

import torch
from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer

from app.db_utils import execute_sql

# ============================================================
# 配置常量
# ============================================================

# 模型路径（优先使用本地路径，不存在则回退到 HuggingFace ID）
_MODEL_LOCAL = str(_project_root / "model" / "Qwen2.5-Coder-3B-Instruct")
MODEL = _MODEL_LOCAL if Path(_MODEL_LOCAL).exists() else "Qwen/Qwen2.5-Coder-3B-Instruct"
# 数据文件路径
DATA = str(_project_root / "train" / "spider_complex.json")

# ============================================================
# 1. 构造数据集：只需要 prompt + 金标准
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
# Qwen 模型无默认 pad_token，设为 eos_token 避免 GRPO 采样时报警
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def build_prompt(instruction: str) -> str:
    """将 instruction 转为 Qwen chat template 格式的 prompt。"""
    msgs = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


raw = json.load(open(DATA, encoding="utf-8"))
rows = [{
    "prompt": build_prompt(x["instruction"]),
    "gold_sql": x["output"],          # 列名 gold_sql 会被 TRL 传入 reward_funcs
    "db_id": x.get("db_id", ""),
} for x in raw]
dataset = Dataset.from_list(rows)
print(f"数据集加载完成: {len(dataset)} 条样本")

# ============================================================
# 2. SQL 提取工具
# ============================================================
def extract_sql(text: str) -> str:
    """从模型生成文本中提取纯 SQL 语句。

    处理流程：
        1. 提取 ```sql ``` 代码块内容
        2. 匹配第一条 SELECT/WITH 开头的语句
        3. 去除末尾分号
    """
    m = re.search(r"```sql\s*(.*?)```", text, re.S | re.I)
    if m:
        text = m.group(1)
    m = re.search(r"(SELECT|WITH|INSERT|UPDATE|DELETE).*", text, re.S | re.I)
    sql = (m.group(0) if m else text).strip().rstrip(";")
    return sql


# ============================================================
# 3. Reward 函数（GRPO 核心）
# ============================================================
def reward_exec(completions: list[str], gold_sql: list[str], db_id: list[str], **kwargs) -> list[float]:
    """执行正确性奖励。

    使用 app.db_utils.execute_sql 执行 SQL 并比较结果集：
        - 结果集一致 → 1.0
        - 可执行但结果不一致 → 0.2（鼓励生成可执行 SQL）
        - 语法错 / 不可执行 → -0.5
    """
    rewards = []
    for comp, gold, dbid in zip(completions, gold_sql, db_id):
        pred_sql = extract_sql(comp)
        pred_res = execute_sql(dbid, pred_sql)
        gold_res = execute_sql(dbid, gold)

        if not pred_res["success"]:
            r = -0.5
        elif not gold_res["success"]:
            # gold 执行失败但生成 SQL 可执行 → 部分给分
            r = 0.2
        else:
            # 比较结果集（忽略行顺序）
            pred_rows = set(map(tuple, pred_res["rows"]))
            gold_rows = set(map(tuple, gold_res["rows"]))
            if pred_rows == gold_rows:
                r = 1.0
            else:
                r = 0.2
        rewards.append(r)
    return rewards


def reward_format(completions: list[str], **kwargs) -> list[float]:
    """格式奖励：鼓励输出包含 SELECT 关键字，惩罚明显跑飞的输出。"""
    return [0.2 if "select" in c.lower() else -0.2 for c in completions]


# ============================================================
# 4. LoRA + GRPO 配置
# ============================================================
lora = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

config = GRPOConfig(
    output_dir=str(_project_root / "saves" / "qwen-coder-3b-grpo-sql"),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_generations=4,            # 每个 prompt 采样几条（12G 显存别调太大）
    max_prompt_length=1536,
    max_completion_length=256,
    learning_rate=1e-6,
    num_train_epochs=1,
    logging_steps=5,
    save_steps=200,
    bf16=True,
    gradient_checkpointing=True,
    report_to="none",
)

# ============================================================
# 5. 4bit 加载模型 & 训练
# ============================================================
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map={"": 0},
)

# 4-bit 量化模型需要提前准备才能用于 LoRA 训练
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_exec, reward_format],  # 多个 reward 会自动相加
    args=config,
    train_dataset=dataset,
    peft_config=lora,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(str(_project_root / "saves" / "qwen-coder-3b-grpo-sql" / "final"))
