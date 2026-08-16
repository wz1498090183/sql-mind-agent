"""
LoRA 权重合并脚本 — 将 GRPO 训练的 LoRA adapter 合并回基座模型，输出 vLLM 可加载的完整模型。

背景：
    grpo_sql.py 里 trainer.save_model 只保存了 LoRA adapter（adapter_model.safetensors），
    基座权重未写入；且训练时基座以 bitsandbytes 4-bit 加载，vLLM 无法直接加载该格式。
    本脚本以 bf16 全精度加载基座 → 合并 LoRA → 保存为单一完整权重，供 vLLM 直接 serve。

用法:
    python train/merge_lora.py
"""

import os
import sys
from pathlib import Path

# Windows 下 PyTorch 的 libiomp5md.dll 与其他库的 libomp.dll 会重复加载，
# 报 OMP Error #15；需在 import torch 之前设置该变量规避（Linux 下无副作用）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============================================================
# 项目路径 & 配置（模型路径解析与 grpo_sql.py 保持一致）
# ============================================================
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

_MODEL_LOCAL = str(_project_root.parent / "models" / "Qwen2.5-Coder-3B-Instruct")
_MODEL_MOUNT = "/models/Qwen2.5-Coder-3B-Instruct"
MODEL = (
    _MODEL_LOCAL if Path(_MODEL_LOCAL).exists()
    else _MODEL_MOUNT if Path(_MODEL_MOUNT).exists()
    else "Qwen/Qwen2.5-Coder-3B-Instruct"
)

# LoRA adapter 路径（grpo_sql.py 保存的输出）与合并结果输出目录。
# 训练产物已归入 grpo-train/saves/，与 grpo_sql.py 的 _project_root / "saves" 保持一致。
ADAPTER_DIR = str(_project_root / "saves" / "qwen-coder-3b-grpo-sql" / "final")
OUTPUT_DIR = str(_project_root / "saves" / "qwen-coder-3b-grpo-sql" / "merged")


def main():
    adapter_path = Path(ADAPTER_DIR) / "adapter_model.safetensors"
    if not adapter_path.is_file():
        raise FileNotFoundError(f"未找到 LoRA adapter: {adapter_path}")

    print(f"基座模型: {MODEL}")
    print(f"LoRA adapter: {ADAPTER_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")

    # bf16 全精度加载基座（vLLM 需要全精度，不能复用训练的 4-bit 量化）
    print("加载基座模型 (bf16)...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    # 加载 adapter 并合并（merge_and_unload 会把 LoRA 增量叠加进基座权重）
    print("加载并合并 LoRA...")
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model = model.merge_and_unload()

    print("保存合并后的完整模型...")
    model.save_pretrained(OUTPUT_DIR)

    # tokenizer 一并保存，vLLM 加载时与权重同目录
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"合并完成: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
