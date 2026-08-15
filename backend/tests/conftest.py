"""
pytest 配置文件。
"""

import os
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 加载 .env 文件（优先 backend/.env，其次仓库根 .env）
from dotenv import load_dotenv
_env_path = _project_root / ".env"
if not _env_path.is_file():
    _env_path = _project_root.parent / ".env"
load_dotenv(_env_path, override=False)
