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

# 加载 .env 文件
from dotenv import load_dotenv
load_dotenv(_project_root / ".env", override=False)
