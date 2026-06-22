"""pytest 根配置：确保项目根在 sys.path，便于导入 dataset_gen / 既有模块。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
