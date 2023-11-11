"""Config constants"""

from pathlib import Path

from src.utils.utils import get_current_date_and_time

ROOT = Path(__file__).parent.parent.parent
DS_ROOT = ROOT / "data"

NOW = get_current_date_and_time()
