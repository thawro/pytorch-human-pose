"""Config constants"""

from pathlib import Path

from src.utils.utils import get_current_date_and_time

ROOT = Path(__file__).parent.parent.parent
RESULTS_PATH = ROOT / "results"
YAML_EXP_PATH = ROOT / "experiments"
DS_ROOT = ROOT / "data"
LOG_DEVICE_ID = 0 # which device is responsible for logging


NOW = get_current_date_and_time()
