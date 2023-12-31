"""Other utilities"""

import random
from datetime import datetime


def random_float(min: float, max: float):
    return random.random() * (max - min) + min


def get_current_date_and_time() -> str:
    now = datetime.now()
    # dt_string = now.strftime("%Y-%m-%d_%H:%M")
    dt_string = now.strftime("%m-%d_%H:%M")
    return dt_string
