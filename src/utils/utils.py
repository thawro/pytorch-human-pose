"""Other utilities"""

import random
from datetime import datetime


def random_float(min: float, max: float):
    return random.random() * (max - min) + min


def get_current_date_and_time() -> str:
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    return dt_string
