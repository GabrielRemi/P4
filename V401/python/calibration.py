import numpy as np
from scipy import odr
from typing import Callable

def do_field_calibration(file: str) -> Callable[[float], float]:
    data = np.loadtxt(file)
    