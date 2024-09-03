#__init__.py
import copy
import numpy as np

from warnings import warn
from typing import Optional, Any
from scipy.integrate import solve_ivp

class LogicError(Exception):
    pass

__version__ = "0.1.0"
__author__ = "Hyo-Eun Kang <hilucy00@pukyong.ac.kr>"
__all__ = ["step"]

