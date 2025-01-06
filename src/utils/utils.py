import enum
import logging
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)


class ActionMode(enum.Enum):
    STUDENT = 0
    TEACHER = 1
    UNCERTAIN = 2


def energy_value(state: Any, p_mat: np.ndarray) -> int:
    """
    Get energy value represented by s^T @ P @ s -> return a value
    """
    # print(f"state is: {state}")
    # print(f"p_mat: {p_mat}")
    return np.squeeze(np.asarray(state).T @ p_mat @ state)


def energy_value_2d(state: Any, p_mat: np.ndarray) -> np.ndarray:
    """
    Get energy value represented by s^T @ P @ s (state is a 2d vector) -> return a 1d numpy array
    """
    # print(f"state is: {state}")
    # print(f"p_mat: {p_mat}")
    s = np.asarray(state)
    sp = np.matmul(s, p_mat)

    return np.sum(sp * s, axis=1)
