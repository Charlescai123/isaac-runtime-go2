import enum
import logging
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)


class ActionMode(enum.Enum):
    STUDENT = 1
    TEACHER = 2


def energy_value(state: Any, p_mat: np.ndarray) -> int:
    """
    Get energy value represented by s^T @ P @ s
    """
    # print(f"state is: {state}")
    # print(f"p_mat: {p_mat}")
    return np.squeeze(np.asarray(state).T @ p_mat @ state)
