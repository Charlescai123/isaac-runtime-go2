import logging
import time

import torch
# import numpy as np

from isaacgym.torch_utils import to_torch
from src.utils.utils import ActionMode, energy_value_2d, check_safety
from src.physical_design import MATRIX_P


# torch.set_printoptions(sci_mode=False)


class Coordinator:

    def __init__(self, num_envs, device: str = "cuda"):
        self._device = device
        self._num_envs = num_envs
        self._plant_action = torch.zeros((self._num_envs, 6), dtype=torch.float32, device=device)
        self._action_mode = torch.full((self._num_envs,), ActionMode.STUDENT.value, dtype=torch.int64, device=device)

        # self._default_epsilon = 1  # Default epsilon
        self._last_action_mode = None

    def get_terminal_action(self,
                            hp_action: torch.Tensor,
                            ha_action: torch.Tensor,
                            plant_state: torch.Tensor,
                            safety_subset: torch.Tensor,
                            dwell_flag=None):
        """Given the system state and envelope boundary (epsilon), analyze the current safety status
        and return which action (hp/ha) to switch for control"""

        if dwell_flag is None:
            dwell_flag = torch.full((self._num_envs,), False, dtype=torch.bool, device=self._device)

        terminal_stance_ddq = torch.zeros((self._num_envs, 6), dtype=torch.float32, device=self._device)
        action_mode = torch.full((self._num_envs,), ActionMode.UNCERTAIN.value, dtype=torch.int64, device=self._device)

        self._last_action_mode = self._action_mode

        # Obtain all energies
        energy_2d = energy_value_2d(plant_state[:, 2:], to_torch(MATRIX_P, device=self._device))

        # Check current safety status
        is_unsafe = check_safety(error_state=plant_state, safety_subset=safety_subset)

        for i, energy in enumerate(energy_2d):

            # Display current system status based on energy
            if is_unsafe[i].item():
                logging.info(f"current system {i} is unsafe")
            else:
                logging.info(f"current system {i} is safe")

            # When Teacher disabled or deactivated
            if not torch.any(ha_action[i]) and bool(dwell_flag[i]) is False:
                logging.info("HA-Teacher is deactivated, use HP-Student's action instead")
                self._action_mode[i] = ActionMode.STUDENT.value
                self._plant_action[i] = hp_action[i]

                terminal_stance_ddq[i] = hp_action[i]
                action_mode[i] = ActionMode.STUDENT.value
                continue

            # Teacher activated
            if self._last_action_mode[i] == ActionMode.TEACHER.value:

                # Teacher Dwell time
                if dwell_flag[i]:
                    if ha_action[i] is None:
                        raise RuntimeError(f"Unrecognized HA-Teacher action {ha_action[i]} from {i} for dwelling")
                    else:
                        logging.info("Continue HA-Teacher action in dwell time")
                        self._action_mode[i] = ActionMode.TEACHER.value
                        self._plant_action[i] = ha_action[i]

                        terminal_stance_ddq[i] = ha_action[i]
                        action_mode[i] = ActionMode.TEACHER.value

                # Switch back to HPC
                else:
                    self._action_mode[i] = ActionMode.STUDENT.value
                    self._plant_action[i] = hp_action[i]
                    logging.info(f"Max HA-Teacher dwell time achieved, switch back to HP-Student control")

                    terminal_stance_ddq[i] = hp_action[i]
                    action_mode[i] = ActionMode.STUDENT.value

            elif self._last_action_mode[i] == ActionMode.STUDENT.value:

                # Inside safety subset
                if not is_unsafe[i].item():
                    self._action_mode[i] = ActionMode.STUDENT.value
                    self._plant_action[i] = hp_action[i]
                    logging.info(f"Continue HP-Student action")

                    terminal_stance_ddq[i] = hp_action[i]
                    action_mode[i] = ActionMode.STUDENT.value

                # Outside safety envelope (bounded by epsilon)
                else:
                    logging.info(f"Switch to HA-Teacher action for safety concern")
                    self._action_mode[i] = ActionMode.TEACHER.value
                    self._plant_action[i] = ha_action[i]

                    terminal_stance_ddq[i] = ha_action[i]
                    action_mode[i] = ActionMode.TEACHER.value
            else:
                raise RuntimeError(f"Unrecognized last action mode: {self._last_action_mode[i]} for {i}")

        return terminal_stance_ddq, action_mode

    @property
    def device(self):
        return self._device

    @property
    def plant_action(self):
        return self._plant_action

    @property
    def action_mode(self):
        return self._action_mode

    @property
    def last_action_mode(self):
        return self._last_action_mode
