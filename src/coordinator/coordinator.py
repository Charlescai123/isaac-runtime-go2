import torch
import numpy as np

# from src.logger.logger import Logger, plot_trajectory
from src.utils.utils import ActionMode, energy_value, energy_value_2d
from src.physical_design import MATRIX_P

np.set_printoptions(suppress=True)


class Coordinator:

    def __init__(self, num_envs):
        self._num_envs = num_envs
        self._plant_action = np.zeros((self._num_envs, 6))
        self._action_mode = np.full(self._num_envs, ActionMode.STUDENT)
        self._default_epsilon = 1  # Default epsilon
        self._last_action_mode = None

    def get_terminal_action(self, hp_action, ha_action, plant_state, epsilon=None, dwell_flag=None):

        if epsilon is None:
            epsilon = np.full(self._num_envs, self._default_epsilon)

        if dwell_flag is None:
            dwell_flag = np.full(self._num_envs, False)

        ha_action = ha_action.cpu().numpy()
        terminal_stance_ddq_rtn = np.zeros((self._num_envs, 6))
        action_mode_rtn = np.full(self._num_envs, ActionMode.UNCERTAIN)

        self._last_action_mode = self._action_mode

        # Obtain all energies
        energy_2d = energy_value_2d(plant_state[:, 2:], MATRIX_P)

        print(f"dwell_flag**********************************************: {dwell_flag}")
        print(f"energy_2d: {energy_2d}")

        for i, energy in enumerate(energy_2d):
            # Display current system status based on energy
            if energy < epsilon[i]:
                print(f"current system {i} energy status: {energy} < {epsilon[i]}, system is safe")
            else:
                print(f"current system {i} energy status: {energy} >= {epsilon[i]}, system is unsafe")

            # When Teacher disabled or deactivated
            print(f"np.any(ha_action[i]): {ha_action[i]}")
            if not np.any(ha_action[i]) and dwell_flag[i] is False:
                print("HA-Teacher is deactivated, use HP-Student's action instead")
                self._action_mode[i] = ActionMode.STUDENT
                self._plant_action[i] = hp_action[i]

                terminal_stance_ddq_rtn[i] = hp_action[i]
                action_mode_rtn[i] = ActionMode.STUDENT
                # return hp_action, ActionMode.STUDENT

            # Teacher activated
            if self._last_action_mode[i] == ActionMode.TEACHER:

                # Teacher Dwell time
                if dwell_flag[i]:
                    if ha_action[i] is None:
                        raise RuntimeError(f"Unrecognized HA-Teacher action {ha_action[i]} from {i} for dwelling")
                    else:
                        print("Continue HA-Teacher action in dwell time")
                        self._action_mode[i] = ActionMode.TEACHER
                        self._plant_action[i] = ha_action[i]

                        terminal_stance_ddq_rtn[i] = ha_action[i]
                        action_mode_rtn[i] = ActionMode.TEACHER
                        # return ha_action, ActionMode.TEACHER

                # Switch back to HPC
                else:
                    self._action_mode[i] = ActionMode.STUDENT
                    self._plant_action[i] = hp_action[i]
                    print(f"Max HA-Teacher dwell time achieved, switch back to HP-Student control")

                    terminal_stance_ddq_rtn[i] = hp_action[i]
                    action_mode_rtn[i] = ActionMode.STUDENT
                    # return hp_action, ActionMode.STUDENT

            elif self._last_action_mode[i] == ActionMode.STUDENT:

                # Inside safety envelope (bounded by epsilon)
                if energy < epsilon[i]:
                    self._action_mode[i] = ActionMode.STUDENT
                    self._plant_action[i] = hp_action[i]
                    print(f"Continue HP-Student action")

                    terminal_stance_ddq_rtn[i] = hp_action[i]
                    action_mode_rtn[i] = ActionMode.STUDENT
                    # return hp_action, ActionMode.STUDENT

                # Outside safety envelope (bounded by epsilon)
                else:
                    print(f"Switch to HA-Teacher action for safety concern")
                    self._action_mode[i] = ActionMode.TEACHER
                    self._plant_action[i] = ha_action[i]

                    terminal_stance_ddq_rtn[i] = ha_action[i]
                    action_mode_rtn[i] = ActionMode.TEACHER
                    # return ha_action, ActionMode.TEACHER
            else:
                raise RuntimeError(f"Unrecognized last action mode: {self._last_action_mode[i]} for {i}")

        return terminal_stance_ddq_rtn, action_mode_rtn

    @property
    def plant_action(self):
        return self._plant_action

    @property
    def action_mode(self):
        return self._action_mode

    @property
    def last_action_mode(self):
        return self._last_action_mode
