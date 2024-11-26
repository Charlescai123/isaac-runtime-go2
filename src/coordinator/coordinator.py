import os
import time
import enum
import copy
import logging
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import linalg as LA

from src.hp_student.agents.ddpg import DDPGAgent
# from src.logger.logger import Logger, plot_trajectory
from src.utils.utils import ActionMode, energy_value, logger
from src.physical_design import MATRIX_P

np.set_printoptions(suppress=True)


class Coordinator:

    def __init__(self):

        # Configurations
        # self.teacher_learn = config.teacher_learn
        # self.max_dwell_steps = config.max_dwell_steps

        # Real time status
        # self._dwell_step = 0
        # self._plant_state = None
        self._plant_action = np.zeros(6)
        self._action_mode = ActionMode.STUDENT
        self._last_action_mode = None

    # def update(self, state: np.ndarray):
    #     self._plant_state = state

    def get_terminal_action(self, hp_action, ha_action, plant_state, epsilon=1, dwell_flag=False):
        # print(f"last_action_mode: {self.last_action_mode}")
        # print(f"action_mode: {self._action_mode}")
        self._last_action_mode = self._action_mode

        # return ha_action, ActionMode.TEACHER

        # print(f"hp_action: {hp_action}")
        # print(f"ha_action: {ha_action}")

        # Display current system status based on energy
        energy = energy_value(plant_state[2:], MATRIX_P)
        if energy < epsilon:
            print(f"current system energy status: {energy} < {epsilon}, system is safe")
        else:
            print(f"current system energy status: {energy} >= {epsilon}, system is unsafe")

        # When Teacher disabled or deactivated
        if ha_action is None:
            print("HA-Teacher is deactivated, use HP-Student's action instead")
            self._action_mode = ActionMode.STUDENT
            self._plant_action = hp_action
            return hp_action, ActionMode.STUDENT

        # Teacher activated
        if self._last_action_mode == ActionMode.TEACHER:
            # Teacher Dwell time
            if dwell_flag is True:
                if ha_action is None:
                    raise RuntimeError(f"Unrecognized HA-Teacher action {ha_action} for dwelling")
                else:
                    print("Continue HA-Teacher action in dwell time")
                    self._action_mode = ActionMode.TEACHER
                    self._plant_action = ha_action
                    return ha_action, ActionMode.TEACHER

            # Switch back to HPC
            else:
                self._action_mode = ActionMode.STUDENT
                self._plant_action = hp_action
                print(f"Max HA-Teacher dwell time achieved, switch back to HP-Student control")
                return hp_action, ActionMode.STUDENT

        elif self._last_action_mode == ActionMode.STUDENT:

            # Inside safety envelope (bounded by epsilon)
            if energy < epsilon:
                self._action_mode = ActionMode.STUDENT
                self._plant_action = hp_action
                print(f"Continue HP-Student action")
                return hp_action, ActionMode.STUDENT

            # Outside safety envelope (bounded by epsilon)
            else:
                print(f"Switch to HA-Teacher action for safety concern")
                self._action_mode = ActionMode.TEACHER
                self._plant_action = ha_action
                return ha_action, ActionMode.TEACHER
        else:
            raise RuntimeError(f"Unrecognized last action mode: {self._last_action_mode}")

    @property
    def dwell_step(self):
        return self._dwell_step

    @property
    def plant_action(self):
        return self._plant_action

    @property
    def action_mode(self):
        return self._action_mode

    @property
    def last_action_mode(self):
        return self._last_action_mode

    @property
    def plant_state(self):
        return self._plant_state

    # @plant_state.setter
    # def plant_state(self, plant_state: np.ndarray):
    #     self._plant_state = plant_state
