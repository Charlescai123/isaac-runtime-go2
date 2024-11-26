import sys
import copy
import enum
import time
import ctypes
import pickle
import traceback

import asyncio
import numpy as np
import cvxpy as cp
# import cvxpy as cp
from typing import Tuple, Any
import multiprocessing as mp

# import matlab
# import matlab.engine
from omegaconf import DictConfig
from numpy import linalg as LA
import matplotlib.pyplot as plt
from collections import deque
from numpy.linalg import pinv

from src.physical_design import MATRIX_P
from scipy.linalg import solve_continuous_are, inv
from src.utils.utils import energy_value
from cvxopt import matrix, solvers
from isaacgym.torch_utils import to_torch


np.set_printoptions(suppress=True)


class HATeacher:
    def __init__(self, teacher_cfg: DictConfig):

        # Teacher Configure
        self.chi = teacher_cfg.chi
        self.epsilon = teacher_cfg.epsilon
        self.cvxpy_solver = teacher_cfg.cvxpy_solver
        self.p_mat = MATRIX_P
        self.max_dwell_steps = teacher_cfg.tau

        self.teacher_enable = teacher_cfg.teacher_enable
        self.teacher_learn = teacher_cfg.teacher_learn

        # HAC Runtime
        # self._ref_state = None
        self._plant_state = None
        self._teacher_activate = False
        # self._error_state = None
        self._patch_center = np.zeros(12)
        self._center_update = True  # Patch center update flag
        self._dwell_step = 0  # Dwell step
        self.patch_interval = 5

        # Patch kp and kd
        # self._patch_kp = np.diag((0., 0., 100., 100., 100., 0.))
        # self._patch_kd = np.diag((40., 30., 10., 10., 10., 30.))
        # self.apply_realtime_patch = False
        self.apply_realtime_patch = True
        #
        #
        self._patch_kp = np.array([[-0., -0., -0., -0., -0., -0.],
                                   [-0., -0., -0., -0., -0., -0.],
                                   [-0., -0., 296., 0., - 0., 0.],
                                   [-0., -0., - 0., 200., 0, 0],
                                   [-0., -0., 0., 0, 200, 0],
                                   [-0., -0., 0., 0, -0., 194]])

        self._patch_kd = np.array([[31., 0., 0., -0., 0., 0.],
                                   [0., 13., -0., 0., -0, 0.],
                                   [0., -0., 28., 0., -0., 0.],
                                   [0., 0., -0., 26, 0., 0.],
                                   [-0., 0., 0., -0., 26., -0.],
                                   [0., 0., 0., 0., -0., 25.]])
        #
        # self._patch_kp = np.array([[-0., -0., -0., -0., -0., -0.],
        #                            [-0., -0., -0., -0., -0., -0.],
        #                            [-0., -0., 253., -0., -0., 0.],
        #                            [-0., -0., -0., 248., 0., 0.],
        #                            [-0., -0., -0., 0., 248., 0.],
        #                            [-0., -0., 0., 0., -0., 240.]])
        # self._patch_kd = np.array([[16., -0., 0., 0., - 0., - 0.],
        #                            [0., 16., 0., 0., 0., -0.],
        #                            [-0., 0., 32., -0., -0., 0.],
        #                            [-0., 0., 0., 31., 0., -0.],
        #                            [0., -0., -0., 0., 31., -0.],
        #                            [0., -0., 0., 0., 0., 31.]])

        # self._patch_kp = np.array([[-0., -0., -0., -0., -0., -0.],
        #                            [-0., -0., -0., -0., -0., -0.],
        #                            [-0., -0., 1249., -0., 0., -0.],
        #                            [-0., -0., 0., 1181., -0., 0.],
        #                            [-0., -0., 0., 0., 1181., -0.],
        #                            [-0., -0., -0., 0., -0., 1152.]]) * 0.2
        # self._patch_kd = np.array([[35., -0., 0., 0., 0., 0.],
        #                            [0., 35., 0., 0., -0., 0.],
        #                            [-0., -0., 71., -0., 0., -0.],
        #                            [-0., -0., -0., 69., -0., 0.],
        #                            [-0., 0., 0., 0., 69., -0.],
        #                            [-0., 0., 0., 0., 0., 68.]])

        # self._patch_kp = np.array([[-0., -0., -0., -0., -0., -0.],
        #                            [-0., -0., -0., -0., -0., -0.],
        #                            [-0., -0., 118., -0., 0., -0.],
        #                            [-0., -0., 0., 125., 3., 26.],
        #                            [-0., -0., 0., -22., 125., -2.],
        #                            [-0., -0., -0., 16., -14., 105.]])
        # self._patch_kd = np.array([[11., -0., 0., 0., 0., 0.],
        #                            [0., 11., 0., 0., -0., 0.],
        #                            [-0., -0., 22., -0., 0., -0.],
        #                            [-0., -0., -0., 22., -0., 4.],
        #                            [-0., 0., 0., 0., 22., -5.],
        #                            [-0., 0., 0., 0., 0., 21.]])

        self.action_counter = 0

        # Multiprocessing compute for patch
        manager = mp.Manager()
        self.lock = manager.Lock()
        self.triggered_roll = manager.Value('d', 0)
        self.triggered_pitch = manager.Value('d', 0)
        self.triggered_yaw = manager.Value('d', 0)
        state_update_flag = manager.Value('b', 0)
        self._f_kp = manager.list(copy.deepcopy(self._patch_kp.reshape(36)))
        self._f_kd = manager.list(copy.deepcopy(self._patch_kd.reshape(36)))
        self.patch_process = None

        if self.teacher_enable:
            # self.mp_start()
            pass
        # queue = manager.Queue()
        # queue = asyncio.Queue()

        # state_triggered = mp.RawArray(ctypes.c_double, np.array([0] * 12))
        # state_triggered = mp.RawArray(ctypes.c_double, np.array([0] * 12))

    def mp_start(self):
        data = {
            'roll': self.triggered_roll,
            'pitch': self.triggered_pitch,
            'yaw': self.triggered_yaw,
            'kp': self._patch_kp,
            'kd': self._patch_kd
        }
        print("creating process for patch computing")
        self.patch_process = mp.Process(
            target=self.patch_compute, args=(self.triggered_roll, self.triggered_pitch, self.triggered_yaw,
                                             self._f_kp, self._f_kd, self.lock)
            # target=self.patch_compute, args=(data)
        )

        self.patch_process.daemon = True
        print("starting patch process")
        self.patch_process.start()
        print(f"Pid of patch process: {self.patch_process.pid}")
        # mat_process.join()

    @staticmethod
    def patch_compute2():
        while True:
            kp, kd = HATeacher.system_patch(0, 0, 0)
            print(f"sub kp kd: {kp},\n{kd}")
            print("compute from patch_compute2")
            time.sleep(1)

    # def patch_compute(self, roll, pitch, yaw, _f_kp, _f_kd, lock):
    @staticmethod
    def patch_compute(roll, pitch, yaw, _f_kp, _f_kd, lock):
        try:
            print("Starting a subprocess for LMI computation...")
            path = "./robot/ha_teacher"
            # mat_engine = matlab.engine.start_matlab()
            # mat_engine.cd(path)
            # print("Matlab current working directory is ---->>>", mat_engine.pwd())
            while True:
                # _state_trig = await _queue.get()  # get updated trigger state
                print("LMI process computing...")
                # print(f"update_flag value: {update_flag.value}")
                # print(f"roll value: {roll.value}")
                # print(f"pitch value: {pitch.value}")
                # print(f"yaw value: {yaw.value}")
                # print(f"_f_kp value: {_f_kp}")
                # print(f"_f_kd value: {_f_kd}")

                roll_v, pitch_v, yaw_v = roll.value, pitch.value, yaw.value
                print(f"roll_v: {roll_v}")
                print(f"pitch_v: {pitch_v}")
                print(f"yaw_v: {yaw_v}")
                print("Obtained new state, updating the patch gain with cvxpy")
                F_kp, F_kd = HATeacher.system_patch(roll_v, pitch_v, yaw_v)
                # lock.acquire()
                for i in range(36):
                    _f_kp[i] = np.asarray(F_kp).reshape(36)[i]
                    _f_kd[i] = np.asarray(F_kd).reshape(36)[i]
                # _f_kp.reshape(6, 6)
                # _f_kd.reshape(6, 6)
                print(f"type of _f_kp: {type(_f_kp)}")
                # lock.release()
                print("Patch gain is updated now ---->>>")
                time.sleep(0.5)

                # _lock.release()
                # time.sleep(0.04)
        except:
            # traceback.print_exc(file=open("suberror.txt", "w+"))
            error = traceback.format_exc()
            print(f"subprocess error: {error}")
            sys.stdout.flush()
            return error

    def update(self, error_state: np.ndarray):
        """
        Update real-time plant and corresponding patch center if state is unsafe
        """
        print(f"error_state: {error_state}")
        print(f"self._patch_center: {self._patch_center}")

        self._plant_state = error_state
        # self._ref_state = ref_state
        # self._error_state = error_state
        energy = energy_value(state=error_state[2:], p_mat=MATRIX_P)

        # HA-Teacher activated
        if self._teacher_activate:
            if self._dwell_step >= self.max_dwell_steps:
                print(f"Reaching maximum dwell steps, deactivate HA-Teacher")
                self._teacher_activate = False

        # HA-Teacher deactivated
        else:
            # Outside envelope boundary
            if energy >= self.epsilon:
                self._dwell_step = 0
                self._teacher_activate = True  # Activate teacher
                self._patch_center = self._plant_state * self.chi  # Update patch center
                print(f"Activate HA-Teacher and updated patch center is: {self._patch_center}")

        return energy
        # # Restore patch flag
        # if energy < self.epsilon:
        #     self._center_update = True
        #
        # # States unsafe (outside safety envelope)
        # else:
        #     # Update patch center with current plant state
        #     if self._center_update is True:
        #         self._patch_center = self._plant_state * self.chi
        #         self._center_update = False

    def feedback_law(self, roll, pitch, yaw):
        # roll = matlab.double(roll)s
        # pitch = matlab.double(pitch)
        # yaw = matlab.double(yaw)
        # self._F_kp, self._F_kd = np.array(self.mat_engine.feedback_law2(roll, pitch, yaw, nargout=2))
        # return self._F_kp, self._F_kd
        return np.asarray(f_kp).reshape(6, 6), np.asarray(f_kd).reshape(6, 6)

    def get_action(self):
        """
        Get updated teacher action during real-time
        """
        self.action_counter += 1

        # If teacher deactivated
        if self.teacher_enable is False or self._teacher_activate is False:
            print(f"teacher is deactivated")
            return None, False

        self.triggered_roll.value = self._plant_state[3]
        self.triggered_pitch.value = self._plant_state[4]
        self.triggered_yaw.value = self._plant_state[5]

        # if self.action_counter % 200 == 0:
        #     self.triggered_roll.value = self._plant_state[3]
        #     self.triggered_pitch.value = self._plant_state[4]
        #     self.triggered_yaw.value = self._plant_state[5]
        #     # self._patch_kp = np.array(self._f_kp).reshape(6, 6)
        #     # self._patch_kd = np.array(self._f_kd).reshape(6, 6)
        #     for idx in range(36):
        #         i = idx // 6
        #         j = idx % 6
        #         if self._f_kp[idx] != 0:
        #             self._patch_kp[i][j] = self._f_kp[idx]
        #         if self._f_kd[idx] != 0:
        #             self._patch_kd[i][j] = self._f_kd[idx]

        def realtime_patch():
            s = time.time()
            roll, pitch, yaw = self._plant_state[3:6]
            # roll = roll.numpy()
            # pitch = pitch.numpy()
            # yaw = yaw.numpy()
            self._patch_kp, self._patch_kd = self.system_patch(roll=roll, pitch=pitch, yaw=yaw)
            e = time.time()
            print(f"roll pitch yaw: {roll}, {pitch}, {yaw}")
            print(f"self._patch_kp: {self._patch_kp}")
            print(f"self._patch_kd: {self._patch_kd}")
            print(f"patch time: {e - s}")


        # Turn on real-time patch
        if self.apply_realtime_patch and self.action_counter % self.patch_interval == 0:
            realtime_patch()

        # Do not turn on
        # time.sleep(0.02)

        s0 = time.time()
        # self.triggered_roll.value = self._plant_state[3]
        # self.triggered_pitch.value = self._plant_state[4]
        # self.triggered_yaw.value = self._plant_state[5]
        s1 = time.time()
        # print(f"self._f_kp: {self._f_kp}")
        # print(f"self._f_kd: {self._f_kd}")
        # print(f"self._patch_kp: {self._patch_kp}")
        # print(f"self._patch_kd: {self._patch_kd}")
        # print(f"self._patch_center: {self._patch_center}")
        # print(f"self._plant_state: {self._plant_state}")
        s2 = time.time()
        print(f"self._plant_state[:6] - self._patch_center[:6]: {self._plant_state[:6] - self._patch_center[:6]}")

        # print(
        #     f"self._patch_kp @ (self._plant_state[:6] - self._patch_center[:6]): {self._patch_kp @ (self._plant_state[:6] - self._patch_center[:6])}")
        # print(
        #     f"self._patch_kd @ (self._plant_state[6:] - self._patch_center[6:]): {self._patch_kd @ (self._plant_state[6:] - self._patch_center[6:])}")
        teacher_action = np.squeeze(self._patch_kp @ (self._plant_state[:6] - self._patch_center[:6]) * -1
                                    + self._patch_kd @ (self._plant_state[6:] - self._patch_center[6:]) * -1)
        print(f"teacher_action: {teacher_action}")

        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print(f"patch_kp: {self._patch_kp}")
        # print(f"patch_kd: {self._patch_kd}")
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        s3 = time.time()
        # print(f"teacher_action part 1 time: {s1 - s0}")
        # print(f"teacher_action part 2 time: {s2 - s1}")
        # print(f"teacher_action part 3 time: {s3 - s2}")
        # print(f"teacher_action total time: {s3 - s0}")
        assert self._dwell_step <= self.max_dwell_steps
        self._dwell_step += 1
        print(f"HA-Teacher runs for dwell time: {self._dwell_step}/{self.max_dwell_steps}")

        # self._phy_ddq = (kp @ (err_q - chi * state_trig[:6])
        #                  + kd @ (err_dq - chi * state_trig[6:])).squeeze()

        return teacher_action, True

    @staticmethod
    def system_patch_old(roll, pitch, yaw):
        """
        Computes the patch gain with roll pitch yaw.

        Args:
          roll: Roll angle (rad).
          pitch: Pitch angle (rad).
          yaw: Yaw angle (rad).

        Returns:
          F_kp: Proportional feedback gain matrix.
          F_kd: Derivative feedback gain matrix.
        """

        # Rotation matrices
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
        Rzyx = Rz.dot(Ry.dot(Rx))
        # print(f"Rzyx: {Rzyx}")

        Rzyx = np.array([[np.cos(yaw) / np.cos(pitch), np.sin(yaw) / np.cos(pitch), 0],
                         [-np.sin(yaw), np.cos(yaw), 0],
                         [np.cos(yaw) * np.tan(pitch), np.sin(yaw) * np.tan(pitch), 1]])

        # Sampling period
        T = 1 / 16  # 36 35 34

        # System matrices (continuous-time)
        aA = np.zeros((10, 10))
        aA[0, 6] = 1
        aA[1:4, 7:10] = Rzyx
        aB = np.zeros((10, 6))
        aB[4:, :] = np.eye(6)

        # System matrices (discrete-time)
        B = aB * T
        A = np.eye(10) + T * aA

        # bP = self.p_mat
        bP = np.array([[140.1190891, 0, 0, -0, -0, 0, 3.7742345, -0, 0, -0],
                       [0, 2.3e-06, 0, -0, -0, 0, 0, 1.4e-06, 0, 0],
                       [0, 0, 2.3e-06, 0, -0, 0, 0, 0, 1.4e-06, 0],
                       [-0, -0, 0, 467.9872184, 0, -0, -0, -0, 0, 152.9259161],
                       [-0, -0, -0, 0, 2.9088242, 0, -0, -0, -0, 0],
                       [0, 0, 0, -0, 0, 1.9e-06, -0, 0, 0, -0],
                       [3.7742345, 0, 0, -0, -0, -0, 0.3773971, 0, 0, -0],
                       [-0, 1.4e-06, 0, -0, -0, 0, 0, 1e-06, 0, -0],
                       [0, 0, 1.4e-06, 0, -0, 0, 0, 0, 1e-06, 0],
                       [-0, 0, 0, 152.9259161, 0, -0, -0, -0, 0, 155.2407021]]) * 1

        eta = 2  # 2
        # beta = 0.24
        beta = 0.2
        # beta = 0.3        # also works in simulation
        kappa = 0.001

        Q = cp.Variable((10, 10), PSD=True)
        R = cp.Variable((6, 10))

        w = 7

        constraints = [
            cp.bmat([
                [(beta - (1 + (1 / w)) * kappa * eta) * Q, Q @ A.T + R.T @ B.T],
                [A @ Q + B @ R, Q / (1 + w)]
            ]) >> 0,
            np.eye(10) - Q @ bP >> 0
        ]

        # Define problem and objective
        problem = cp.Problem(cp.Minimize(0), constraints)

        # Solve the problem
        # problem.solve()
        problem.solve(solver=cp.CVXOPT)

        # Extract optimal values
        # Check if the problem is solved successfully
        if problem.status == 'optimal':
            print("Optimization successful.")
        else:
            print("Optimization failed.")

        optimal_Q = Q.value
        optimal_R = R.value
        P = np.linalg.inv(optimal_Q)

        # Compute aF
        aF = np.round(aB @ optimal_R @ P, 0)

        Fb2 = aF[6:10, 0:4]

        # Compute F_kp and F_kd
        F_kp = -np.block([
            [np.zeros((2, 6))],
            [np.zeros((4, 2)), Fb2]])
        F_kd = -aF[4:10, 4:10]

        # Check if the problem is solved successfully
        if np.all(np.linalg.eigvals(P) > 0):
            print("LMIs feasible")
        else:
            print("LMIs infeasible")

        return F_kp, F_kd

    import tensorflow as tf
    # from numba import njit, jit
    @staticmethod
    # @tf.function
    def system_patch(roll, pitch, yaw):
        """
         Computes the patch gain with roll pitch yaw.

         Args:
           roll: Roll angle (rad).
           pitch: Pitch angle (rad).
           yaw: Yaw angle (rad).

         Returns:
           F_kp: Proportional feedback gain matrix.
           F_kd: Derivative feedback gain matrix.
         """

        # Rotation matrices
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
        Rzyx = Rz.dot(Ry.dot(Rx))
        # print(f"Rzyx: {Rzyx}")

        Rzyx = np.array([[np.cos(yaw) / np.cos(pitch), np.sin(yaw) / np.cos(pitch), 0],
                         [-np.sin(yaw), np.cos(yaw), 0],
                         [np.cos(yaw) * np.tan(pitch), np.sin(yaw) * np.tan(pitch), 1]])

        bP = np.array([[140.6434, 0, 0, 0, 0, 0, 5.3276, 0, 0, 0],
                       [0, 134.7596, 0, 0, 0, 0, 0, 6.6219, 0, 0],
                       [0, 0, 134.7596, 0, 0, 0, 0, 0, 6.622, 0],
                       [0, 0, 0, 49.641, 0, 0, 0, 0, 0, 6.8662],
                       [0, 0, 0, 0, 11.1111, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 3.3058, 0, 0, 0, 0],
                       [5.3276, 0, 0, 0, 0, 0, 3.6008, 0, 0, 0],
                       [0, 6.6219, 0, 0, 0, 0, 0, 3.6394, 0, 0],
                       [0, 0, 6.622, 0, 0, 0, 0, 0, 3.6394, 0],
                       [0, 0, 0, 6.8662, 0, 0, 0, 0, 0, 4.3232]])

        # Sampling period
        T = 1 / 30  # work in 25 to 30

        # System matrices (continuous-time)
        aA = np.zeros((10, 10))
        aA[0, 6] = 1
        aA[1:4, 7:10] = Rzyx
        aB = np.zeros((10, 6))
        aB[4:, :] = np.eye(6)

        # System matrices (discrete-time)
        B = aB * T
        A = np.eye(10) + T * aA

        alpha = 0.8
        kappa = 0.01
        chi = 0.2
        # gamma = 1
        # hd = 0.000
        gamma1 = 1
        gamma2 = 1  # 1

        b1 = 1 / 0.15  # height  0.15
        b2 = 1 / 0.35  # velocity 0.3
        # b3 = 1 / 0.1   # yaw 0.1
        # b4 = 1 / 0.5   # yaw rate 1

        D = np.array([[b1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, b2, 0, 0, 0, 0, 0]])
        # [0, 0, 0, b3, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, b4]])
        c1 = 1 / 45
        c2 = 1 / 70
        C = np.array([[c1, 0, 0, 0, 0, 0],
                      [0, c1, 0, 0, 0, 0],
                      [0, 0, c1, 0, 0, 0],
                      [0, 0, 0, c2, 0, 0],
                      [0, 0, 0, 0, c2, 0],
                      [0, 0, 0, 0, 0, c2]])

        Q = cp.Variable((10, 10), PSD=True)
        T = cp.Variable((6, 6), PSD=True)
        R = cp.Variable((6, 10))
        mu = cp.Variable((1, 1))

        constraints = [cp.bmat([[(alpha - kappa * (1 + (1 / gamma2))) * Q, Q @ A.T + R.T @ B.T],
                                [A @ Q + B @ R, Q / (1 + gamma2)]]) >> 0,
                       cp.bmat([[Q, R.T],
                                [R, T]]) >> 0,
                       (1 - chi * gamma1) * mu - (1 - (2 * chi) + (chi / gamma1)) >> 0,
                       Q - mu * np.linalg.inv(bP) >> 0,
                       np.identity(2) - D @ Q @ D.transpose() >> 0,
                       np.identity(6) - C @ T @ C.transpose() >> 0,
                       # T - hd * np.identity(6) >> 0,
                       mu - 1.0 >> 0,
                       ]

        # Define problem and objective
        problem = cp.Problem(cp.Minimize(0), constraints)

        # Solve the problem
        problem.solve(solver=cp.CVXOPT)

        # Extract optimal values
        # Check if the problem is solved successfully
        if problem.status == 'optimal':
            print("Optimization successful.")
        else:
            print("Optimization failed.")

        optimal_Q = Q.value
        optimal_R = R.value
        optimal_mu = mu.value

        # print(mu.value)
        # print(Q.value)

        P = np.linalg.inv(optimal_Q)

        # Compute aF
        aF = np.round(aB @ optimal_R @ P, 0)
        Fb2 = aF[6:10, 0:4]

        # Compute F_kp
        F_kp = -np.block([
            [np.zeros((2, 6))],
            [np.zeros((4, 2)), Fb2]])
        # Compute F_kd
        F_kd = -aF[4:10, 4:10]

        print(f"F_kp is: {F_kp}")
        print(f"F_kd is: {F_kd}")

        # Check if the problem is solved successfully
        if np.all(np.linalg.eigvals(P) > 0):
            print("LMIs feasible")
            print(F_kp)
            print(F_kd)

        else:
            print("LMIs infeasible")

        return F_kp, F_kd

    @property
    def ref_state(self):
        return self._ref_state

    @property
    def plant_state(self):
        return self._plant_state

    @property
    def error_state(self):
        return self._error_state

    @property
    def patch_center(self):
        return self._patch_center

    @property
    def patch_gain(self):
        return self._patch_kp, self._patch_kd

    @property
    def dwell_step(self):
        return self._dwell_step


if __name__ == '__main__':
    As = np.array([[0, 1, 0, 0],
                   [0, 0, -1.42281786576776, 0.182898194776782],
                   [0, 0, 0, 1],
                   [0, 0, 25.1798795199119, 0.385056459685276]])

    Bs = np.array([[0,
                    0.970107410065162,
                    0,
                    -2.04237185222105]])

    Ak = np.array([[1, 0.0100000000000000, 0, 0],
                   [0, 1, -0.0142281786576776, 0.00182898194776782],
                   [0, 0, 1, 0.0100000000000000],
                   [0, 0, 0.251798795199119, 0.996149435403147]])

    Bk = np.array([[0,
                    0.00970107410065163,
                    0,
                    -0.0204237185222105]])

    sd = np.array([[0.234343490000000,
                    0,
                    -0.226448960000000,
                    0]])
    roll, pitch, yaw = 0.043301478028297424, 0.023575300350785255, -0.0005105035379528999
    F_kp, F_kd = HATeacher.system_patch(roll, pitch, yaw)
    # ha_teacher = HATeacher()
    # K = ha_teacher.feedback_law(0, 0, 0)
    # print(K)

    # testN = 100
    # s = time.time()
    # for i in range(testN):
    #     F_kp, F_kd = HATeacher.system_patch(0., 0., 0.)
    # e = time.time()
    # duration = (e - s) / testN
    print(f"F_kp is: {F_kp}")
    print(f"F_kd is: {F_kd}")
    # print(f"time: {duration}")
