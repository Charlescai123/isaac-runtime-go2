import sys
import copy
import time
import traceback

import ml_collections
import numpy as np
import cvxpy as cp
import torch
from isaacgym.torch_utils import to_torch

from omegaconf import DictConfig
from numpy.linalg import pinv

from src.physical_design import MATRIX_P
from src.utils.utils import energy_value, energy_value_2d

np.set_printoptions(suppress=True)


class HATeacher:
    """Monitors the safety-critical systems in all envs"""

    def __init__(self, num_envs, teacher_cfg: ml_collections.ConfigDict(), device):
        self._device = device
        self._num_envs = num_envs

        # Teacher Configure
        self.chi = torch.full((self._num_envs,), teacher_cfg.chi, dtype=torch.float32, device=device)
        self.epsilon = torch.full((self._num_envs,), teacher_cfg.epsilon, dtype=torch.float32, device=device)
        self.max_dwell_steps = torch.full((self._num_envs,), teacher_cfg.tau, dtype=torch.int64, device=device)
        self.teacher_enable = torch.full((self._num_envs,), teacher_cfg.enable, dtype=torch.bool, device=device)
        self.teacher_correct = torch.full((self._num_envs,), teacher_cfg.correct, dtype=torch.bool, device=device)

        self.cvxpy_solver = teacher_cfg.cvxpy_solver
        self.p_mat = to_torch(MATRIX_P, device=device)

        # HAC Runtime
        self._plant_state = [None] * self._num_envs
        self._teacher_activate = torch.full((self._num_envs,), False, dtype=torch.bool, device=device)
        self._patch_center = torch.zeros((self._num_envs, 12), dtype=torch.float32, device=device)
        self._center_update = torch.full((self._num_envs,), True, dtype=torch.bool, device=device)
        self._dwell_step = torch.zeros(self._num_envs, dtype=torch.float32, device=device)
        _patch_interval = 10
        self.patch_interval = torch.full((self._num_envs,), _patch_interval, dtype=torch.int64, device=device)
        _apply_rt_patch = True  #
        self.apply_realtime_patch = torch.full((self._num_envs,), _apply_rt_patch, dtype=torch.bool, device=device)

        # Patch kp and kd
        # self._default_kp = torch.diag(to_torch([0., 0., 50., 50., 50., 0.], device=device))
        # self._default_kd = torch.diag(to_torch([10., 10., 10., 10., 10., 10.], device=device))
        self._default_kp = to_torch([[-0., -0., -0., -0., -0., -0.],
                                     [-0., -0., -0., -0., -0., -0.],
                                     [-0., -0., 296., 0., - 0., 0.],
                                     [-0., -0., - 0., 200., 0, 0],
                                     [-0., -0., 0., 0, 200, 0],
                                     [-0., -0., 0., 0, -0., 194]], device=device)
        self._default_kd = to_torch([[31., 0., 0., -0., 0., 0.],
                                     [0., 13., -0., 0., -0, 0.],
                                     [0., -0., 28., 0., -0., 0.],
                                     [0., 0., -0., 26, 0., 0.],
                                     [-0., 0., 0., -0., 26., -0.],
                                     [0., 0., 0., 0., -0., 25.]], device=device)

        self._patch_kp = torch.stack([to_torch(self._default_kp, device=self._device)] * self._num_envs, dim=0)
        self._patch_kd = torch.stack([to_torch(self._default_kd, device=self._device)] * self._num_envs, dim=0)

        self.action_counter = torch.zeros(self._num_envs, dtype=torch.int, device=device)

    def update(self, error_state: torch.Tensor):
        """
        Update real-time plant and corresponding patch center if state is unsafe (error_state is 2d)
        """

        self._plant_state = error_state
        energy_2d = energy_value_2d(state=error_state[:, 2:], p_mat=to_torch(MATRIX_P, device=self._device))

        # Find objects that need to be deactivated
        to_deactivate = (self._dwell_step >= self.max_dwell_steps) & self._teacher_activate
        if torch.any(to_deactivate):
            indices = torch.argwhere(to_deactivate)
            for idx in indices:
                print(f"Reaching maximum dwell steps at index {int(idx)}, deactivate HA-Teacher")
            self._teacher_activate[to_deactivate] = False

        # Find objects that need to be activated
        to_activate = (energy_2d >= self.epsilon) & (~self._teacher_activate)
        if torch.any(to_activate):
            indices = torch.argwhere(to_activate)
            for idx in indices:
                self._dwell_step[idx] = 0
                self._teacher_activate[idx] = True  # Activate teacher
                self._patch_center[tuple(idx)] = self._plant_state[tuple(idx)] * self.chi[tuple(idx)]
                print(f"Activate HA-Teacher at {int(idx)} with new patch center: {self._patch_center[tuple(idx)]}")

        # print(f"self._plant_state: {self._plant_state}")
        # print(f"self._patch_center: {self._patch_center}")

        return energy_2d

    def get_action(self):
        """
        Get updated teacher action during real-time
        """
        self.action_counter += 1

        actions = torch.zeros((self._num_envs, 6), dtype=torch.float32, device=self._device)
        dwell_flags = torch.full((self._num_envs,), False, dtype=torch.bool, device=self._device)

        # If All HA-Teacher disabled
        if not torch.any(self.teacher_enable):
            print("All HA-teachers are disabled")
            return actions, dwell_flags

        # If All HA-Teacher deactivated
        if not torch.any(self._teacher_activate):
            print("All teachers are deactivated")
            return actions, dwell_flags

        # Find the object that needs to be patched
        to_patch = self.apply_realtime_patch & (self.action_counter % self.patch_interval == 0)
        # print(f"to_patch: {to_patch}")
        if torch.any(to_patch):
            indices = torch.argwhere(to_patch)
            for idx in indices:
                print(f"Applying realtime patch at index {int(idx)}")
                self.realtime_patch(int(idx))
        # Do not turn on
        # time.sleep(0.02)

        # Find objects with HA-Teacher enabled and activated
        ha_alive = self.teacher_enable & self._teacher_activate
        indices = torch.argwhere(ha_alive)
        if indices.size == 0:
            raise RuntimeError("ha_alive contains no True values, indices is empty. Check the code please")

        # Set dwell flag for them
        dwell_flags[indices] = True

        kp_mul_term = (self._plant_state[indices, :6] - self._patch_center[indices, :6])
        kd_mul_term = (self._plant_state[indices, 6:] - self._patch_center[indices, 6:])

        # res = self._patch_kp[indices] @ kp_mul_term.unsqueeze(-1) + self._patch_kd[indices] @ kd_mul_term.unsqueeze(-1)
        actions[indices.flatten()] = torch.squeeze(self._patch_kp[indices] @ kp_mul_term.unsqueeze(-1) +
                                                   self._patch_kd[indices] @ kd_mul_term.unsqueeze(-1))

        assert torch.all(self._dwell_step <= self.max_dwell_steps)

        for idx in indices:
            self._dwell_step[idx] += 1
            print(f"HA-Teacher {int(idx)} runs for dwell time: "
                  f"{int(self._dwell_step[idx])}/{int(self.max_dwell_steps[idx])}")

        return actions, dwell_flags

    def realtime_patch(self, idx):
        roll, pitch, yaw = self._plant_state[idx, 3:6]
        self._patch_kp[idx, :], self._patch_kd[idx, :] = self.system_patch(roll=roll.cpu(), pitch=pitch.cpu(),
                                                                           yaw=yaw.cpu(),
                                                                           device=self._device)
        e = time.time()
        # print(f"patch time: {e - s}")

    # from numba import njit, jit
    @staticmethod
    # @tf.function
    def system_patch(roll, pitch, yaw, device="cuda:0"):
        """
         Computes the patch gain with roll pitch yaw.

         Args:
           roll: Roll angle (rad).
           pitch: Pitch angle (rad).
           yaw: Yaw angle (rad).
           device: device to torch tensor

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

        # print(f"Solved F_kp is: {F_kp}")
        # print(f"Solved F_kd is: {F_kd}")

        # Check if the problem is solved successfully
        if np.all(np.linalg.eigvals(P) > 0):
            print("LMIs feasible")
        else:
            print("LMIs infeasible")

        return (to_torch(F_kp, device=device),
                to_torch(F_kd, device=device))

    def system_patch_2d(self, roll_1d, pitch_1d, yaw_1d):
        kp, kd = [], []
        for i in range(len(roll_1d)):
            r, p, y = roll_1d[i], pitch_1d[i], yaw_1d[i]
            _kp, _kd = self.system_patch(r, p, y)
            kp.append(_kp)
            kd.append(_kd)
        return np.asarray(kp), np.asarray(kd)

    @property
    def device(self):
        return self._device

    @property
    def plant_state(self):
        return self._plant_state

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
