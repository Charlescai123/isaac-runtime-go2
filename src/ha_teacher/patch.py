import sys
import copy
import enum
import time
import ctypes
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
# from src.envs.locomotion.robots.motors import MotorCommand
# from src.envs.locomotion.robots.motors import MotorControlMode
from scipy.linalg import solve_continuous_are, inv
from cvxopt import matrix, solvers

# global lock
# global state_triggered
# global state_update_flag

default_kp = np.diag((0., 0., 100., 100., 100., 0.))
default_kd = np.diag((40., 30., 10., 10., 10., 30.))

# default_kp = np.array([[0, 0, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0, 0],
#                        [0, 0, 128, 0, 0, 0],
#                        [0, 0, 0, 83, -25, -2],
#                        [0, 0, 0, -33, 80, 2],
#                        [0, 0, 0, 1, 0, 80]])
#
# default_kd = np.array([[39, 0, 0, 0, 0, 0],
#                        [0, 35, 0, 0, 0, 0],
#                        [0, 0, 35, 0, 0, 0],
#                        [0, 0, 0, 37, -1, -9],
#                        [0, 0, 0, -1, 37, 9],
#                        [0, 0, 0, 0, 0, 40]])

manager = mp.Manager()
lock = manager.Lock()
triggered_roll = manager.Value('d', 0)
triggered_pitch = manager.Value('d', 0)
triggered_yaw = manager.Value('d', 0)
state_update_flag = manager.Value('b', 0)
f_kp = manager.list(copy.deepcopy(default_kp.reshape(36)))
f_kd = manager.list(copy.deepcopy(default_kd.reshape(36)))


# queue = manager.Queue()
# queue = asyncio.Queue()


# state_triggered = mp.RawArray(ctypes.c_double, np.array([0] * 12))
# state_triggered = mp.RawArray(ctypes.c_double, np.array([0] * 12))

def mp_start():
    print("creating mat process")
    mat_process = mp.Process(target=update_feedback_gain2,
                             args=(state_update_flag, triggered_roll, triggered_pitch, triggered_yaw, f_kp, f_kd, lock))

    mat_process.daemon = True
    print("starting mat process")
    mat_process.start()
    print(f"Pid of mat process: {mat_process.pid}")
    # mat_process.join()


def mp_start2():
    print("starting mat process pool")
    lmi_run()
    # with mp.Pool(1) as pool:
    #     x = pool.apply_async(update_feedback_gain2,
    #                          args=(state_update_flag, triggered_roll, triggered_pitch, triggered_yaw,))
    #     # x = pool.starmap_async(update_feedback_gain,
    #     #                        [state_update_flag, triggered_roll, triggered_pitch, triggered_yaw])
    #     x.get()


def update_feedback_gain(args):
    # print(f"args input: {args_input}")
    # args = args_input.value
    update_flag, roll, pitch, yaw = args[0], args[1], args[2], args[3]
    try:
        print("Starting a subprocess for LMI computation...")
        path = "./robot/ha_teacher"
        # mat_engine = matlab.engine.start_matlab()
        # mat_engine.cd(path)
        # print("Matlab current working directory is ---->>>", mat_engine.pwd())
        while True:
            # _state_trig = await _queue.get()  # get updated trigger state
            print("LMI process computing...")
            print(f"update_flag value: {update_flag.value}")
            print(f"roll value: {roll.value}")
            print(f"pitch value: {pitch.value}")
            print(f"yaw value: {yaw.value}")

            if update_flag.value == 1:
                # lock.acquire()
                print("success!!!")
                # if _lock.acquire():
                # print("Acquiring lock")
                roll_v, pitch_v, yaw_v = roll.value, pitch.value, yaw.value
                print("Obtained new state, updating the feedback gain using matlab engine")
                F_kp, F_kd = feedback_law2(roll_v, pitch_v, yaw_v)
                # F_kp, F_kd = mat_engine.feedback_law2(roll, pitch, yaw, nargout=2)
                # _update_flag.value = 0
                print("Feedback gain is updated now ---->>>")

                # _lock.release()
            time.sleep(1)
    except:
        # traceback.print_exc(file=open("suberror.txt", "w+"))
        error = traceback.format_exc()
        print(f"subprocess error: {error}")
        sys.stdout.flush()
        return error


async def lmi_run():
    task = asyncio.create_task(update_feedback_gain2(state_update_flag, triggered_roll, triggered_pitch, triggered_yaw))
    await asyncio.sleep(2)
    asyncio.gather(task)


def update_feedback_gain2(update_flag, roll, pitch, yaw, _f_kp, _f_kd, lock):
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

            if update_flag.value == 1:
                print("success!!!")
                roll_v, pitch_v, yaw_v = roll.value, pitch.value, yaw.value
                print("Obtained new state, updating the feedback gain using matlab engine")
                F_kp, F_kd = feedback_law2(roll_v, pitch_v, yaw_v)
                # lock.acquire()
                for i in range(36):
                    _f_kp[i] = np.asarray(F_kp).reshape(36)[i]
                    _f_kd[i] = np.asarray(F_kd).reshape(36)[i]
                # print(f"np.asarray(F_kp): {np.asarray(F_kp)}")
                # lock.release()
                print("Feedback gain is updated now ---->>>")

                # _lock.release()
            time.sleep(0.04)
    except:
        # traceback.print_exc(file=open("suberror.txt", "w+"))
        error = traceback.format_exc()
        print(f"subprocess error: {error}")
        sys.stdout.flush()
        return error


def system_patch_origin(roll, pitch, yaw):
    """
    Computes the feedback gains for a 3-axis attitude control system.

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

    # Sampling period
    T = 1 / 35

    # Control output matrix
    C = np.array([1, 0, 0])

    # System matrices (continuous-time)
    aA = np.zeros((10, 10))
    aA[0, 4] = 1
    # A[:3, 3:6] = C
    # A[3:, 3:] = np.eye(6)
    aA[1:4, 7:10] = Rzyx
    aB = np.zeros((10, 6))
    aB[4:, :] = np.eye(6)

    # System matrices (discrete-time)
    B = np.vstack((np.zeros((4, 6)), np.eye(6))) * T
    A = np.eye(10) + T * aA

    bP = np.array([[122.164786064669, 0, 0, 0, 2.48716597374493, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 480.62107526958, 0, 0, 0, 0, 0, 155.295455907449],
                   [2.48716597374493, 0, 0, 0, 3.21760325418695, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 155.295455907449, 0, 0, 0, 0, 0, 156.306807893237]])

    eta = 2
    # beta = 0.25
    beta = 0.24
    # beta = 0.3        # also works in simulation
    kappa = 0.001

    Q = cp.Variable((10, 10), PSD=True)
    R = cp.Variable((6, 10))

    # Define constraints
    # constraints = [
    #     (beta - 2 * eta * kappa) * np.eye(10) << Q,
    #     A @ Q + B @ R >> 0,
    #     Q >> 0.95 * np.eye(10),
    #     cp.trace(bP @ Q) >= 1
    # ]

    constraints = [
        cp.bmat([
            [(beta - 2 * kappa * eta) * Q, A.T + R.T @ B.T],
            [A @ Q + B @ R, Q / 2]
        ]) >> 0,
        # (beta - 2 * eta * kappa) * np.eye(10) << Q,
        # A @ Q + B @ R >> 0,
        Q >> 0.95 * np.eye(10),
        cp.trace(bP @ Q) >= 1
    ]

    # Define problem and objective
    problem = cp.Problem(cp.Minimize(0), constraints)

    # Solve the problem
    problem.solve()

    # Check if the problem is solved successfully
    if problem.status == 'optimal':
        print("Optimization successful.")
    else:
        print("Optimization failed.")

    # Extract optimal values
    optimal_Q = Q.value
    optimal_R = R.value

    print(f"Q: {optimal_Q}")
    print(f"R: {optimal_R}")
    # Compute P
    P = np.linalg.inv(optimal_Q)

    # Compute aF
    aF = np.round(aB @ optimal_R @ P, 0)
    print(f"aF: {aF}")
    # Extract submatrices Fb1 and Fb2
    Fb1 = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, aF[4, 0]]
    ])
    Fb2 = aF[7:10, 1:4]
    print(f"Fb1: {Fb1}")
    print(f"Fb2: {Fb2}")

    # Compute F_kp
    F_kp = -np.block([
        [Fb1, np.zeros((3, 3))],
        [np.zeros((3, 3)), Fb2]
    ])

    # Compute F_kd
    F_kd = -aF[4:10, 4:10]

    return F_kp, F_kd


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
    import pickle
    data = {
        'roll': .2,
        'pitch': .3,
        'yaw': .4,
        'Ak': Ak,
        'As': As
    }
    s1 = time.time()
    serialized_data = pickle.dumps(data)
    s2 = time.time()
    deserialized_data = pickle.loads(serialized_data)
    patch_As, patch_Ak = deserialized_data['As'], deserialized_data['Ak']
    s3 = time.time()
    print(f"serialized_data: {serialized_data}")
    print(f"deserialized_data: {deserialized_data}")
    print(f"serialization time: {s2 - s1}")
    print(f"deserialization time: {s3 - s2}")
    # ha_teacher = HATeacher()
    # K = ha_teacher.feedback_law(0, 0, 0)
    # print(K)
    # F_kp, F_kd = feedback_law2(0.05, 0.05, 0.05)
    # print(f"F_kp is: {F_kp}")
    # print(f"F_kd is: {F_kd}")
