import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from numpy import linalg as LA
from numpy.linalg import inv

saved_folder = "./generated"

# import pdb
# pdb.set_trace()
data_rl = np.loadtxt("saved/runtime-learning.csv", dtype=np.float32, delimiter=',', skiprows=1)
data_wrl = np.loadtxt("saved/without-runtime-learning.csv", dtype=np.float32, delimiter=',', skiprows=1)
data_teacher_cost = np.loadtxt("saved/teacher_only.csv", dtype=np.float32, delimiter=',', skiprows=1)
data_student_cost = np.loadtxt("saved/student.csv", dtype=np.float32, delimiter=',', skiprows=1)


def save_robot_trajectory_fig():
    """Generate figure of the robot trajectory"""
    Fn = 900
    Bn = 0

    timestamp = data_rl[:Fn, 0]

    rl_pz = data_rl[:Fn, 5]
    wrl_pz = data_wrl[:Fn, 5]

    rl_vx = data_rl[:Fn, 9]
    wrl_vx = data_wrl[:Fn, 9]

    ####################################   Plot CoM Height   ####################################
    fig = plt.figure(figsize=(13, 12))
    t = np.arange(Fn)
    plt.plot(timestamp, rl_pz, linewidth=4, color='limegreen', label="With Our Framework")
    plt.plot(timestamp, wrl_pz, linewidth=4, color='blue', label="Without Our Framework")
    # plt.plot(fprlz, linewidth=4, color='y', label="HP-Student: Phy-DRL")
    plt.plot(timestamp, 0.3 * np.ones(Fn), linewidth=4, color='black', linestyle='dashed',
             label=r'$\mathrm{Height} \ \mathrm{Command} $')
    plt.plot(timestamp, 0.45 * np.ones(Fn), linewidth=4, color='red', linestyle='dashed',
             label=r'$\mathrm{Safety} \ \mathrm{Bounds} $')
    plt.plot(timestamp, 0.15 * np.ones(Fn), linewidth=4, color='red', linestyle='dashed')
    plt.grid()
    plt.xlabel("Time (s)", fontsize=25)
    plt.ylabel("CoM Height (m)", fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    # plt.legend(ncol=1, bbox_to_anchor=(0.00, 0.61, 0.9, 0.1))
    plt.legend(loc="upper left", bbox_to_anchor=(0.43, 0.96), fontsize=25)
    # plt.legend(fontsize=23)
    # plt.tight_layout()
    fig.savefig(f'{saved_folder}/height.pdf', dpi=600)
    # plt.show()

    #######################################   Plot Vx   #######################################
    fig = plt.figure(figsize=(13, 12))
    t = np.arange(Fn)
    plt.plot(timestamp, rl_vx, linewidth=4, color='limegreen', label="With Our Framework")
    plt.plot(timestamp, wrl_vx, linewidth=4, color='blue', label="Without Our Framework")
    # plt.plot(fprlv, linewidth=4, color='y', label="HP-Student: Phy-DRL")
    plt.plot(timestamp, 0.4 * np.ones(Fn), linewidth=4, color='black', linestyle='dashed',
             label=r'$\mathrm{Velocity} \ \mathrm{Command} $')
    plt.plot(timestamp, 0. * np.ones(Fn), linewidth=4, color='red', linestyle='dashed',
             label=r'$\mathrm{Safety} \ \mathrm{Bounds} $')
    plt.plot(timestamp, 0.8 * np.ones(Fn), linewidth=4, color='red', linestyle='dashed')
    plt.grid()
    plt.xlabel("Time (s)", fontsize=25)
    plt.ylabel("CoM-x Velocity (m/s)", fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(ncol=1, loc="upper left", bbox_to_anchor=(0.43, 0.96), fontsize=25)
    # plt.rc('legend', fontsize=22)
    fig.savefig(f'{saved_folder}/velocity.pdf', dpi=600)


def save_energy_fig():
    # Fn = 900
    Bn = 0

    timestamp = data_teacher_cost[:, 0]
    timestamp2 = data_student_cost[:, 0]

    motor_power = data_teacher_cost[:, -2]
    energy_cost = data_teacher_cost[:, -1]

    motor_power2 = data_student_cost[:, -2]
    energy_cost2 = data_student_cost[:, -1]

    #######################################   Plot Vx   #######################################
    fig = plt.figure(figsize=(13, 12))
    # plt.plot(timestamp, motor_power, linewidth=4, color='limegreen', label="With Our Framework")
    plt.plot(timestamp, energy_cost, linewidth=4, color='red', label="Only HA-Teacher")
    plt.plot(timestamp, motor_power, linewidth=4, color='red', label="Only HA-Teacher")
    plt.plot(timestamp2, energy_cost2, linewidth=4, color='blue', label="Student")
    plt.plot(timestamp2, motor_power2, linewidth=4, color='blue', label="Student")
    # plt.plot(fprlv, linewidth=4, color='y', label="HP-Student: Phy-DRL")
    # plt.plot(timestamp, 0.4 * np.ones(Fn), linewidth=4, color='black', linestyle='dashed',
    #          label=r'$\mathrm{Velocity} \ \mathrm{Command} $')
    # plt.plot(timestamp, 0. * np.ones(Fn), linewidth=4, color='red', linestyle='dashed',
    #          label=r'$\mathrm{Safety} \ \mathrm{Bounds} $')
    # plt.plot(timestamp, 0.8 * np.ones(Fn), linewidth=4, color='red', linestyle='dashed')
    # plt.grid()
    plt.xlabel("Time (s)", fontsize=25)
    plt.ylabel("Energy Consumption (J)", fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(ncol=1, loc="upper left", bbox_to_anchor=(0.0, 0.96), fontsize=25)
    # plt.rc('legend', fontsize=22)
    fig.savefig(f'{saved_folder}/cost.pdf', dpi=600)


if __name__ == "__main__":
    save_robot_trajectory_fig()
    # save_energy_fig()
