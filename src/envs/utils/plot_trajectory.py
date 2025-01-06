import os
import sys
# import json
import copy
import pickle
import numpy as np

# from src.ha_teacher import ha_teacher
from src.utils.utils import ActionMode
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection


def plot_robot_trajectory(filepath: str) -> None:
    with open(filepath, 'rb') as f:
        phases = pickle.load(f)
    print(f"phases: {len(phases)}")
    print(f"phases: {phases[0]}")
    # time.sleep(123)
    zero_ref = []
    step = []
    desired_linear_vel = []
    desired_com_height = []
    desired_vx = []
    linear_vel = []
    rpy = []
    position = []
    timestamp = []
    angular_vel = []
    ground_reaction_forces = []
    phy_ddq = []
    drl_ddq = []
    total_ddq = []
    max_ddq = []
    min_ddq = []
    energy = []
    action_mode = []
    hp_action = []
    ha_action = []

    tracking_err = dict({
        'p': [],
        'rpy': [],
        'v': [],
        'rpy_dot': []
    })

    n = len(phases)
    # print(f"phases: {phases}")

    for i in range(len(phases)):
        #     tracking_error = list(phases[i]['tracking_error'])
        #     if np.isnan(tracking_error).any():
        #         tracking_err['p'].append([0, 0, 0])
        #         tracking_err['rpy'].append([0, 0, 0])
        #         tracking_err['v'].append([0, 0, 0])
        #         tracking_err['rpy_dot'].append([0, 0, 0])
        #     else:
        #         tracking_err['p'].append(tracking_error[0:3])
        #         tracking_err['rpy'].append(tracking_error[3:6])
        #         tracking_err['v'].append(tracking_error[6:9])
        #         tracking_err['rpy_dot'].append(tracking_error[9:12])
        #
        zero_ref.append(0)
        step.append(i)

        # phy_ddq.append(phases[i]['stance_ddq'][0])
        # drl_ddq.append(phases[i]['stance_ddq'][1])
        # total_ddq.append(phases[i]['stance_ddq'][2])
        # min_ddq.append(phases[i]['stance_ddq_limit'][0])
        # max_ddq.append(phases[i]['stance_ddq_limit'][1])

        position.append(phases[i]['base_position'].tolist())
        desired_com_height.append(phases[i]['desired_com_height'])
        timestamp.append(phases[i]['timestamp'].tolist())
        desired_vx.append(phases[i]['desired_vx'])
        # desired_linear_vel.append(phases[i]['desired_speed'][0])
        linear_vel.append(phases[i]['base_velocity'].tolist())
        angular_vel.append(phases[i]['base_angular_velocity'].tolist())
        rpy.append(phases[i]['base_orientation_rpy'].tolist())

        ground_reaction_forces.append(phases[i]['foot_contact_force'].tolist())
        # action_mode.append(phases[i]['action_mode'])
        # print(content[i]['desired_linear_vel'])
        # print(type(content[i]['desired_linear_vel']))
        phy_ddq.append(phases[i]['desired_acc_body_frame'].tolist())
        min_ddq.append(phases[i]['acc_min'].tolist())
        max_ddq.append(phases[i]['acc_max'].tolist())

        energy.append(phases[i]['energy'].tolist())
        action_mode.append(phases[i]['action_mode'])
        print(f"phases[i]['ha_action']: {phases[i]['ha_action']}")
        hp_action.append(phases[i]['hp_action'].cpu().numpy().squeeze())

        if phases[i]['ha_action'] is None:
            ha_action.append(([0., 0., 0., 0., 0., 0.]))
        else:
            ha_action.append(phases[i]['ha_action'].tolist())
        # dict(timestamp=self._robot.time_since_reset,
        #      base_position=torch.clone(self._robot.base_position),
        #      base_orientation_rpy=torch.clone(
        #          self._robot.base_orientation_rpy),
        #      base_velocity=torch.clone(self._robot.base_velocity_body_frame),
        #      base_angular_velocity=torch.clone(
        #          self._robot.base_angular_velocity_body_frame),
        #      motor_positions=torch.clone(self._robot.motor_positions),
        #      motor_velocities=torch.clone(self._robot.motor_velocities),
        #      motor_action=motor_action,
        #      motor_torques=self._robot.motor_torques,
        #      num_clips=self._num_clips,
        #      foot_contact_state=self._gait_generator.desired_contact_state,
        #      foot_contact_force=self._robot.foot_contact_forces,
        #      desired_swing_foot_position=desired_foot_positions,
        #      desired_acc_body_frame=self._desired_acc,
        #      solved_acc_body_frame=self._solved_acc,
        #      foot_positions_in_base_frame=self._robot.
        #      foot_positions_in_base_frame,
        #      env_action=action,
        #      env_obs=torch.clone(self._obs_buf)))

    step = np.array(step)
    drl_ddq = np.array(drl_ddq)
    phy_ddq = np.array(phy_ddq)
    total_ddq = np.array(total_ddq)
    max_ddq = np.array(max_ddq)
    min_ddq = np.array(min_ddq)
    energy = np.array(energy)
    ha_action = np.array(ha_action)
    hp_action = np.array(hp_action)
    hp_action = np.expand_dims(hp_action, axis=-1)
    print(f"hp_action: {hp_action}")
    print(f"ha_action: {ha_action}")
    ha_action = np.where(ha_action is None, 0., ha_action)
    # ha_action = ha_action.squeeze()
    # ha_action = np.expand_dims(ha_action, axis=-1)
    action_mode = np.array(action_mode)

    actions_in_ha = ha_action[action_mode == ActionMode.TEACHER]
    actions_in_hp = ha_action[action_mode == ActionMode.STUDENT]
    steps_in_ha = step[action_mode == ActionMode.TEACHER]
    steps_in_hp = step[action_mode == ActionMode.STUDENT]
    # print(f"action_mode: {action_mode}")
    # print(f"action_in_ha: {len(actions_in_ha)}")
    # print(f"action_in_hp: {len(actions_in_hp)}")
    # print(f"as: {[action_mode == ActionMode.TEACHER]}")

    tracking_err['p'] = np.asarray(tracking_err['p'])
    tracking_err['rpy'] = np.asarray(tracking_err['rpy'])
    tracking_err['v'] = np.asarray(tracking_err['v'])
    tracking_err['rpy_dot'] = np.asarray(tracking_err['rpy_dot'])
    timestamp = np.asarray(timestamp)
    position = np.asarray(position).squeeze()
    zero_ref = np.asarray(zero_ref)
    linear_vel = np.asarray(linear_vel).squeeze()

    desired_linear_vel = np.asarray(desired_linear_vel)
    desired_com_height = np.asarray(desired_com_height)
    desired_vx = np.asarray(desired_vx)

    angular_vel = np.asarray(angular_vel).squeeze()
    rpy = np.asarray(rpy).squeeze()

    position_ref = copy.deepcopy(position)
    position_ref[:, 0] = 0
    position_ref[:, 1] = 0
    position_ref[:, 2] = desired_com_height

    rpy_ref = copy.deepcopy(rpy)
    rpy_ref[:, 0] = 0
    rpy_ref[:, 1] = 0
    rpy_ref[:, 2] = 0

    linear_vel_ref = copy.deepcopy(linear_vel)
    linear_vel_ref[:, 0] = desired_vx
    linear_vel_ref[:, 1] = 0
    linear_vel_ref[:, 2] = 0.0

    angular_vel_ref = copy.deepcopy(angular_vel)
    angular_vel_ref[:, 0] = 0
    angular_vel_ref[:, 1] = 0
    angular_vel_ref[:, 2] = 0

    # print(f"action_mode: {action_mode}")

    step = timestamp

    # Simplex input
    simplex_step = []
    non_simplex_step = []
    simplex_input = dict({
        'p': [],
        'rpy': [],
        'v': [],
        'rpy_dot': []
    })

    linear_vel_simplex = []
    rpy_simplex = []
    angular_vel_simplex = []
    position_simplex = []

    cnt = 0
    # for i in range(n):
    #     if action_mode[i] != ha_teacher.HACActionMode.SIMPLEX:
    #         linear_vel_simplex.append(linear_vel[i])
    #         rpy_simplex.append(rpy[i])
    #         angular_vel_simplex.append(angular_vel[i])
    #         position_simplex.append(position[i])
    #         non_simplex_step.append(step[i])
    #         cnt += 1
    #     else:
    #         simplex_input['p'].append(linear_vel[i])
    #         simplex_input['rpy'].append(rpy[i])
    #         simplex_input['v'].append(angular_vel[i])
    #         simplex_input['rpy_dot'].append(position[i])
    #         simplex_step.append(step[i])
    #
    # simplex_input['p'] = np.asarray(simplex_input['p'])
    # simplex_input['rpy'] = np.asarray(simplex_input['rpy'])
    # simplex_input['v'] = np.asarray(simplex_input['v'])
    # simplex_input['rpy_dot'] = np.asarray(simplex_input['rpy_dot'])
    #
    # linear_vel_simplex = np.asarray(linear_vel_simplex)
    # rpy_simplex = np.asarray(rpy_simplex)
    # angular_vel_simplex = np.asarray(angular_vel_simplex)
    # position_simplex = np.asarray(position_simplex)
    #
    # print(f"Simplex point: {n - cnt}")
    # print(f"Non-Simplex point: {cnt}")
    # # print(f"Tracking error: {tracking_err_plot['p']}")
    # # print(tracking_err['p'] == simplex_err['p'])
    # print(len(non_simplex_step))
    # print(len(simplex_step))
    #
    # print(f"step: {step}")
    # print(f"steps: {steps}")
    # print(desired_linear_vel.shape)
    # print(linear_vel.shape)
    # print(desired_linear_vel[:, 0])
    # print(tracking_err)
    # print("............")
    # print(tracking_err['v'][:, 0])
    fig, axes = plt.subplots(4, 3)
    # for i, step in enumerate(steps):
    #     print(i)

    colors = {ActionMode.STUDENT: 'blue', ActionMode.TEACHER: 'red'}

    # # 创建图形和轴
    # fig, ax = plt.subplots(figsize=(10, 6))
    #
    # # 将 x 和 y 数据转换为用于 LineCollection 的坐标段
    # points = np.array([x, y]).T.reshape(-1, 1, 2)
    # segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 根据 action_mode 创建一个颜色列表
    seg_colors = [colors[mode] for mode in action_mode[:-1]]
    # print(f"seg_colors: {seg_colors}")

    # Vx
    # axes[0, 0].plot(step, tracking_err['v'][:, 0], zorder=4, label='error_vx')
    # axes[0, 0].plot(simplex_step, simplex_input['v'][:, 0], zorder=3, color='red', label='simplex_vx')
    axes[0, 0].plot(step, linear_vel_ref[:, 0], zorder=2, label='desire_vx')
    print(f"linear_vel: {linear_vel[:, 0]}")
    print(f"type: {type(linear_vel)}")
    print(f"step: {step}")
    axes[0, 0].plot(step, linear_vel[:, 0], zorder=1, label='vx')
    axes[0, 0].set_xlabel('Time/s', fontsize=14)
    axes[0, 0].set_ylabel('Vx', fontsize=14)
    axes[0, 0].legend(fontsize=12)

    # Vy
    # axes[0, 1].plot(step, simplex_err['v'][:, 1], zorder=4, label='simplex_vx')
    # axes[0, 1].plot(step, tracking_err['v'][:, 1], zorder=3, label='error_vy')
    axes[0, 1].plot(step, linear_vel_ref[:, 1], zorder=2, label='desire_vy')
    axes[0, 1].plot(step, linear_vel[:, 1], zorder=1, label='vy')
    axes[0, 1].set_xlabel('Time/s', fontsize=14)
    axes[0, 1].set_ylabel('Vy', fontsize=14)
    axes[0, 1].legend(fontsize=12)

    # Vz
    # axes[0, 2].plot(step, tracking_err['v'][:, 2], zorder=3, label='error_vz')
    axes[0, 2].plot(step, linear_vel_ref[:, 2], zorder=2, label='desire_vz')
    axes[0, 2].plot(step, linear_vel[:, 2], zorder=1, label='vz')
    axes[0, 2].set_xlabel('Time/s', fontsize=14)
    axes[0, 2].set_ylabel('Vz', fontsize=14)
    axes[0, 2].legend(fontsize=12)

    # Roll
    # axes[1, 0].plot(step, tracking_err['rpy'][:, 0], zorder=3, label='error_roll')
    # axes[1, 0].plot(step, zero_ref, zorder=2, label='desire_roll')
    axes[1, 0].plot(step, rpy_ref[:, 0], zorder=2, label='desired_roll')
    axes[1, 0].plot(step, rpy[:, 0], zorder=1, label='roll')
    axes[1, 0].set_xlabel('Time', fontsize=14)
    axes[1, 0].set_ylabel('Roll', fontsize=14)
    axes[1, 0].legend(fontsize=12)

    # Pitch
    # axes[1, 1].plot(step, tracking_err['rpy'][:, 1], zorder=3, label='error_pitch')
    axes[1, 1].plot(step, rpy_ref[:, 1], zorder=2, label='desire_pitch')
    axes[1, 1].plot(step, rpy[:, 1], zorder=1, label='pitch')
    axes[1, 1].set_xlabel('Time', fontsize=14)
    axes[1, 1].set_ylabel('Pitch', fontsize=14)
    axes[1, 1].legend(fontsize=12)

    # Yaw
    # axes[1, 2].plot(step, tracking_err['rpy'][:, 2], zorder=3, label='error_yaw')
    axes[1, 2].plot(step, rpy_ref[:, 2], zorder=2, label='desire_yaw')
    axes[1, 2].plot(step, rpy[:, 2], zorder=1, label='yaw')
    axes[1, 2].set_xlabel('Time/s', fontsize=14)
    axes[1, 2].set_ylabel('Yaw', fontsize=14)
    axes[1, 2].legend(fontsize=12)

    # Wx
    # axes[2, 0].plot(step, tracking_err['rpy_dot'][:, 0], zorder=3, label='error_wx')
    axes[2, 0].plot(step, angular_vel_ref[:, 0], zorder=2, label='desire_wx')
    axes[2, 0].plot(step, angular_vel[:, 0], zorder=1, label='wx')
    axes[2, 0].set_xlabel('Time/s', fontsize=14)
    axes[2, 0].set_ylabel('Wx', fontsize=14)
    axes[2, 0].legend(fontsize=12)

    # Wy
    # axes[2, 1].plot(step, tracking_err['rpy_dot'][:, 1], zorder=3, label='error_wy')
    axes[2, 1].plot(step, angular_vel_ref[:, 1], zorder=2, label='desire_wy')
    axes[2, 1].plot(step, angular_vel[:, 1], zorder=1, label='wy')
    axes[2, 1].set_xlabel('Time/s', fontsize=14)
    axes[2, 1].set_ylabel('Wy', fontsize=14)
    axes[2, 1].legend(fontsize=12)

    # Wz
    # axes[2, 2].plot(step, tracking_err['rpy_dot'][:, 2], zorder=3, label='error_wz')
    axes[2, 2].plot(step, angular_vel_ref[:, 2], zorder=2, label='desire_wz')
    axes[2, 2].plot(step, angular_vel[:, 2], zorder=1, label='wz')
    axes[2, 2].set_xlabel('Time/s', fontsize=14)
    axes[2, 2].set_ylabel('Wz', fontsize=14)
    axes[2, 2].legend(fontsize=12)

    # Px
    # axes[3, 0].plot(step, tracking_err['p'][:, 0], zorder=3, label='error_px')
    axes[3, 0].plot(step, zero_ref, zorder=2, label='desire_px')
    axes[3, 0].plot(step, zero_ref, zorder=1, label='px')
    axes[3, 0].set_xlabel('Time/s', fontsize=14)
    axes[3, 0].set_ylabel('Px', fontsize=14)
    axes[3, 0].legend(fontsize=12)

    # Py
    # axes[3, 1].plot(step, tracking_err['p'][:, 1], zorder=3, label='error_py')
    axes[3, 1].plot(step, zero_ref, zorder=2, label='desire_py')
    axes[3, 1].plot(step, zero_ref, zorder=1, label='py')
    axes[3, 1].set_xlabel('Time/s', fontsize=14)
    axes[3, 1].set_ylabel('Py', fontsize=14)
    axes[3, 1].legend(fontsize=12)

    # Pz
    # axes[3, 2].plot(step, tracking_err['p'][:, 2], zorder=3, label='error_pz')
    axes[3, 2].plot(step, position_ref[:, 2], zorder=2, label='desire_pz')
    axes[3, 2].plot(step, position[:, 2], zorder=1, label='pz')
    axes[3, 2].set_xlabel('Time/s', fontsize=14)
    axes[3, 2].set_ylabel('Pz', fontsize=14)
    axes[3, 2].legend(fontsize=12)

    # fig2, axes2 = plt.subplots(2, 3)

    fig2 = plt.figure()
    gs = gridspec.GridSpec(3, 3, figure=fig)
    label_font2 = 14
    legend_font2 = 14

    phy_ddq = phy_ddq.squeeze(axis=1)
    print(f"min_ddq: {min_ddq}")
    print(f"min_ddq: {min_ddq.shape}")
    # min_ddq = min_ddq.squeeze(axis=1)
    # max_ddq = max_ddq.squeeze(axis=1)
    # print(f"phy_ddq: {phy_ddq}")
    # print(f"phy_ddq: {phy_ddq.shape}")

    axes2 = fig2.add_subplot(gs[0, 0])
    axes2.plot(step, phy_ddq[:, 0], zorder=4, label='phy_vx')
    axes2.plot(step, hp_action[:, 0], zorder=3, label='hp_vx')
    axes2.plot(step, ha_action[:, 0], zorder=2, label='ha_vx')
    # axes2.plot(step, actions_in_ha[:, 0], zorder=2, label='ha_vx', color='blue')
    # axes2.plot(step, actions_in_hp[:, 0], zorder=2, label='ha_vx', color='red')
    # axes2[0, 0].plot(step, total_ddq[:, 0], zorder=2, label='total_vx')
    # axes2.plot(steps_in_ha, actions_in_ha[:, 0], color='red')
    # axes2.plot(steps_in_hp, actions_in_hp[:, 0], color='blue')
    # lc = LineCollection(segments, colors=seg_colors, linewidths=2)
    # axes2.add_collection(lc)
    axes2.plot(step, min_ddq[:, 0], zorder=1, label='ddq_min')
    axes2.plot(step, max_ddq[:, 0], zorder=1, label='ddq_max')
    axes2.set_xlabel('Time/s', fontsize=label_font2)
    axes2.set_ylabel('ddq vx', fontsize=label_font2)
    axes2.legend(fontsize=legend_font2)
    #
    axes2 = fig2.add_subplot(gs[0, 1])
    axes2.plot(step, phy_ddq[:, 1], zorder=4, label='phy_vy')
    axes2.plot(step, hp_action[:, 1], zorder=3, label='hp_vy')
    axes2.plot(step, ha_action[:, 1], zorder=2, label='ha_vy')
    # axes2[0, 1].plot(step, drl_ddq[:, 1], zorder=3, label='drl_vy')
    # axes2[0, 1].plot(step, total_ddq[:, 1], zorder=2, label='total_vy')
    axes2.plot(step, min_ddq[:, 1], zorder=1, label='ddq_min')
    axes2.plot(step, max_ddq[:, 1], zorder=1, label='ddq_max')
    axes2.set_xlabel('Time/s', fontsize=label_font2)
    axes2.set_ylabel('ddq wy', fontsize=label_font2)
    axes2.legend(fontsize=legend_font2)
    #
    axes2 = fig2.add_subplot(gs[0, 2])
    axes2.plot(step, phy_ddq[:, 2], zorder=3, label='phy_vz')
    axes2.plot(step, hp_action[:, 2], zorder=3, label='hp_vz')
    axes2.plot(step, ha_action[:, 2], zorder=2, label='ha_vz')
    # axes2[0, 2].plot(step, drl_ddq[:, 2], zorder=2, label='drl_vz')
    # axes2[0, 2].plot(step, total_ddq[:, 2], zorder=1, label='total_vz')
    axes2.plot(step, min_ddq[:, 2], zorder=1, label='ddq_min')
    axes2.plot(step, max_ddq[:, 2], zorder=1, label='ddq_max')
    axes2.set_xlabel('Time/s', fontsize=label_font2)
    axes2.set_ylabel('ddq vz', fontsize=label_font2)
    axes2.legend(fontsize=legend_font2)
    #
    axes2 = fig2.add_subplot(gs[1, 0])
    axes2.plot(step, phy_ddq[:, 3], zorder=3, label='phy_wx')
    axes2.plot(step, hp_action[:, 3], zorder=3, label='hp_wx')
    axes2.plot(step, ha_action[:, 3], zorder=2, label='ha_wx')
    # axes2[1, 0].plot(step, drl_ddq[:, 3], zorder=2, label='drl_wx')
    # axes2[1, 0].plot(step, total_ddq[:, 3], zorder=1, label='total_wx')
    axes2.plot(step, min_ddq[:, 3], zorder=1, label='ddq_min')
    axes2.plot(step, max_ddq[:, 3], zorder=1, label='ddq_max')
    axes2.set_xlabel('Time/s', fontsize=label_font2)
    axes2.set_ylabel('ddq wx', fontsize=label_font2)
    axes2.legend(fontsize=legend_font2)
    #
    axes2 = fig2.add_subplot(gs[1, 1])
    axes2.plot(step, phy_ddq[:, 4], zorder=3, label='phy_wy')
    axes2.plot(step, hp_action[:, 4], zorder=3, label='hp_wy')
    axes2.plot(step, ha_action[:, 4], zorder=2, label='ha_wy')
    # axes2[1, 1].plot(step, drl_ddq[:, 4], zorder=2, label='drl_wy')
    # axes2[1, 1].plot(step, total_ddq[:, 4], zorder=1, label='total_wy')
    axes2.plot(step, min_ddq[:, 4], zorder=1, label='ddq_min')
    axes2.plot(step, max_ddq[:, 4], zorder=1, label='ddq_max')
    axes2.set_xlabel('Time/s', fontsize=label_font2)
    axes2.set_ylabel('ddq wy', fontsize=label_font2)
    axes2.legend(fontsize=legend_font2)
    #
    axes2 = fig2.add_subplot(gs[1, 2])
    axes2.plot(step, phy_ddq[:, 5], zorder=3, label='phy_wz')
    axes2.plot(step, hp_action[:, 5], zorder=3, label='hp_wz')
    axes2.plot(step, ha_action[:, 5], zorder=2, label='ha_wz')
    # axes2[1, 2].plot(step, drl_ddq[:, 5], zorder=2, label='drl_wz')
    # axes2[1, 2].plot(step, total_ddq[:, 5], zorder=1, label='total_wz')
    axes2.plot(step, min_ddq[:, 5], zorder=1, label='ddq_min')
    axes2.plot(step, max_ddq[:, 5], zorder=1, label='ddq_max')
    axes2.set_xlabel('Time/s', fontsize=label_font2)
    axes2.set_ylabel('ddq wz', fontsize=label_font2)
    axes2.legend(fontsize=legend_font2)

    axes2 = fig2.add_subplot(gs[2, :])
    energy = np.expand_dims(energy, axis=-1)

    points = np.array([step, energy[:]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, colors=seg_colors)
    axes2.add_collection(lc)
    for mode, color in colors.items():
        axes2.plot([], [], color=color, label=f'{mode}')
    axes2.autoscale()

    # axes2.plot(step, energy[:], zorder=1, label='energy')
    axes2.set_xlabel('Time/s', fontsize=label_font2)
    axes2.set_ylabel('energy', fontsize=label_font2)
    axes2.legend(fontsize=legend_font2)

    # fig3, axes3 = plt.subplots(1, 1)
    # energy = np.expand_dims(energy, axis=-1)
    # print(f"energy is: {energy}")
    # print(f"energy is: {energy.shape}")
    # # axes3[0, 0].plot(step, phy_ddq[:, 5], zorder=3, label='phy_wz')
    # # axes2[1, 2].plot(step, drl_ddq[:, 5], zorder=2, label='drl_wz')
    # # axes2[1, 2].plot(step, total_ddq[:, 5], zorder=1, label='total_wz')
    # print(f"step: {step}")
    # print(f"step: {step.shape}")
    # axes3.plot(step, energy[:], zorder=1, label='energy')
    # axes3.set_xlabel('Time/s')
    # axes3.set_ylabel('energy')
    # axes3.legend()
    # fig.tight_layout()
    fig.subplots_adjust(left=0.06, right=0.943, top=0.95, bottom=0.076, wspace=0.16, hspace=0.19)
    fig2.subplots_adjust(left=0.052, right=0.986, top=0.968, bottom=0.06, wspace=0.145, hspace=0.179)
    # plt.tight_layout()
    plt.show()


def plot_trajectory2(trajectory_tensor, reference_trajectory_tensor=None):
    """
   trajectory_tensor: a numpy array [n, 4], where n is the length of the trajectory,
                       5 is the dimension of each point on the trajectory, containing [x, x_dot, theta, theta_dot]
   """
    trajectory_tensor = np.array(trajectory_tensor)
    reference_trajectory_tensor = np.array(
        reference_trajectory_tensor) if reference_trajectory_tensor is not None else None
    n, c = trajectory_tensor.shape

    y_label_list = ["x", "x_dot", "theta", "theta_dot"]

    plt.figure(figsize=(9, 6))

    for i in range(c):

        plt.subplot(c, 1, i + 1)
        plt.plot(np.arange(n), trajectory_tensor[:, i], label=y_label_list[i])

        if reference_trajectory_tensor is not None:
            plt.plot(np.arange(n), reference_trajectory_tensor[:, i], label=y_label_list[i])

        plt.legend(loc='best')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("trajectory.png", dpi=300)
    # plt.show()


def rot_mat_3d(angle, axis):
    """
    Create a 3D rotation matrix for rotating points around a specified axis.

    Parameters:
        angle (float): The angle of rotation in radians.
        axis (str): The axis of rotation ('x', 'y', or 'z').

    Returns:
        numpy.ndarray: The 3D rotation matrix.
    """
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, cos_theta, -sin_theta],
                         [0, sin_theta, cos_theta]])
    elif axis == 'y':
        return np.array([[cos_theta, 0, sin_theta],
                         [0, 1, 0],
                         [-sin_theta, 0, cos_theta]])
    elif axis == 'z':
        return np.array([[cos_theta, -sin_theta, 0],
                         [sin_theta, cos_theta, 0],
                         [0, 0, 1]])


def find_latest_file(dir):
    latest_file = None
    latest_mtime = None

    for root, dirs, files in os.walk(dir):
        for file in files:
            file_path = os.path.join(root, file)
            mtime = os.path.getmtime(file_path)
            if latest_mtime is None or mtime > latest_mtime:
                latest_file = file_path
                latest_mtime = mtime

    return latest_file


def save_trajectory_to_txt(filepath: str, outfile_path="data.csv") -> None:
    with open(filepath, 'rb') as f:
        phases = pickle.load(f)
    print(f"phases: {len(phases)}")
    print(f"phases: {phases[0]}")
    # time.sleep(123)
    zero_ref = []
    step = []
    desired_linear_vel = []
    desired_com_height = []
    desired_vx = []
    linear_vel = []
    rpy = []
    position = []
    timestamp = []
    angular_vel = []
    ground_reaction_forces = []
    phy_ddq = []
    drl_ddq = []
    total_ddq = []
    max_ddq = []
    min_ddq = []
    energy = []
    action_mode = []
    hp_action = []
    ha_action = []

    tracking_err = dict({
        'p': [],
        'rpy': [],
        'v': [],
        'rpy_dot': []
    })

    n = len(phases)
    # print(f"phases: {phases}")

    for i in range(len(phases)):
        #     tracking_error = list(phases[i]['tracking_error'])
        #     if np.isnan(tracking_error).any():
        #         tracking_err['p'].append([0, 0, 0])
        #         tracking_err['rpy'].append([0, 0, 0])
        #         tracking_err['v'].append([0, 0, 0])
        #         tracking_err['rpy_dot'].append([0, 0, 0])
        #     else:
        #         tracking_err['p'].append(tracking_error[0:3])
        #         tracking_err['rpy'].append(tracking_error[3:6])
        #         tracking_err['v'].append(tracking_error[6:9])
        #         tracking_err['rpy_dot'].append(tracking_error[9:12])
        #
        zero_ref.append(0)
        step.append(i)

        # phy_ddq.append(phases[i]['stance_ddq'][0])
        # drl_ddq.append(phases[i]['stance_ddq'][1])
        # total_ddq.append(phases[i]['stance_ddq'][2])
        # min_ddq.append(phases[i]['stance_ddq_limit'][0])
        # max_ddq.append(phases[i]['stance_ddq_limit'][1])

        position.append(phases[i]['base_position'].tolist())
        desired_com_height.append(phases[i]['desired_com_height'])
        timestamp.append(phases[i]['timestamp'].tolist())
        desired_vx.append(phases[i]['desired_vx'])
        # desired_linear_vel.append(phases[i]['desired_speed'][0])
        linear_vel.append(phases[i]['base_velocity'].tolist())
        angular_vel.append(phases[i]['base_angular_velocity'].tolist())
        rpy.append(phases[i]['base_orientation_rpy'].tolist())

        ground_reaction_forces.append(phases[i]['foot_contact_force'].tolist())
        # action_mode.append(phases[i]['action_mode'])
        # print(content[i]['desired_linear_vel'])
        # print(type(content[i]['desired_linear_vel']))
        phy_ddq.append(phases[i]['desired_acc_body_frame'].tolist())
        min_ddq.append(phases[i]['acc_min'].tolist())
        max_ddq.append(phases[i]['acc_max'].tolist())

        energy.append(phases[i]['energy'].tolist())
        action_mode.append(phases[i]['action_mode'])
        print(f"phases[i]['ha_action']: {phases[i]['ha_action']}")
        hp_action.append(phases[i]['hp_action'].cpu().numpy().squeeze())

        if phases[i]['ha_action'] is None:
            ha_action.append(([0., 0., 0., 0., 0., 0.]))
        else:
            ha_action.append(phases[i]['ha_action'].tolist())
        # dict(timestamp=self._robot.time_since_reset,
        #      base_position=torch.clone(self._robot.base_position),
        #      base_orientation_rpy=torch.clone(
        #          self._robot.base_orientation_rpy),
        #      base_velocity=torch.clone(self._robot.base_velocity_body_frame),
        #      base_angular_velocity=torch.clone(
        #          self._robot.base_angular_velocity_body_frame),
        #      motor_positions=torch.clone(self._robot.motor_positions),
        #      motor_velocities=torch.clone(self._robot.motor_velocities),
        #      motor_action=motor_action,
        #      motor_torques=self._robot.motor_torques,
        #      num_clips=self._num_clips,
        #      foot_contact_state=self._gait_generator.desired_contact_state,
        #      foot_contact_force=self._robot.foot_contact_forces,
        #      desired_swing_foot_position=desired_foot_positions,
        #      desired_acc_body_frame=self._desired_acc,
        #      solved_acc_body_frame=self._solved_acc,
        #      foot_positions_in_base_frame=self._robot.
        #      foot_positions_in_base_frame,
        #      env_action=action,
        #      env_obs=torch.clone(self._obs_buf)))

    step = np.array(step)
    drl_ddq = np.array(drl_ddq)
    phy_ddq = np.array(phy_ddq)
    total_ddq = np.array(total_ddq)
    max_ddq = np.array(max_ddq)
    min_ddq = np.array(min_ddq)
    energy = np.array(energy)
    ha_action = np.array(ha_action)
    hp_action = np.array(hp_action)
    hp_action = np.expand_dims(hp_action, axis=-1)
    print(f"hp_action: {hp_action}")
    print(f"ha_action: {ha_action}")
    ha_action = np.where(ha_action is None, 0., ha_action)
    # ha_action = ha_action.squeeze()
    # ha_action = np.expand_dims(ha_action, axis=-1)
    action_mode = np.array(action_mode)

    actions_in_ha = ha_action[action_mode == ActionMode.TEACHER]
    actions_in_hp = ha_action[action_mode == ActionMode.STUDENT]
    steps_in_ha = step[action_mode == ActionMode.TEACHER]
    steps_in_hp = step[action_mode == ActionMode.STUDENT]
    # print(f"action_mode: {action_mode}")
    # print(f"action_in_ha: {len(actions_in_ha)}")
    # print(f"action_in_hp: {len(actions_in_hp)}")
    # print(f"as: {[action_mode == ActionMode.TEACHER]}")

    tracking_err['p'] = np.asarray(tracking_err['p'])
    tracking_err['rpy'] = np.asarray(tracking_err['rpy'])
    tracking_err['v'] = np.asarray(tracking_err['v'])
    tracking_err['rpy_dot'] = np.asarray(tracking_err['rpy_dot'])
    timestamp = np.asarray(timestamp)
    position = np.asarray(position).squeeze()

    # set position to zero
    position[:, 0] = 0
    position[:, 1] = 0

    zero_ref = np.asarray(zero_ref)
    linear_vel = np.asarray(linear_vel).squeeze()

    desired_linear_vel = np.asarray(desired_linear_vel)
    desired_com_height = np.asarray(desired_com_height)
    desired_vx = np.asarray(desired_vx)

    angular_vel = np.asarray(angular_vel).squeeze()
    rpy = np.asarray(rpy).squeeze()

    position_ref = copy.deepcopy(position)
    position_ref[:, 0] = 0
    position_ref[:, 1] = 0
    position_ref[:, 2] = desired_com_height

    rpy_ref = copy.deepcopy(rpy)
    rpy_ref[:, 0] = 0
    rpy_ref[:, 1] = 0
    rpy_ref[:, 2] = 0

    linear_vel_ref = copy.deepcopy(linear_vel)
    linear_vel_ref[:, 0] = desired_vx
    linear_vel_ref[:, 1] = 0
    linear_vel_ref[:, 2] = 0.0

    angular_vel_ref = copy.deepcopy(angular_vel)
    angular_vel_ref[:, 0] = 0
    angular_vel_ref[:, 1] = 0
    angular_vel_ref[:, 2] = 0

    # print(f"action_mode: {action_mode}")

    import pandas as pd
    step = timestamp
    print(f"step: {step}")
    data = {
        'Timestamp': step.squeeze(),
        'px': position[:, 0].squeeze(),
        'py': position[:, 1].squeeze(),
        'pz': position[:, 2].squeeze(),
        'roll': rpy[:, 0].squeeze(),
        'pitch': rpy[:, 1].squeeze(),
        'yaw': rpy[:, 2].squeeze(),
        'vx': linear_vel[:, 0].squeeze(),
        'vy': linear_vel[:, 1].squeeze(),
        'vz': linear_vel[:, 2].squeeze(),
        'wx': angular_vel[:, 0].squeeze(),
        'wy': angular_vel[:, 1].squeeze(),
        'wz': angular_vel[:, 2].squeeze(),
        'px_ref': position_ref[:, 0].squeeze(),
        'py_ref': position_ref[:, 1].squeeze(),
        'pz_ref': position_ref[:, 2].squeeze(),
        'roll_ref': rpy_ref[:, 0].squeeze(),
        'pitch_ref': rpy_ref[:, 1].squeeze(),
        'yaw_ref': rpy_ref[:, 2].squeeze(),
        'vx_ref': linear_vel_ref[:, 0].squeeze(),
        'vy_ref': linear_vel_ref[:, 1].squeeze(),
        'vz_ref': linear_vel_ref[:, 2].squeeze(),
        'wx_ref': angular_vel_ref[:, 0].squeeze(),
        'wy_ref': angular_vel_ref[:, 1].squeeze(),
        'wz_ref': angular_vel_ref[:, 2].squeeze(),
    }

    df = pd.DataFrame(data)
    df.to_csv(outfile_path, index=False)
    # np.savetxt(outfile_path, (step,linear_vel[:, 0]), fmt="%f", delimiter=',')
    # np.savetxt(outfile_path, , fmt="%f", delimiter=',')


if __name__ == '__main__':

    if len(sys.argv) == 1:
        folder_name = "eval"
        file_order = -1
    else:
        # folder_name = str(sys.argv[1])
        folder_name = "eval"
        file_order = int(sys.argv[1])

    # dir_name = f"logs/{folder_name}/2024_10_23_20_21_19"
    dir_name = f"logs/{folder_name}/"

    # dir_name = "logs/robot/real_plant"
    files = os.listdir(dir_name)
    file_list = sorted(files, key=lambda x: os.path.getmtime(os.path.join(dir_name, x)))
    print(f"filepath: {dir_name}/{file_list[file_order]}")
    # fp = f"{dir_name}/eval_sim_2024_10_30_15_16_59.pkl"
    fp = f"{dir_name}/{file_list[file_order]}"
    dir_path = "logs/iclr_rebuttal/noise_and_push/Backward"
    fp = f"{dir_path}/data.pkl"

    # plot_robot_trajectory(filepath=f"{fp}")
    save_trajectory_to_txt(fp, outfile_path=f"{dir_path}/data.csv")
    plot_robot_trajectory(filepath=f"{fp}")


    # plot_robot_trajectory("saved/logs/real_plant/updated_patch.pkl")
