"""
This is a templated of setting a new environment
"""

import os
import sys
from quad_gym.env.randomizer import a1_randomizer_terrain
from quad_gym.env.robots import a1
from quad_gym.env.env_config import TerrainTypeDict
from quad_gym.env.randomizer import a1_randomizer_dynamics
from quad_gym.env.sensors import sensor_wrappers
from quad_gym.wrapper import observation_wrapper
from quad_gym.task import curriculum_task
from quad_gym.env import locomotion_gym_env
from quad_gym.wrapper.action_wrapper import ActionRestrain, DiagonalAction, RandoDirWrapper
from quad_gym.gym_config import GymConfig
from quad_gym.env.env_config import AllSensors
from quad_gym.task.task_config import AllTasks


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_a1_ground_env(gym_config: GymConfig):
    if gym_config.TaskParams.sub_goal:
        gym_config.SimParams.enable_hard_reset = False

    sensor_name_list = list(gym_config.RobotParams.robot_sensors_list) + list(gym_config.SceneParams.env_sensors_list)

    sensor_list = []

    for sensor_names in sensor_name_list:
        if sensor_names not in AllSensors:
            raise ValueError("Unknown sensor type: {}".format(sensor_names))

        if sensor_names == "EnvLastActionSensor":

            sensor = AllSensors[sensor_names](num_actions=a1.NUM_MOTORS)
        else:
            sensor = AllSensors[sensor_names]()

        if sensor_names != "EnvGoalPosSensor":

            if gym_config.TaskParams.include_historic_sensors:
                sensor = sensor_wrappers.HistoricSensorWrapper(sensor, num_history=gym_config.TaskParams.num_history)

        sensor_list.append(sensor)

    if gym_config.SceneParams.terrain_type == "mount" or gym_config.SceneParams.terrain_type == "hill":
        gym_config.TaskParams.task_params.check_contact = True

    task = AllTasks[gym_config.TaskParams.task_type](gym_config.TaskParams.task_params)

    randomizers = []

    if gym_config.TaskParams.domain_randomization:
        randomizer = a1_randomizer_dynamics.DynamicsRandomizer(verbose=False)
        randomizers.append(randomizer)

    if gym_config.SceneParams.terrain_randomizer:
        terrain_randomizer = a1_randomizer_terrain.TerrainRandomizer(
            mesh_filename='terrain9735.obj',
            terrain_type=TerrainTypeDict[gym_config.SceneParams.terrain_type],
            mesh_scale=[0.6, 0.3, 0.2],  # todo check what is this
            height_range=gym_config.SceneParams.high_range,
            random_shape=gym_config.SceneParams.terrain_random_shape,
            moving=gym_config.SceneParams.moving,
        )
        randomizers.append(terrain_randomizer)

    env = locomotion_gym_env.LocomotionGymEnv(gym_config, task, sensor_list, randomizers)

    env = observation_wrapper.ObservationDictionaryToArrayWrapper(env)

    if gym_config.RobotParams.controller_clip_num is not None:
        env = ActionRestrain(env, gym_config.RobotParams.controller_clip_num)

    if gym_config.TaskParams.diagonal_act:
        env = DiagonalAction(env)

    if gym_config.TaskParams.random_dir:
        assert gym_config.SceneParams.terrain_type == "mount" or gym_config.SceneParams.terrain_type == "hill"
        env = RandoDirWrapper(env, gym_config.TaskParams.dir_update_interval)

    if gym_config.TaskParams.curriculum:
        env = curriculum_task.CurriculumWrapperEnv(env, episode_length_start=1000,
                                                   episode_length_end=2000,
                                                   curriculum_steps=10000000,
                                                   num_parallel_envs=8)
    return env


if __name__ == "__main__":
    env = build_a1_ground_env(GymConfig())
    import time

    c_t = time.time()
    env.reset()
    for i in range(100000000):
        print("reset")
        env.reset()
        for j in range(1000):
            _, _, ter, trunc, _ = env.step(env.action_space.sample())
            if ter:
                print("reset")
                env.reset()
    print(time.time() - c_t)
    print(env.count_t)
    print(10000 / (time.time() - c_t))
