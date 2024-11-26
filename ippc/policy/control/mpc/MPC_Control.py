from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from policy.control.mpc.lib import a1_sim
from policy.control.mpc.lib.a1_sim import *
from policy.control.mpc.lib import torque_stance_leg_controller
from policy.control.mpc.lib import raibert_swing_leg_controller
from policy.control.mpc.lib import openloop_gait_generator
from policy.control.mpc.lib import mpc_controller
from policy.control.mpc.lib import gait_generator as gait_generator_lib
from policy.control.mpc.lib import com_velocity_estimator
import scipy.interpolate
import dataclasses
from dataclasses import field
from typing import List

@dataclasses.dataclass
class MPCPatternTripod:
    duty_factor: List[float] = field(default=(0.8, 0.8, 0.8, 0.8))
    init_phase_full_cycle: List[float] = dataclasses.field(default=(0., 0.25, 0.5, 0.))
    init_leg_state: List[int] = dataclasses.field(default=(gait_generator_lib.LegState.STANCE,
                                                           gait_generator_lib.LegState.STANCE,
                                                           gait_generator_lib.LegState.STANCE,
                                                           gait_generator_lib.LegState.SWING))


@dataclasses.dataclass
class MPCPatternTrot:
    duty_factor: List[float] = field(default=(0.6, 0.6, 0.6, 0.6))
    init_phase_full_cycle: List[float] = field(default=(0.9, 0, 0, 0.9))
    init_leg_state: List[int] = field(default=(gait_generator_lib.LegState.SWING,
                                               gait_generator_lib.LegState.STANCE,
                                               gait_generator_lib.LegState.STANCE,
                                               gait_generator_lib.LegState.SWING,))


@dataclasses.dataclass
class MPCPatternStanding:
    duty_factor: List[float] = field(default=(1.0, 1.0, 1.0, 1.0))
    init_phase_full_cycle: List[float] = field(default=(0., 0., 0., 0.))
    init_leg_state: List[int] = field(default=(gait_generator_lib.LegState.STANCE,
                                               gait_generator_lib.LegState.STANCE,
                                               gait_generator_lib.LegState.STANCE,
                                               gait_generator_lib.LegState.STANCE))


@dataclasses.dataclass
class MPCConfig:
    num_simulation_iteration_steps: int = 300
    stance_duration_seconds: List[float] = field(default=(0.3, 0.3, 0.3, 0.3))
    locomotion_pattern: str = 'trot'
    locomotion_params: object = MPCPatternTrot()

    def init_pattern(self):
        if self.locomotion_pattern == 'trot':
            self.locomotion_params = MPCPatternTrot()
        elif self.locomotion_pattern == 'tripod':
            self.locomotion_params = MPCPatternTripod()
        elif self.locomotion_pattern == 'standing':
            self.locomotion_params = MPCPatternStanding()
        else:
            raise ValueError("Unknown locomotion pattern: {}".format(self.locomotion_pattern))


class MPC:
    def __init__(self, params: MPCConfig, robot):
        self.params = params

        self.controller = None
        self._setup_controller(robot)
        # Here we will need motor model to convert Hybrid control format to torque
        self._kp = a1_sim.SimpleRobot.GetMotorPositionGains()
        self._kd = a1_sim.SimpleRobot.GetMotorVelocityGains()
        # self.motor_model = A1MotorModel(kp=self._kp, kd=self._kd, motor_control_mode=a1_sim.MOTOR_CONTROL_HYBRID)

    def get_motor_states(self, robot):
        """To get the motor angles and velocities for converting hybrid control command to position"""
        """Here we can call robot.GetMotorVelocities() to get noisy reading"""
        motor_angles = robot.GetTrueMotorAngles()
        motor_velocities = robot.GetTrueMotorVelocities()
        return motor_angles, motor_velocities

    def _setup_controller(self, robot):
        """Demonstrates how to create a locomotion controller."""
        self.desired_speed = (0, 0)
        self.desired_twisting_speed = 0

        self.gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
            robot,
            stance_duration=self.params.stance_duration_seconds,
            duty_factor=self.params.locomotion_params.duty_factor,
            initial_leg_phase=self.params.locomotion_params.init_phase_full_cycle,
            initial_leg_state=self.params.locomotion_params.init_leg_state)
        self.state_estimator = com_velocity_estimator.COMVelocityEstimator(robot, window_size=20)
        self.sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
            robot,
            self.gait_generator,
            self.state_estimator,
            desired_speed=self.desired_speed,
            desired_twisting_speed=self.desired_twisting_speed,
            desired_height=a1_sim.MPC_BODY_HEIGHT,
            foot_clearance=0.01)

        self.st_controller = torque_stance_leg_controller.TorqueStanceLegController(
            robot,
            self.gait_generator,
            self.state_estimator,
            desired_speed=self.desired_speed,
            desired_twisting_speed=self.desired_twisting_speed,
            desired_body_height=robot.MPC_BODY_HEIGHT,
            body_mass=robot.MPC_BODY_MASS,
            body_inertia=robot.MPC_BODY_INERTIA)

        self.controller = mpc_controller.LocomotionController(
            robot,
            gait_generator=self.gait_generator,
            state_estimator=self.state_estimator,
            swing_leg_controller=self.sw_controller,
            stance_leg_controller=self.st_controller,
            clock=robot.GetTimeSinceReset)

        self.controller.reset()

    def reset(self, robot):
        self._setup_controller(robot)

    def update_controller_params(self, lin_speed, ang_speed):
        self.controller.swing_leg_controller.desired_speed = lin_speed
        self.controller.swing_leg_controller.desired_twisting_speed = ang_speed
        self.controller.stance_leg_controller.desired_speed = lin_speed
        self.controller.stance_leg_controller.desired_twisting_speed = ang_speed

    def generate_example_linear_angular_speed(self, t):
        """Creates an example speed profile based on time for demo purpose."""
        """This is a basic planner """
        vx = 0.6 * a1_sim.MPC_VELOCITY_MULTIPLIER
        vy = 0.2 * a1_sim.MPC_VELOCITY_MULTIPLIER
        wz = 0.8 * a1_sim.MPC_VELOCITY_MULTIPLIER

        time_points = (0, 5, 10, 15, 20, 25, 30)
        speed_points = ((0, 0, 0, 0), (0, 0, 0, wz), (vx, 0, 0, 0), (0, 0, 0, -wz), (0, -vy, 0, 0),
                        (0, 0, 0, 0), (0, 0, 0, wz))

        speed = scipy.interpolate.interp1d(
            time_points,
            speed_points,
            kind="previous",
            fill_value="extrapolate",
            axis=0)(
            t)

        return speed[0:3], speed[3]

    def convert2torque(self, hybrid_actions, robot):
        assert len(hybrid_actions) == MOTOR_COMMAND_DIMENSION * NUM_MOTORS
        motor_angle, motor_velocity = self.get_motor_states(robot)
        kp = hybrid_actions[POSITION_GAIN_INDEX::MOTOR_COMMAND_DIMENSION]
        kd = hybrid_actions[VELOCITY_GAIN_INDEX::MOTOR_COMMAND_DIMENSION]
        desired_motor_angles = hybrid_actions[POSITION_INDEX::MOTOR_COMMAND_DIMENSION]
        desired_motor_velocities = hybrid_actions[VELOCITY_INDEX::MOTOR_COMMAND_DIMENSION]
        additional_torques = hybrid_actions[TORQUE_INDEX::MOTOR_COMMAND_DIMENSION]
        motor_torques = -1 * (kp * (motor_angle - desired_motor_angles)) - kd * (
                    motor_velocity - desired_motor_velocities) + additional_torques
        return motor_torques
