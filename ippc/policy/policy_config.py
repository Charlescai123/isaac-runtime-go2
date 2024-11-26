import dataclasses
from dataclasses import field
from typing import List
from policy.control.mpc.MPC_Control import MPC, MPCConfig
from policy.control.drl.DDPG_Control import DDPGParams, DDPGAgent
from policy.control.drl.TD3_Control import TD3Params, TD3Agent
from policy.control.drl.SAC_Control import SACParams, SACAgent
from policy.control.drl.PPO_Control import PPOParams, PPOAgent

AllControlModels = {
    "mpc": {"params": MPCConfig(), "model": MPC},
    "ddpg": {"params": DDPGParams(), "model": DDPGAgent},
    "td3": {"params": TD3Params(), "model": TD3Agent},
    "sac": {"params": SACParams(), "model": SACAgent},
    "ppo": {"params": PPOParams(), "model": PPOAgent}
}


@dataclasses.dataclass
class PerceptionModuleConfig:
    input_type: str = field(default="image")
    output_type: str = field(default="image")
    perception_model: str = field(default="resnet18")
    perception_model_path: str = field(default="")


@dataclasses.dataclass
class PlanningModuleConfig:
    input_type: str = field(default="image")
    output_type: List[float] = field(default=(0, 0, 0))
    planning_model: str = field(default="resnet18")


@dataclasses.dataclass
class ControlModuleConfig:
    input_type: str = field(default="image")
    output_type: List[float] = field(default=(0, 0, 0))
    control_model: str = field(default="mpc")
    pretrained_model_path: str = None
    control_model_params: object = field(default=DDPGParams())

    def __post_init__(self):
        self.control_model_params = AllControlModels[self.control_model]["params"]


@dataclasses.dataclass
class PolicyConfig:
    PerceptionModuleParams: PerceptionModuleConfig = PerceptionModuleConfig()
    PlanningModuleParams: PlanningModuleConfig = PlanningModuleConfig()
    ControlModuleParams: ControlModuleConfig = ControlModuleConfig()
