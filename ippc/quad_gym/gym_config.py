from dataclasses import dataclass
from dataclasses_json import dataclass_json
from quad_gym.task.task_config import TaskConfig
from quad_gym.env.env_config import SceneConfig, RobotConfig, SimConfig


@dataclass
@dataclass_json
class GymConfig:
    SceneParams: SceneConfig = SceneConfig()
    RobotParams: RobotConfig = RobotConfig()
    SimParams: SimConfig = SimConfig()
    TaskParams: TaskConfig = TaskConfig()


