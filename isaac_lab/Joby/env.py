import os
root_navsim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))

"""Rest everything follows."""

import math
import torch
from scipy.spatial.transform import Rotation

import isaaclab.envs.mdp as mdp
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
import isaaclab.sim as sim_utils
# para la flecha
# -------------------------------------------------
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
# -------------------------------------------------
from isaaclab.assets import AssetBaseCfg, Articulation, ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg, ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CommandTermCfg, CommandTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import DCMotorCfg
import isaaclab.utils.math as math_utils
from . import rewards as my_rewards
from . import terminations as my_terminations
from .flight_plan import FlightPlan
# ruido gaussiano para la regularización -- Teresa ---
from isaaclab.utils.noise import GaussianNoiseCfg
# para las flechas
from isaaclab.utils.math import quat_from_matrix

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for a UAV scene"""
    pass

@configclass
class ActionsCfg:
    """Action specifications for the environtment"""
    joint_efforts = mdp.JointEffortActionCfg(asset_name = )

@configclass
class ObservationCfg:
    """Observation specifications for the environtment"""
    pass

@configclass
class CommandCfg:
    """Command specifications for the environment."""
    pass

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    pass

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    pass

@configclass
class EventCfg:
    """Event specifications for the environment."""
    pass

@configclass
class ObervervationCfg:
    """Observation specifications for the environment."""
    pass


@configclass
class UAVEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the UAV environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(env_spacing=200.0, replicate_physics=False)
    seed: int = 0
    
    # Basic settings
    actions: ActionsCfg = ActionsCfg()
    observations: ObervervationCfg = ObervervationCfg()
    commands: CommandCfg = CommandCfg()
    events: EventCfg = EventCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    is_test_mode: bool = False

    def __post_init__(self):
        """Post initialization"""
        # pasos que se usan en la red de PPO
        self.num_steps_per_env = 100
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # 50 Hz de actualización para la IA
        self.episode_length_s = 25.0 # al ser punto cambiante no debe ser tan largo
        # simulation settings
        self.sim.dt = 0.005  # 100 Hz para las físicas
        self.sim.render_interval = self.decimation