import os
root_navsim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))

"""Rest everything follows."""

import math
import torch

import isaaclab.envs.mdp as mdp
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
import isaaclab.sim as sim_utils
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
from . import observations as my_observations
# ruido gaussiano para la regularización -- Teresa ---
from isaaclab.utils.noise import GaussianNoiseCfg
# para las flechas
from isaaclab.utils.math import quat_from_matrix
# ---------------------------------------------------

#ToDo: Borrar y modificar todo lo que pase a cuaternion (quat)
# |---------------------------------------------------------|
# |--------------------- ACTIONS ---------------------------|
# |---------------------------------------------------------|
class JobyActionTerm(ActionTerm):
    """Action term for the UAV."""

    _asset: Articulation
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: JobyActionTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # NN outputs: 12 (6 rotor orientations + 6 blade speeds)
        self._raw_actions = torch.zeros(env.num_envs, 12, device=self.device)
        # Processed actions: orientations and forces to apply (6 rotors + 6 blades)
        self._processed_actions = torch.zeros(env.num_envs, 12, device=self.device)
        # Maximum number of links (1 body + 6 rotors + 6 blades)
        self.max_prim_links = 13
        # Number of rotors
        self.num_rotors = 6
        # Hover omega (rad/s)
        self.hover_vel = 80
        # Pre-allocate forces tensor for all environments and links (body + rotors + blades)
        self.forces_to_apply = torch.zeros(env.num_envs, self.max_prim_links, 3, device=self.device)

        # Create all positions at once in a single tensor operation
        #poner rotores,blades y bodys
        position_data = torch.tensor([
            [0, 0, 0],
            
            [1.6,2.3,0.8], #JRotorNW
            [1.6,-2.3,0.8], #JRotorNE
            [-0.8,5,0.8], #JRotorW
            [-0.8,-5,0.8], #JRotorE
            [-3.2,2.3,0.8], #JRotorSW
            [-3.2,-2.3,0.8], #JRotorSE
            
            [2.4,2.3,0.8], #JBladeNW 
            [2.4,-2.3,0.8], #JBladeNE -
            [0,5,0.8], #JBladeW -
            [0,-5,0.8], #JBladeE
            [-2.4,2.3,0.8], #JBladeSW-
            [-2.4,-2.3,0.8], #JBladeSE 

        ], device=self.device)
        self.action_mins = torch.tensor([-90,-90,-90,-90,-90,-90,0,-120,-120,0,-120,0],device = self.device)
        self.action_maxs = torch.tensor([0,0,0,0,0,0,120,0,0,120,0,120],device = self.device)
        self.action_scale = (self.action_maxs - self.action_mins) / 2.0
        self.action_offset = (self.action_maxs + self.action_mins) / 2.0
        # Expand to all environments (no copy, just view)
        self.positions = position_data.unsqueeze(0).expand(env.num_envs, -1, -1).clone()

        # Create indexes efficiently
        self.indexes = torch.arange(env.num_envs, device=self.device)

        # self._asset = env.scene[cfg.asset_name]

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
        
    def process_actions(self, actions: torch.Tensor):
        # actions contiene la orientación de los rotores y la velocidad de las palas
        # Si algun dron reventó y el NaN se cuela, lo ponemos a 0
        # Evitar que valores inválidos lleguen a la simulación, poniendo los valores entre -1 y 1
        #esto deberia ser 6 para la orientacion dde los rotores y 6 para la velocidad
        actions = torch.clamp(actions, -1.0, 1.0)
        actions = torch.nan_to_num(actions, nan=0.0)
        
        self._raw_actions = (actions * self.action_scale) + self.action_offset
        
        # Las acciones producidas por la red están normalizadas (cambiarlo entre 0 y 90)
        #actions = torch.clamp(actions, -0, 1.0)
        
        # Guardar las 12 acciones originales:
        # 0:6  -> orientación de los rotores
        # 6:12 -> velocidad de las palas
        # self._raw_actions[:] = actions
        
        # Calcular el peso del aerotaxi
        mass = self._asset.data._root_physx_view.get_masses().sum(dim=1, keepdim = True)
        mg = mass * 9.81
        
        # Calcular k usando la condición de hover        
        k = mg / torch.square(self.hover_vel)
        
        # De momento se almacenan las orientaciones sin utilizarlas
        self._processed_actions[:, 0:6] = self._raw_actions[:, 0:6]

        # Apply 'vel to thrust' formula (T = k * omega^2)
        self._processed_actions[:, 6:12] = k * torch.square(self._raw_actions[:, 6:12])

        # Assign forces to the correct links (body + rotors + blades) at Z axis
        self.forces_to_apply[:, 7:13, 2] = self._processed_actions[:, 6:12] - mg / self.num_rotors
        
        
        

    def apply_actions(self):
        self._asset.root_physx_view.apply_forces_and_torques_at_position(
            force_data=self.forces_to_apply, 
            position_data=self.positions,
            indices=self.indexes,
            is_global=False
        )

@configclass
class JobyActionTermCfg(ActionTermCfg):
    """Action term configuration for the UAV."""

    class_type: type = JobyActionTerm
    """Class type of the action term."""

@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    # joint_efforts = mdp.JointEffortActionCfg(
    #     asset_name = "Joby", 
    #     joint_names = [
    #         "JRotorNW",
    #         "JRotorNE",
    #         "JRotorW",
    #         "JRotorE",
    #         "JRotorSW",
    #         "JRotorSE",
    #         "JBladeNW",
    #         "JBladeNE",
    #         "JBladeW",
    #         "JBladeE",
    #         "JBladeSW",
    #         "JBladeSE"
    #     ]
    # )
    rotor_ori_and_blade_vel = JobyActionTermCfg(asset_name="Joby")

@configclass
class ObervervationCfg:
    """Observation specifications for the environment."""

    # Actor: lo que el dron verá en train y test
    @configclass
    class PolicyCfg(ObsGroup):
        """Observation group for the policy."""
        # dist = ObsTerm(func=my_obs_dist2, params={"asset_cfg": SceneEntityCfg(name="Joby")},noise=GaussianNoiseCfg(std=0.01))
        command = ObsTerm(func=my_observations.my_obs_get_command, params={"asset_cfg": SceneEntityCfg(name="Joby")},noise=GaussianNoiseCfg(std=0.05))
        lin_vel = ObsTerm(func=my_observations.my_obs_lin_vel, params={"asset_cfg": SceneEntityCfg(name="Joby")},noise=GaussianNoiseCfg(std=0.05))
        ang_vel = ObsTerm(func=my_observations.my_obs_ang_vel, params={"asset_cfg": SceneEntityCfg(name="Joby")},noise=GaussianNoiseCfg(std=0.05))
        #projected_gravity = ObsTerm(func=my_observations.my_obs_projected_gravity, params={"asset_cfg": SceneEntityCfg(name="Joby")},noise=GaussianNoiseCfg(std=0.01))
        #target_vel = ObsTerm(func=my_observations.my_obs_target_vel, params={"asset_cfg": SceneEntityCfg(name="Joby")},noise=GaussianNoiseCfg(std=0.01)) 
        yaw_error = ObsTerm(func=my_observations.my_obs_yaw_error, params={"asset_cfg": SceneEntityCfg(name="Joby")},noise=GaussianNoiseCfg(std=0.01))

        def __post_init__(self):
            self.enable_corruption = True  # Regularización con ruido en las observaciones
            self.concatenate_terms = True
            # self.history_length = 3 
            # self.flatten_history_dim = True

    # Crítico: la corrección que se hará sobre lo que ve el dron en train. En test no hay crítico
    # Por eso aquí vamos a incluir la velocidad del punto guía, para que pueda ajustarse a ella en train, pero en test no
    # la vea
    @configclass
    class CriticCfg(ObsGroup):
        """Lo que el entrenador sabe (la verdad absoluta, sin ruido)"""
        # 1. Posición y velocidad de la Policy (pero sin ruido)
        command = ObsTerm(func=my_observations.my_obs_get_command, params={"asset_cfg": SceneEntityCfg(name="Joby")})
        lin_vel = ObsTerm(func=my_observations.my_obs_lin_vel, params={"asset_cfg": SceneEntityCfg(name="Joby")})
        ang_vel = ObsTerm(func=my_observations.my_obs_ang_vel, params={"asset_cfg": SceneEntityCfg(name="Joby")})
        #projected_gravity = ObsTerm(func=my_observations.my_obs_projected_gravity, params={"asset_cfg": SceneEntityCfg(name="Joby")})
        #target_vel = ObsTerm(func=my_observations.my_obs_target_vel, params={"asset_cfg": SceneEntityCfg(name="Joby")})
        yaw_error = ObsTerm(func=my_observations.my_obs_yaw_error, params={"asset_cfg": SceneEntityCfg(name="Joby")})

        def __post_init__(self):
            self.enable_corruption = False # El crítico no necesita ruido
            self.concatenate_terms = True
            # self.history_length = 3 
            # self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()



# |---------------------------------------------------------|
# |--------------------- COMMANDS --------------------------|
# |---------------------------------------------------------|

#ToDo: TARGET_VEL ¿2 o 1? 
class UAVcommandTerm(CommandTerm):
    """Command term for the UAV that generates meaningful velocity and yaw rate commands."""
    
    _asset: Articulation
    
    def __init__(self, cfg: UAVcommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._marker_visualizer = VisualizationMarkers(VISUAL_TARGET_CFG) #exoected pos
        self._marker_visualizer_green = VisualizationMarkers(GREEN_ARROW_CFG) #espected vel
        self._marker_visualizer_red = VisualizationMarkers(RED_ARROW_CFG)
        self._command_visualizer_yellow = VisualizationMarkers(YELLOW_ARROW_CFG) #yellor vel
        
        # self._asset = env.scene[cfg.asset_name]
        
        # ponemos la velocidad y giro al que deberá ir nuestro punto guía
        # velocidad en x,y,z (tamaño 3)
        self._command = torch.zeros(self.num_envs, 4, device=self.device) 
        # almacenar puntos futuros
        self.lookahead_time = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command
    
    def _update_metrics(self):
        pass

    def get_arrow_quat(self, direction_vec):
        """Convierte un vector de dirección en un cuaternión que apunta hacia esa dirección."""
        # 1. Normalizamos la dirección (Eje X de la flecha)
        x_axis = torch.nn.functional.normalize(direction_vec, dim=-1)
        
        # 2. Creamos una base ortonormal (Gram-Schmidt)
        # Usamos un vector 'up' auxiliar
        up = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(direction_vec.shape[0], 1)
        # Eje Y = Up x X
        y_axis = torch.nn.functional.normalize(torch.cross(up, x_axis, dim=-1), dim=-1)
        # Eje Z = X x Y
        z_axis = torch.cross(x_axis, y_axis, dim=-1)
        
        # 3. Formamos la matriz de rotación [N, 3, 3] y convertimos a cuaternión
        # Isaac Lab espera las columnas en orden X, Y, Z
        res_matrix = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        return quat_from_matrix(res_matrix)

    def compute_flight_command(self,expected_pos, expected_vel, expected_yaw, pos, lin_vel, yaw, t_to_solve):
        
        # COMPUTING CORRECTION VELOCITY (to achieve status.pos in 'tToSolve' seconds)
        correction_vel = (expected_pos - pos) / t_to_solve
 
        # COMPUTING COMMANDED VELOCITY
        command_linear_vel = expected_vel + correction_vel
 
        # SMOOTHING COMMANDED VELOCITY
        variation_vel = command_linear_vel - lin_vel
        command_linear_vel = lin_vel + variation_vel
 
        # COMPUTING TARGET ERROR YAW
        yaw_error = expected_yaw - yaw
        # Normalize to [-pi, pi]
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
 
        # COMPUTING TARGET ANGULAR VELOCITY
        command_yaw_rotation = yaw_error / t_to_solve
        self._command[:,:3] = command_linear_vel
        self._command[:,3] = command_yaw_rotation
        
    
    def _resample_command(self, env_ids: torch.Tensor):
        #ToDo: iniciar al principio con un 3 
        
        self._asset.update(self.dt)
        uav_pos_w = self._asset.data.root_com_pos_w[env_ids] - self._env.scene.env_origins[env_ids]
        self.target_pos[env_ids] = uav_pos_w[:, :3]
        
        resample_amount = len(env_ids)

        # Definimos velocidad aleatoria del punto guía (3 a 6 m/s)
        speed = torch.rand(resample_amount, device=self.device) * 3.0 + 3.0
        
        # Dirección inicial aleatoria en el plano XY
        angle = torch.rand(resample_amount, device=self.device) * 2 * math.pi
        self.target_vel[env_ids, 0] = torch.cos(angle) * speed
        self.target_vel[env_ids, 1] = torch.sin(angle) * speed
        # self.target_vel[env_ids, 2] = torch.rand(resample_amount, device=self.device) * 0.5 + 0.2
        self.target_vel[env_ids, 2] = (torch.rand(resample_amount, device=self.device) - 0.5) * 0.8
        # Curvatura aleatoria (Yaw rate del camino: -0.3 a 0.3 rad/s)
        # Esto genera círculos, curvas en S o rectas aleatorias
        max_yaw_rate_change = 0.25
        self.target_yaw[env_ids] = (torch.rand(resample_amount, device=self.device) - 0.5) * 2.0 * max_yaw_rate_change
        self.target_yaw_prev[env_ids] = self.target_yaw[env_ids].clone()
        self.lookahead_time[env_ids] = torch.rand(resample_amount, device=self.device) * 0.5 + 0.4

    def _update_command(self):
        # segundo o más frame de tiempo
        """Mueve el fantasma en cada paso de física."""
        cos_theta = torch.cos(self.target_yaw * self.dt)
        sin_theta = torch.sin(self.target_yaw * self.dt)

        vx = self.target_vel[:, 0].clone()
        vy = self.target_vel[:, 1].clone()

        self.target_vel[:, 0] = vx * cos_theta - vy * sin_theta
        self.target_vel[:, 1] = vx * sin_theta + vy * cos_theta

        # Variación de altura dinámica durante el vuelo (cada ~2 seg)
        change_z = torch.rand(self.num_envs, device=self.device) < 0.01
        self.target_vel[change_z, 2] = (torch.rand(change_z.sum(), device=self.device) - 0.5) * 1.6 # -0.8 a 0.8 m/s de variación de velocidad 

        max_yaw_change = 0.02 * self.dt
        self.target_yaw = self.target_yaw_prev + torch.clamp(
            self.target_yaw - self.target_yaw_prev, -max_yaw_change, max_yaw_change
        )
        self.target_yaw_prev = self.target_yaw.clone()

        # Valla Virtual (Límite 100m spacing -> 50m radio)
        limite_xy = 45.0
        limite_z = (1.75, 20.0)
        
        # # Rebote XY
        margin = 3.0

        out_x = (self.target_pos[:,0].abs() > limite_xy - margin)
        self.target_vel[out_x,0] *= -1.0

        out_y = (self.target_pos[:,1].abs() > limite_xy - margin)
        self.target_vel[out_y,1] *= -1.0

        # Z
        at_top = (self.target_pos[:,2] > limite_z[1]) & (self.target_vel[:,2] > 0)
        at_bot = (self.target_pos[:,2] < limite_z[0]) & (self.target_vel[:,2] < 0)
        self.target_vel[at_top | at_bot,2] *= -1.0

        # limitación de velocidad máxima
        max_speed = 6.0  # m/s
        speed = torch.norm(self.target_vel[:, :2], dim=1, keepdim=True)
        scale = torch.clamp(max_speed / (speed + 1e-6), max=1.0)
        self.target_vel[:, :2] *= scale
        # Actualizar posición del punto guía
        self.target_pos += self.target_vel * self.dt

        #ToDo: cambiar el punto futuro que ve
        lookahead_pos = self.target_pos + self.target_vel * self.lookahead_time
        lookahead_pos = self.target_pos + self.target_vel * self.lookahead_time[:, None]

        # uav_pos_local = self._asset.data.root_com_pos_w[:, :3] - self._env.scene.env_origins[:, :3]

        # rel_vec_w = lookahead_pos - uav_pos_local

        # quat_inv = math_utils.quat_inv(self._asset.data.root_com_quat_w)

        # rel_vec_body = math_utils.quat_apply(quat_inv, rel_vec_w)
        _, _, yaw = math_utils.euler_xyz_from_quat(self._asset.data.root_com_quat_w)
        self.compute_flight_command(
            self.target_pos, 
            self.target_vel,
            self.target_yaw,
            self._asset.data.root_com_pos_w,
            self._asset.data.root_com_lin_vel_b,
            yaw,
            2,
        )
        
        # Visualización dinámica
        target_pos_w = self.target_pos + self._env.scene.env_origins
        self._marker_visualizer.visualize(translations=target_pos_w)

        # --- Flecha orientación objetivo (verde)
        delta_xy = self.target_pos[:, :2] - (self._asset.data.root_com_pos_w[:, :2] - self._env.scene.env_origins[:, :2])
        unit_vec = delta_xy / (torch.norm(delta_xy, dim=1, keepdim=True) + 1e-6)
        arrow_length = 2.0  # m
        arrow_vec = torch.zeros(self.num_envs, 3, device=self.device)
        arrow_vec[:, 0] = unit_vec[:, 0] * arrow_length
        arrow_vec[:, 1] = unit_vec[:, 1] * arrow_length
        arrow_vec[:, 2] = 0.1
        origin_pos_w = self._asset.data.root_com_pos_w
        # 1. Flecha Objetivo (Verde)
        # Vector desde el dron al target
        target_dir_w = target_pos_w - origin_pos_w
        quat_target = self.get_arrow_quat(target_dir_w)
        
        # 2. Flecha Dron (Roja)
        # --- VISUALIZAR ---

        # Si usas el mismo y quieres ver AMBAS, concatena:
        uav_pos_w = self._asset.data.root_com_pos_w[:, :3]
        target_pos_w = self.target_pos + self._env.scene.env_origins

        # 2. FLECHA OBJETIVO (Verde - Dirección hacia donde tiene que ir)
        target_dir_w = target_pos_w - uav_pos_w
        quat_target = self.get_arrow_quat(target_dir_w)
        
        # 3. FLECHA DRON (Roja - Hacia donde mira el morro del dron)
        # El forward del dron es su propio quaternion (si el asset mira hacia +X)
        quat_drone = self._asset.data.root_com_quat_w
        
        self._marker_visualizer_red.visualize(
            translations=uav_pos_w + torch.tensor([0, 0, 2.0], device=self.device), 
            orientations=quat_target,
            scales=torch.tensor([10.0, 3.0, 3.0], device=self.device).repeat(self.num_envs, 1)
        )

        # Flecha Roja (Orientación actual)
        self._marker_visualizer_green.visualize(
            translations=uav_pos_w + torch.tensor([0, 0, 1.0], device=self.device),
            orientations=quat_drone,
            scales=torch.tensor([8.0, 3.0, 3.0], device=self.device).repeat(self.num_envs, 1)
        )
        
        # command arrow
        self._marker_visualizer_green.visualize(
            translations=uav_pos_w + torch.tensor([0, 0, 3.0], device=self.device),
            orientations=self._command[:,:3],
            scales=torch.tensor([8.0, 3.0, 3.0], device=self.device).repeat(self.num_envs, 1)
        )
        

@configclass
class UAVcommandTermCfg(CommandTermCfg):
    """Command term configuration for the UAV."""

    class_type: type = UAVcommandTerm
    """Class type of the command term."""
    asset_name: str = "Joby"

@configclass
class CommandCfg:
    """Command specifications for the environment."""
    
    vel_command = UAVcommandTermCfg(asset_name="Joby",resampling_time_range=(25, 25)) # cada 25 segundos cambiamos


# |---------------------------------------------------------|
# |--------------------- EVENTS ----------------------------|
# |---------------------------------------------------------|

@configclass
class EventCfg:
    """Event specifications for the environment."""

    # DOMAIN RANDOMIZATION: posición, velocidad, roll, pitch, yaw, masa y velocidad del viento
    reset_pos = EventTerm(
        func=mdp.reset_root_state_uniform, 
        mode="reset",
        params={
            "pose_range": {
                "x": (-5.0, 5.0), 
                "y": (-5.0, 5.0), 
                "z": (8.0, 12.0),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-3.14, 3.14)
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5)
            },
            "asset_cfg": SceneEntityCfg(name="Joby")
        }
    )

    # Masa: 0.9 su masa y 1.1 su masa (con pasajeros, o por si alguno es especialmente menos pesado)
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="Joby"),
            "mass_distribution_params": (0.9, 1.1), 
            "operation": "scale"
        }
    )
    # Viento: el dron recibirá una corriente de aire en diversas direcciones de forma aleatoria.
    randomize_wind = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="Joby"),
            "velocity_range": {
                "x": (-1.0, 1.0), 
                "y": (-1.0, 1.0),
                "z": (-0.5, 0.5)
            }
        }
    )


# |---------------------------------------------------------|
# |--------------------- REWARDS ---------------------------|
# |---------------------------------------------------------|

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    rew_attitude_stability2 = RewTerm(func=my_rewards.rew_attitude_stability3, weight=2.0)
    rew_ang_vel_stability4 = RewTerm(func=my_rewards.rew_ang_vel_stability5, weight=1.0)
    rew_pos2 = RewTerm(func=my_rewards.rew_track_pos, weight=2.0)
    rew_action_rate = RewTerm(func=my_rewards.rew_action_rate, weight=0.1)
    rew_heading = RewTerm(func=my_rewards.rew_heading, weight=3.0)
    rew_track_vel3 = RewTerm(func=my_rewards.rew_track_vel3, weight=8.0)

# |---------------------------------------------------------|
# |--------------------- TERMINATIONS ----------------------|
# |---------------------------------------------------------|

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    below_min_altitude = DoneTerm(
        func=my_terminations.below_min_altitude,
        params={"min_altitude": 0.05,} # el centro de masas del dron está a 5.06m del suelo.
    )
    bad_attitude = DoneTerm(func=my_terminations.roll_pitch_termination)
    safety_shutdown = DoneTerm(func=my_terminations.are_nan_or_exploded)


# |---------------------------------------------------------|
# |--------------------- SCENE -----------------------------|
# |---------------------------------------------------------|

VISUAL_TARGET_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/target_commands",
    markers={
        "target": sim_utils.SphereCfg(
            radius=1.0,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)), # Rojo
        ),
    },
)

GREEN_ARROW_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/TargetArrow",
    markers={
        "arrow": sim_utils.ConeCfg(
            radius=0.1,
            height=0.5,
            axis='X',
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)), # VERDE
        ),
    },
)

YELLOW_ARROW_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/CommandArrow",
    markers={
        "arrow": sim_utils.ConeCfg(
            radius=0.1,
            height=0.5,
            axis='X',
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(245/255, 207/255, 39/255)), # AMARILLO
        ),
    },
)

RED_ARROW_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/DroneArrow",
    markers={
        "arrow": sim_utils.ConeCfg(
            radius=0.1,
            height=0.5,
            axis='X',
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)), # ROJO
        ),
    },
)

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for a UAV scene"""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(6000, 6000))
    )

    Joby: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Joby",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.abspath(os.path.join(root_navsim_path, "isaac_lab", "Joby", "UAM_Joby_lab.usd")),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                rigid_body_enabled=True,
                max_linear_velocity=20.0,
                max_angular_velocity=572.95779578552,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=16,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 20),
            joint_pos={
                "JRotorNW": 0,
                "JRotorNE": 0,
                "JRotorW": 0,
                "JRotorE": 0,
                "JRotorSW": 0,
                "JRotorSE": 0,
                "JBladeNW": 0,
                "JBladeNE": 0,
                "JBladeW": 0,
                "JBladeE": 0,
                "JBladeSW": 0,
                "JBladeSE": 0
            },
        ),
        #ToDo: probar los valores de los rotores en isaac sim 
        actuators={
            "JRotorNW": DCMotorCfg(
                joint_names_expr=["JRotorNW"],
                effort_limit=5000.0,
                velocity_limit=0.6, 
                stiffness=500.0,
                damping=300.0,
                saturation_effort=8000.0,
            ),
            "JRotorNE": DCMotorCfg(
                joint_names_expr=["JRotorNE"],
                effort_limit=100000.0,
                velocity_limit=100000.0,
                stiffness=0.0,
                damping=0.0,
                saturation_effort=100000.0,
            ),
            "JRotorW": DCMotorCfg(
                joint_names_expr=["JRotorW"],
                effort_limit=100000.0,
                velocity_limit=100000.0,
                stiffness=0.0,
                damping=0.0,
                saturation_effort=100000.0,
            ),
            "JRotorE": DCMotorCfg(
                joint_names_expr=["JRotorE"],
                effort_limit=100000.0,
                velocity_limit=100000.0,
                stiffness=0.0,
                damping=0.0,
                saturation_effort=100000.0,
            ),
            "JRotorSW": DCMotorCfg(
                joint_names_expr=["JRotorSW"],
                effort_limit=100000.0,
                velocity_limit=100000.0,
                stiffness=0.0,
                damping=0.0,
                saturation_effort=100000.0,
            ),
            "JRotorSE": DCMotorCfg(
                joint_names_expr=["JRotorSE"],
                effort_limit=100000.0,
                velocity_limit=100000.0,
                stiffness=0.0,
                damping=0.0,
                saturation_effort=100000.0,
            ),
            "JBladeNW": DCMotorCfg(
                joint_names_expr=["JBladeNW"],
                effort_limit=100000.0,
                velocity_limit=120.0,
                stiffness=0.0,
                damping=0.0,
                saturation_effort=100000.0,
            ),
            "JBladeNE": DCMotorCfg(
                joint_names_expr=["JBladeNE"],
                effort_limit=100000.0,
                velocity_limit=120.0,
                stiffness=0.0,
                damping=0.0,
                saturation_effort=100000.0,
            ),
            "JBladeW": DCMotorCfg(
                joint_names_expr=["JBladeW"],
                effort_limit=100000.0,
                velocity_limit=120.0,
                stiffness=0.0,
                damping=0.0,
                saturation_effort=100000.0,
            ),
            "JBladeE": DCMotorCfg(
                joint_names_expr=["JBladeE"],
                effort_limit=100000.0,
                velocity_limit=120.0,
                stiffness=0.0,
                damping=0.0,
                saturation_effort=100000.0,
            ),
            "JBladeSW": DCMotorCfg(
                joint_names_expr=["JBladeSW"],
                effort_limit=100000.0,
                velocity_limit=120.0,
                stiffness=0.0,
                damping=0.0,
                saturation_effort=100000.0,
            ),
            "JBladeSE": DCMotorCfg(
                joint_names_expr=["JBladeSE"],
                effort_limit=5000.0,
                velocity_limit=120.0,
                stiffness=0.0,
                damping=0.0,
                saturation_effort=7500.0,
            ),
        }
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg()
    )




# |---------------------------------------------------------|
# |--------------------- ENVIRONMENT -----------------------|
# |---------------------------------------------------------|

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
        # ----- Teresa -------
        # pasos que se usan en la red de PPO
        self.num_steps_per_env = 100
        # ----- Teresa -------

        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 1  # 60 Hz de actualización para la IA
        self.episode_length_s = 25.0 # al ser punto cambiante no debe ser tan largo
        # simulation settings
        self.sim.dt = 0.01666  # 60 Hz para las físicas
        self.sim.render_interval = self.decimation
        