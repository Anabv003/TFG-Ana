
"""Rest everything follows."""

import math
import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils


# |---------------------------------------------------------|
# |--------------------- OBSERVATIONS ----------------------|
# |---------------------------------------------------------|

def my_obs_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_com_pos_w - env.scene.env_origins # posición relativa

def my_obs_projected_gravity(env:ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b

# ----- Teresa -------
def my_obs_dist(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term("vel_command")
    
    uav_pos_local = asset.data.root_com_pos_w - env.scene.env_origins
    relative_pos = command_term.target_pos - uav_pos_local[:, :3] # que coja no dónde está con respecto al centro, si no
    # a cuánto está del punto, si no sobreajusta

    # que el dron conozca la orientación a la que está ese punto
    invertir_z = math_utils.quat_inv(asset.data.root_com_quat_w)
    rel_pos_b = math_utils.quat_apply(invertir_z, relative_pos)

    return rel_pos_b # es [dist_x,dist_y,dist_z,z_dron_relativa]

def my_obs_dist2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term("vel_command")
    
    uav_pos_local = asset.data.root_com_pos_w - env.scene.env_origins
    rel_pos_b = command_term.target_pos - uav_pos_local[:, :3] # que coja no dónde está con respecto al centro, si no
    # a cuánto está del punto, si no sobreajusta

    # que el dron conozca la orientación a la que está ese punto
    # invertir_z = math_utils.quat_inv(asset.data.root_com_quat_w)
    # rel_pos_b = math_utils.quat_apply(invertir_z, rel_pos_b)

    return rel_pos_b # es [dist_x,dist_y,dist_z,z_dron_relativa]
# ------------------------------
def my_obs_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    uav_pos_local = asset.data.root_com_pos_w - env.scene.env_origins
    altura_z = uav_pos_local[:, 2:3]
    return altura_z

def my_obs_get_command(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    command_term = env.command_manager.get_term("vel_command") 
    return command_term._command 

def my_obs_lin_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_com_lin_vel_b

def my_obs_ang_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_com_ang_vel_b[:,2]

def my_obs_roll(env:ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    roll, _, _ = math_utils.euler_xyz_from_quat(asset.data.root_com_quat_w)
    roll = torch.atan2(torch.sin(roll), torch.cos(roll)) # normalize angle to [-pi, pi]
    roll = roll.unsqueeze(1)  # Add a dimension to match the expected shape

    return roll

def my_obs_pitch(env:ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    _, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_com_quat_w)
    pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch)) # normalize angle to [-pi, pi]
    pitch = pitch.unsqueeze(1)  # Add a dimension to match the expected shape

    return pitch

def my_obs_target_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset = env.scene["Joby"]
    command_term = env.command_manager.get_term("vel_command")
    target_vel_w = command_term.target_vel 
    quat_inv = math_utils.quat_inv(asset.data.root_com_quat_w)
    target_vel_b = math_utils.quat_apply(quat_inv, target_vel_w)
    
    return target_vel_b

def my_obs_yaw(env:ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_com_quat_w)
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw)) # normalize angle to [-pi, pi]
    yaw = yaw.unsqueeze(1)  # Add a dimension to match the expected shape

    return yaw

def my_obs_target_yaw_rate(env:ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    term = env.command_manager.get_term("vel_command")
    yaw_rate = term.target_yaw
    return yaw_rate.unsqueeze(1) / 0.1

def my_obs_yaw_error(env:ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_com_quat_w)

    # velocidad objetivo del punto guía
    term = env.command_manager.get_term("vel_command")
    target_vel = term.target_vel[:, :2]

    # yaw deseado (dirección de movimiento del target)
    target_yaw = torch.atan2(target_vel[:, 1], target_vel[:, 0])

    # error
    yaw_error = target_yaw - yaw

    # normalizar a [-pi, pi]
    yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi

    return yaw_error.unsqueeze(1) / math.pi

def my_obs_target_ang_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    """Velocidad angular deseada en el frame del dron (body frame)."""
    asset = env.scene["Joby"]
    command_term = env.command_manager.get_term("vel_command")
    zeros = torch.zeros_like(command_term.target_pos) # [N, 3]
    zeros[:, 2] = command_term.target_yaw # Ponemos el comando en la componente Z
    
    # 2. Rotamos ese vector al body frame
    quat_inv = math_utils.quat_inv(asset.data.root_com_quat_w)
    target_ang_vel_b = math_utils.quat_apply(quat_inv, zeros)
    
    return target_ang_vel_b


def my_obs_target_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    """Vector hacia el objetivo relativo al dron, sin dar la posición absoluta de entrenamiento."""
    asset = env.scene["Joby"]
    command_term = env.command_manager.get_term("vel_command")
    target_pos_w = command_term.target_pos
    current_pos_w = asset.data.root_com_pos_w[:, :3]
    # Relativo al dron, no al mundo
    target_rel = target_pos_w - current_pos_w
    # Opcional: proyectar al frame del dron si quieres coherencia con velocities
    quat_inv = math_utils.quat_inv(asset.data.root_com_quat_w)
    target_rel_b = math_utils.quat_apply(quat_inv, target_rel)
    return target_rel_b


def my_obs_prev_action(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    """Retorna la acción previa aplicada, útil para rew_action_rate."""
    return env.action_manager.prev_action

def my_obs_target_future(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):

    cmd = env.command_manager.get_term("vel_command")
    # Ya en _update_command se calcula self._command[:, :3] como vector body frame
    return cmd._command[:, :3]



def my_obs_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get current velocity commands."""
    return env.command_manager.get_command("vel_command")
