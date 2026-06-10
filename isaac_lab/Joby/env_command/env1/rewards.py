from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from .flight_plan import FlightPlan
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def rew_attitude_stability3(env):
    asset = env.scene["AeroTaxi"]
    roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_com_quat_w)
    angle_term = torch.exp(-2.0 * (roll**2 + pitch**2))
    return angle_term

def rew_track_pos(env: ManagerBasedRLEnv):
    term = env.command_manager.get_term("vel_command")
    target_pos = term.target_pos  # XYZ
    
    asset = env.scene["AeroTaxi"]
    current_pos_w = asset.data.root_com_pos_w[:, :3]
    current_pos_local = current_pos_w - env.scene.env_origins[:, :3]

    error = torch.norm(current_pos_local - target_pos, dim=1)
    return torch.exp(-0.05 * error**2) # -0.3 antes

def rew_ang_vel_stability5(env):
    asset = env.scene["AeroTaxi"]
    ang_vel = asset.data.root_com_ang_vel_b

    # Separar yaw
    ang_vel_xy = ang_vel[:, :2]
    yaw_rate = ang_vel[:, 2]

    term_xy = torch.exp(-0.5 * torch.norm(ang_vel_xy, dim=1)**2)
    term_yaw = torch.exp(-0.3 * yaw_rate**2)

    return 0.7 * term_xy + 0.3 * term_yaw

def rew_track_vel3(env: ManagerBasedRLEnv):

    term = env.command_manager.get_term("vel_command")
    target_vel = term.target_vel[:, :3]

    asset = env.scene["AeroTaxi"]
    vel = asset.data.root_com_lin_vel_w[:, :3]

    vel_norm = torch.norm(vel, dim=1, keepdim=True) + 1e-6
    target_norm = torch.norm(target_vel, dim=1, keepdim=True) + 1e-6

    vel_dir = vel / vel_norm
    target_dir = target_vel / target_norm

    alignment = torch.sum(vel_dir * target_dir, dim=1)

    return torch.clamp(alignment, 0.0, 1.0)

def rew_action_rate(env):
    diff = -0.01 * torch.norm(env.action_manager.action - env.action_manager.prev_action, dim=1)**2
    return diff

def rew_heading(env):

    asset = env.scene["AeroTaxi"]
    vel = asset.data.root_com_lin_vel_w[:,:2]

    yaw = math_utils.euler_xyz_from_quat(asset.data.root_com_quat_w)[2]

    forward = torch.stack([torch.cos(yaw), torch.sin(yaw)],dim=1)

    vel_dir = vel / (torch.norm(vel,dim=1,keepdim=True)+1e-6)

    alignment = torch.sum(forward * vel_dir, dim=1)

    return torch.clamp(alignment,0,1)