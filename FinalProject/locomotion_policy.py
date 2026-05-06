import os
from pathlib import Path

import numpy as np
import torch


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    return np.array([
        2.0 * (-qz * qx + qw * qy),
        -2.0 * (qz * qy + qw * qx),
        1.0 - 2.0 * (qw * qw + qz * qz),
    ], dtype=np.float32)


def yaw_from_quat(quaternion):
    w, x, y, z = quaternion
    return float(np.arctan2(2.0 * (w * z + x * y),
                            1.0 - 2.0 * (y * y + z * z)))


def wrap_angle(theta):
    return float((theta + np.pi) % (2.0 * np.pi) - np.pi)


class HW2LocomotionPolicy:
    """
    Thin adapter around the HW2 TorchScript walking policy.

    The HW2 policy controls the first 12 G1 leg actuators.  The FinalProject
    27-DoF model has the same first 12 leg actuator order, so this wrapper can
    provide leg torques while PickupCtrl continues to control waist/arms.
    """

    def __init__(self, policy_path=None, control_dt=0.01):
        here = Path(__file__).resolve().parent
        default_policy = here.parents[1] / "HW2" / "HW2" / "policy" / "motion.pt"
        self.policy_path = Path(policy_path or default_policy)
        if not self.policy_path.exists():
            raise FileNotFoundError(f"HW2 policy not found: {self.policy_path}")

        self.policy = torch.jit.load(str(self.policy_path), map_location="cpu")
        self.policy.eval()

        self.kps = np.array(
            [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40],
            dtype=np.float32,
        )
        self.kds = np.array(
            [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2],
            dtype=np.float32,
        )
        self.default_angles = np.array(
            [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
             -0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
            dtype=np.float32,
        )

        self.ang_vel_scale = 0.25
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.action_scale = 0.25
        self.cmd_scale = np.array([2.0, 2.0, 0.25], dtype=np.float32)
        self.num_actions = 12
        self.num_obs = 47

        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.control_dt = float(control_dt)
        self.policy_period = 0.02
        self._elapsed = 0.0
        self._last_goal = None

    def reset(self):
        self.action[:] = 0.0
        self.target_dof_pos = self.default_angles.copy()
        self.obs[:] = 0.0
        self._elapsed = 0.0
        self._last_goal = None

    def _command_to_goal(self, qpos, goal_xy):
        if goal_xy is None:
            return np.zeros(3, dtype=np.float32)

        goal_xy = np.asarray(goal_xy[:2], dtype=np.float32)
        delta_w = goal_xy - qpos[:2]
        dist = float(np.linalg.norm(delta_w))
        if dist < 0.40:
            return np.zeros(3, dtype=np.float32)

        yaw = yaw_from_quat(qpos[3:7])
        cy, sy = np.cos(yaw), np.sin(yaw)
        local_x = cy * delta_w[0] + sy * delta_w[1]
        local_y = -sy * delta_w[0] + cy * delta_w[1]
        desired_yaw = float(np.arctan2(delta_w[1], delta_w[0]))
        yaw_err = wrap_angle(desired_yaw - yaw)

        vx = np.clip(0.28 * local_x, -0.08, 0.18)
        vy = 0.0
        yaw_rate = np.clip(0.9 * yaw_err, -0.65, 0.65)

        if abs(yaw_err) > 0.75:
            vx = 0.0
        elif abs(yaw_err) > 0.35:
            vx *= 0.45

        return np.array([vx, vy, yaw_rate], dtype=np.float32)

    def compute_leg_torque(self, qpos, qvel, goal_xy):
        self._elapsed += self.control_dt

        if self._last_goal is None or np.linalg.norm(np.asarray(goal_xy) - self._last_goal) > 1e-6:
            self._last_goal = np.asarray(goal_xy, dtype=np.float32).copy()

        if self._elapsed == self.control_dt or self._elapsed % self.policy_period < self.control_dt:
            qj = (qpos[7:19] - self.default_angles) * self.dof_pos_scale
            dqj = qvel[6:18] * self.dof_vel_scale
            gravity_orientation = get_gravity_orientation(qpos[3:7])
            omega = qvel[3:6] * self.ang_vel_scale

            period = 0.8
            phase = (self._elapsed % period) / period
            sin_phase = np.sin(2.0 * np.pi * phase)
            cos_phase = np.cos(2.0 * np.pi * phase)

            cmd = self._command_to_goal(qpos, goal_xy)
            self.obs[:3] = omega
            self.obs[3:6] = gravity_orientation
            self.obs[6:9] = cmd * self.cmd_scale
            self.obs[9:21] = qj
            self.obs[21:33] = dqj
            self.obs[33:45] = self.action
            self.obs[45:47] = np.array([sin_phase, cos_phase], dtype=np.float32)

            with torch.no_grad():
                action = self.policy(torch.from_numpy(self.obs).unsqueeze(0))
            self.action = action.detach().cpu().numpy().squeeze().astype(np.float32)
            self.target_dof_pos = self.action * self.action_scale + self.default_angles

        q = qpos[7:19]
        dq = qvel[6:18]
        tau = (self.target_dof_pos - q) * self.kps - dq * self.kds
        return np.clip(tau, -80.0, 80.0)
