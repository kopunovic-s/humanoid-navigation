"""
nav_controller.py  —  HW Final Project
========================================
Simple proportional-derivative navigation for the G1 humanoid base.

The robot has no locomotion policy, so we implement approximate base
steering by modulating the hip and ankle joints to lean the robot toward
the goal.  A proper locomotion policy is out of scope; here we use:

  - A planar velocity command computed from the error to the goal XY
  - The command is translated into a small lean offset on the standing
    angles (hip pitch bias) to achieve slow forward/lateral motion
  - Yaw error is corrected by hip yaw asymmetry

This is intentionally simple and compatible with the HW3 StandingCtrl
architecture.  The StandingCtrl continues to maintain balance while we
inject a hip-pitch bias.
"""

import numpy as np


class NavController:
    """
    Computes a (hip_pitch_bias, hip_yaw_bias) offset to steer the standing
    robot toward a 2-D waypoint.

    Parameters
    ----------
    kp_pos : float   Proportional gain for positional error
    kp_yaw : float   Proportional gain for heading error
    max_pitch_bias : float  Maximum |hip pitch offset| in radians
    max_yaw_bias   : float  Maximum |hip yaw offset| in radians
    arrival_radius : float  Distance (m) at which we declare arrival
    """

    def __init__(self,
                 kp_pos: float = 0.18,
                 kp_yaw: float = 0.25,
                 max_pitch_bias: float = 0.12,
                 max_yaw_bias: float   = 0.15,
                 arrival_radius: float = 0.30):
        self.kp_pos = kp_pos
        self.kp_yaw = kp_yaw
        self.max_pitch_bias = max_pitch_bias
        self.max_yaw_bias   = max_yaw_bias
        self.arrival_radius = arrival_radius

        self._goal_xy: np.ndarray = None
        self._integral = np.zeros(2)
        self.ki_pos = 0.01

    # ---------------------------------------------------------------
    def set_goal(self, goal_xy: np.ndarray):
        """Set a new 2-D navigation goal (world frame)."""
        self._goal_xy  = np.array(goal_xy[:2], dtype=np.float64)
        self._integral = np.zeros(2)

    # ---------------------------------------------------------------
    def compute_bias(self, qpos: np.ndarray, dt: float = 0.02):
        """
        Returns (pitch_bias, yaw_bias, arrived).

        pitch_bias : float  — added equally to both hip_pitch joints
                              (lean forward to walk forward)
        yaw_bias   : float  — +/- on left/right hip yaw to turn
        arrived    : bool   — True if within arrival_radius of goal
        """
        if self._goal_xy is None:
            return 0.0, 0.0, True

        base_xy  = qpos[:2]
        base_quat = qpos[3:7]   # w, x, y, z

        # Yaw from quaternion
        w, x, y, z = base_quat
        yaw = np.arctan2(2.0 * (w * z + x * y),
                         1.0 - 2.0 * (y * y + z * z))

        # Vector to goal in world frame
        delta_world = self._goal_xy - base_xy
        dist        = np.linalg.norm(delta_world)

        if dist < self.arrival_radius:
            return 0.0, 0.0, True

        # Desired heading
        desired_yaw = np.arctan2(delta_world[1], delta_world[0])
        yaw_err     = self._wrap_angle(desired_yaw - yaw)

        # Forward component (project delta onto heading)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        fwd_err   = cos_y * delta_world[0] + sin_y * delta_world[1]
        lat_err   = -sin_y * delta_world[0] + cos_y * delta_world[1]

        # Integral accumulation (mild)
        self._integral[0] += fwd_err * dt
        self._integral[1] += yaw_err * dt
        self._integral = np.clip(self._integral, -2.0, 2.0)

        pitch_bias = float(np.clip(
            self.kp_pos * fwd_err + self.ki_pos * self._integral[0],
            -self.max_pitch_bias, self.max_pitch_bias
        ))
        yaw_bias = float(np.clip(
            self.kp_yaw * yaw_err + self.ki_pos * self._integral[1],
            -self.max_yaw_bias, self.max_yaw_bias
        ))

        return pitch_bias, yaw_bias, False

    # ---------------------------------------------------------------
    def apply_bias_to_leg_targets(self,
                                  leg_targets: np.ndarray,
                                  pitch_bias: float,
                                  yaw_bias: float) -> np.ndarray:
        """
        Inject nav bias into the 12-element leg joint target array.

        leg_targets layout:
          [L_hip_pitch(0), L_hip_roll(1), L_hip_yaw(2), L_knee(3),
           L_ankle_pitch(4), L_ankle_roll(5),
           R_hip_pitch(6), R_hip_roll(7), R_hip_yaw(8), R_knee(9),
           R_ankle_pitch(10), R_ankle_roll(11)]
        """
        tgt = leg_targets.copy()

        # Apply forward lean via hip pitch offset
        tgt[0] += pitch_bias   # L hip pitch
        tgt[6] += pitch_bias   # R hip pitch

        # Compensate ankle pitch to keep foot flat
        ankle_comp = -0.6 * pitch_bias
        tgt[4] += ankle_comp
        tgt[10] += ankle_comp

        # Apply turning via hip yaw asymmetry
        tgt[2] +=  yaw_bias    # L hip yaw
        tgt[8] += -yaw_bias    # R hip yaw

        return tgt

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return float((a + np.pi) % (2 * np.pi) - np.pi)
