"""
nav_controller.py  —  HW Final Project
Steers the robot toward a 2-D goal by injecting small hip-pitch and
hip-yaw offsets into the standing-angle targets.

Key design decisions to prevent falling:
 - pitch_bias is very small (max 0.06 rad) so the balance controller
   is never overwhelmed
 - yaw correction is gated: we only turn when |yaw_err| > 0.15 rad
   (pure heading correction without simultaneous forward lean)
 - When yaw error is large we suppress pitch bias entirely (turn first)
"""
import numpy as np

class NavController:
    def __init__(self,
                 kp_fwd:       float = 0.04,
                 kp_yaw:       float = 0.12,
                 max_pitch:    float = 0.06,
                 max_yaw:      float = 0.10,
                 arrival_r:    float = 0.70,
                 yaw_gate:     float = 0.15):
        self.kp_fwd    = kp_fwd
        self.kp_yaw    = kp_yaw
        self.max_pitch = max_pitch
        self.max_yaw   = max_yaw
        self.arrival_r = arrival_r
        self.yaw_gate  = yaw_gate
        self._goal: np.ndarray = None

    def set_goal(self, goal_xy):
        self._goal = np.array(goal_xy[:2], dtype=np.float64)

    def compute_bias(self, qpos: np.ndarray):
        if self._goal is None:
            return 0.0, 0.0, True

        base_xy = qpos[:2]
        delta   = self._goal - base_xy
        dist    = np.linalg.norm(delta)
        if dist < self.arrival_r:
            return 0.0, 0.0, True

        w, x, y, z = qpos[3:7]
        yaw = np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
        desired_yaw = np.arctan2(delta[1], delta[0])
        yaw_err = float(((desired_yaw - yaw + np.pi) % (2*np.pi)) - np.pi)

        # Turn first when heading is off; suppress forward lean while turning
        yaw_bias = float(np.clip(self.kp_yaw * yaw_err, -self.max_yaw, self.max_yaw))
        if abs(yaw_err) > self.yaw_gate:
            pitch_bias = 0.0   # turn only, don't walk yet
        else:
            fwd = np.cos(yaw)*delta[0] + np.sin(yaw)*delta[1]
            pitch_bias = float(np.clip(self.kp_fwd * fwd, -self.max_pitch, self.max_pitch))

        return pitch_bias, yaw_bias, False

    def apply_bias(self, leg_targets: np.ndarray,
                   pitch_bias: float, yaw_bias: float) -> np.ndarray:
        """
        leg layout:
          [L_hip_pitch(0), L_hip_roll(1), L_hip_yaw(2), L_knee(3),
           L_ankle_pitch(4), L_ankle_roll(5),
           R_hip_pitch(6), R_hip_roll(7), R_hip_yaw(8), R_knee(9),
           R_ankle_pitch(10), R_ankle_roll(11)]
        """
        t = leg_targets.copy()
        # Forward lean on both hip pitches
        t[0] += pitch_bias
        t[6] += pitch_bias
        # Ankle compensation to keep foot flat
        t[4]  -= 0.5 * pitch_bias
        t[10] -= 0.5 * pitch_bias
        # Yaw: asymmetric hip yaw
        t[2] +=  yaw_bias
        t[8] -= yaw_bias
        return t