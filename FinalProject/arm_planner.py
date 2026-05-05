"""
arm_planner.py  —  HW Final Project
=====================================
Analytical arm joint-angle targets for each task phase on the G1 27-DoF humanoid.

The G1 right arm joints (in config order after the 12 leg joints + 1 waist):
  Index  Joint name
  -----  ----------
  0      waist_yaw
  1      left_shoulder_pitch
  2      left_shoulder_roll
  3      left_shoulder_yaw
  4      left_elbow
  5      left_wrist_roll
  6      left_wrist_pitch
  7      left_wrist_yaw
  8      right_shoulder_pitch
  9      right_shoulder_roll
  10     right_shoulder_yaw
  11     right_elbow
  12     right_wrist_roll
  13     right_wrist_pitch
  14     right_wrist_yaw

These indices match the arm_waist_target vector in g1.yaml (15 elements).

Poses are expressed as joint-angle arrays matching arm_waist_target length (15).
"""

import numpy as np


# ---------------------------------------------------------------------------
#  Named arm/waist pose library
#  All angles in radians.
#  Layout: [waist_yaw,
#            L_shoulder_pitch, L_shoulder_roll, L_shoulder_yaw, L_elbow,
#            L_wrist_roll, L_wrist_pitch, L_wrist_yaw,
#            R_shoulder_pitch, R_shoulder_roll, R_shoulder_yaw, R_elbow,
#            R_wrist_roll, R_wrist_pitch, R_wrist_yaw]
# ---------------------------------------------------------------------------

# Default neutral arms-at-sides (from g1.yaml arm_waist_target)
POSE_NEUTRAL = np.array([
     0.0,                        # waist_yaw
     0.0,  0.5,  0.0,  1.5,  0.0,  0.0,  0.5,   # left arm
     0.0, -0.5,  0.0,  1.5,  0.0,  0.0, -0.5,   # right arm
], dtype=np.float64)

# Arms relaxed at sides — used while walking to item / table
# (kept symmetric with POSE_NEUTRAL so the transition out of STAND does
#  not visibly swing the right arm forward)
POSE_CARRY = np.array([
     0.0,
     0.0,  0.4,  0.0,  1.3,  0.0,  0.0,  0.4,   # left  — slight bend, tucked
     0.0, -0.4,  0.0,  1.3,  0.0,  0.0, -0.4,   # right — mirror of left
], dtype=np.float64)

# Right arm reaching down and forward to floor level
# shoulder pitch ~1.6 rad (forward), elbow bent ~1.8 rad (reaching down)
POSE_REACH_DOWN = np.array([
     0.0,
     0.0,  0.5,  0.0,  1.5,  0.0,  0.0,  0.5,   # left arm neutral
     1.55, -0.25, 0.0,  1.80, 0.0,  0.0, -0.4,   # right arm extended down
], dtype=np.float64)

# Slight variation of reach-down — wrist pitched down to approach cube
POSE_REACH_GRASP = np.array([
     0.0,
     0.0,  0.5,  0.0,  1.5,  0.0,  0.0,  0.5,
     1.60, -0.20, 0.0,  1.85, 0.0,  0.35, -0.4,
], dtype=np.float64)

# Right arm lifted to carry height (waist level, object in hand)
POSE_CARRY_OBJECT = np.array([
     0.0,
     0.0,  0.5,  0.0,  1.5,  0.0,  0.0,  0.5,
     0.80, -0.30, 0.0,  1.40, 0.0,  0.0, -0.3,
], dtype=np.float64)

# Right arm extended forward to place object on table
POSE_PLACE = np.array([
     0.0,
     0.0,  0.5,  0.0,  1.5,  0.0,  0.0,  0.5,
     1.20, -0.20, 0.0,  1.20, 0.0,  0.20, -0.3,
], dtype=np.float64)


# ---------------------------------------------------------------------------
#  Leg/squat poses for the 12 leg joints
#  Layout matches HW3 standing_angles:
#  [L_hip_pitch, L_hip_roll, L_hip_yaw, L_knee, L_ankle_pitch, L_ankle_roll,
#   R_hip_pitch, R_hip_roll, R_hip_yaw, R_knee, R_ankle_pitch, R_ankle_roll]
# ---------------------------------------------------------------------------

# Normal standing pose (from g1.yaml)
LEG_STAND = np.array([
    -0.20, 0.00, 0.00,  0.59, -0.34, 0.00,
    -0.20, 0.00, 0.00,  0.59, -0.34, 0.00,
], dtype=np.float64)

# Deeper squat — bend knees and hips to lower CoM ~0.25m for pickup
LEG_SQUAT = np.array([
    -0.55, 0.00, 0.00,  1.10, -0.55, 0.00,
    -0.55, 0.00, 0.00,  1.10, -0.55, 0.00,
], dtype=np.float64)

# Shallow squat — mid-way between stand and full squat
LEG_HALF_SQUAT = np.array([
    -0.38, 0.00, 0.00,  0.80, -0.44, 0.00,
    -0.38, 0.00, 0.00,  0.80, -0.44, 0.00,
], dtype=np.float64)


# ---------------------------------------------------------------------------
#  Interpolation utilities
# ---------------------------------------------------------------------------

def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between pose a and pose b, t in [0, 1]."""
    t = float(np.clip(t, 0.0, 1.0))
    return a + t * (b - a)


def smooth_t(elapsed: float, duration: float) -> float:
    """Smoothstep interpolation parameter given elapsed and total duration."""
    t = np.clip(elapsed / max(duration, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)   # smoothstep


class ArmPlanner:
    """
    Returns target arm_waist joint angles and leg joint angles for each
    task phase, as a function of time elapsed in that phase.

    Usage::

        planner = ArmPlanner()
        arm_target, leg_target = planner.get_targets(phase, elapsed)
    """

    # Transition durations per phase (seconds)
    DURATIONS = {
        "STAND":           0.5,
        "WALK_TO_ITEM":    0.5,
        "REACH_DOWN":      2.5,
        "GRASP_OBJECT":    0.6,
        "LIFT_OBJECT":     2.0,
        "WALK_TO_DROP":    0.5,   # renamed from WALK_TO_TABLE
        "PLACE_OBJECT":    2.5,
        "RELEASE":         0.6,
        "RETURN_TO_STAND": 2.5,
        "DONE":            0.5,
    }

    def get_targets(self, phase_label: str, elapsed: float):
        """
        Returns (arm_target, leg_target) numpy arrays for the given phase.

        Parameters
        ----------
        phase_label : str
            Name of the current phase (from TaskStateManager.label).
        elapsed : float
            Seconds elapsed since entering this phase.

        Returns
        -------
        arm_target : np.ndarray shape (15,)
        leg_target : np.ndarray shape (12,)
        """
        dur = self.DURATIONS.get(phase_label, 1.0)
        t   = smooth_t(elapsed, dur)

        if phase_label == "STAND":
            arm = lerp(POSE_NEUTRAL, POSE_NEUTRAL, t)
            leg = lerp(LEG_STAND,   LEG_STAND,    t)

        elif phase_label == "WALK_TO_ITEM":
            arm = lerp(POSE_NEUTRAL, POSE_CARRY,  t)
            leg = lerp(LEG_STAND,   LEG_STAND,    t)

        elif phase_label == "REACH_DOWN":
            # First half: half-squat; second half: full squat + arm reaches
            if elapsed < dur * 0.5:
                t2  = smooth_t(elapsed, dur * 0.5)
                arm = lerp(POSE_CARRY,      POSE_REACH_DOWN,  t2)
                leg = lerp(LEG_STAND,       LEG_HALF_SQUAT,   t2)
            else:
                t2  = smooth_t(elapsed - dur * 0.5, dur * 0.5)
                arm = lerp(POSE_REACH_DOWN, POSE_REACH_GRASP, t2)
                leg = lerp(LEG_HALF_SQUAT,  LEG_SQUAT,        t2)

        elif phase_label == "GRASP_OBJECT":
            arm = POSE_REACH_GRASP.copy()
            leg = LEG_SQUAT.copy()

        elif phase_label == "LIFT_OBJECT":
            # Rise while moving arm to carry height
            arm = lerp(POSE_REACH_GRASP, POSE_CARRY_OBJECT, t)
            leg = lerp(LEG_SQUAT,        LEG_STAND,          t)

        elif phase_label == "WALK_TO_DROP":
            arm = POSE_CARRY_OBJECT.copy()
            leg = LEG_STAND.copy()

        elif phase_label == "PLACE_OBJECT":
            # Lower arm back to floor level to place cube on ground
            arm = lerp(POSE_CARRY_OBJECT, POSE_REACH_GRASP, t)
            leg = lerp(LEG_STAND,         LEG_SQUAT,        t)

        elif phase_label == "RELEASE":
            arm = POSE_REACH_GRASP.copy()
            leg = LEG_SQUAT.copy()

        elif phase_label == "RETURN_TO_STAND":
            arm = lerp(POSE_REACH_GRASP, POSE_NEUTRAL, t)
            leg = lerp(LEG_SQUAT,        LEG_STAND,    t)

        else:  # DONE or unknown
            arm = POSE_NEUTRAL.copy()
            leg = LEG_STAND.copy()

        return arm, leg