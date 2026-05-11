import numpy as np

POSE_NEUTRAL = np.array([
     0.0,                        # waist_yaw
     0.0,  0.5,  0.0,  1.5,  0.0,  0.0,  0.5,   # left arm
     0.0, -0.5,  0.0,  1.5,  0.0,  0.0, -0.5,   # right arm
], dtype=np.float64)


POSE_CARRY = np.array([
     0.0,
     0.0,  0.4,  0.0,  1.3,  0.0,  0.0,  0.4,   # left  — slight bend, tucked
     0.0, -0.4,  0.0,  1.3,  0.0,  0.0, -0.4,   # right — mirror of left
], dtype=np.float64)


POSE_REACH_DOWN = np.array([
     0.0,
     0.0,  0.5,  0.0,  1.5,  0.0,  0.0,  0.5,   # left arm neutral
    -0.60, -0.32, 0.0,  1.65, 0.0,  0.20, -0.35,
], dtype=np.float64)

POSE_REACH_GRASP = np.array([
     0.0,
     0.0,  0.5,  0.0,  1.5,  0.0,  0.0,  0.5,
    -0.72, -0.30, 0.0,  1.90, 0.0,  0.45, -0.35,
], dtype=np.float64)

POSE_CARRY_OBJECT = np.array([
     0.0,
     0.0,  0.5,  0.0,  1.5,  0.0,  0.0,  0.5,
    -0.20, -0.34, 0.0,  1.35, 0.0,  0.0, -0.3,
], dtype=np.float64)

POSE_PLACE = np.array([
     0.08,
     0.0,  0.5,  0.0,  1.5,  0.0,  0.0,  0.5,
    -0.45, -0.18, 0.0,  1.15, 0.0,  0.10, -0.25,
], dtype=np.float64)


LEG_STAND = np.array([
    -0.20, 0.00, 0.00,  0.59, -0.34, 0.00,
    -0.20, 0.00, 0.00,  0.59, -0.34, 0.00,
], dtype=np.float64)


LEG_SQUAT = np.array([
    -0.72, 0.00, 0.00,  1.35, -0.68, 0.00,
    -0.72, 0.00, 0.00,  1.35, -0.68, 0.00,
], dtype=np.float64)

LEG_HALF_SQUAT = np.array([
    -0.50, 0.00, 0.00,  1.00, -0.52, 0.00,
    -0.50, 0.00, 0.00,  1.00, -0.52, 0.00,
], dtype=np.float64)


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between pose a and pose b, t in [0, 1]."""
    t = float(np.clip(t, 0.0, 1.0))
    return a + t * (b - a)


def smooth_t(elapsed: float, duration: float) -> float:
    """Smoothstep interpolation parameter given elapsed and total duration."""
    t = np.clip(elapsed / max(duration, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)   # smoothstep


class ArmPlanner:

    # Transition durations per phase (seconds)
    DURATIONS = {
        "STAND":           0.5,
        "WALK_TO_ITEM":    0.5,
        "REACH_DOWN":      4.0,
        "GRASP_OBJECT":    1.4,
        "LIFT_OBJECT":     2.0,
        "WALK_TO_DROP":    0.5,   # renamed from WALK_TO_TABLE
        "PLACE_OBJECT":    2.5,
        "RELEASE":         0.6,
        "RETURN_TO_STAND": 2.5,
        "DONE":            0.5,
    }

    def get_targets(self, phase_label: str, elapsed: float):
        
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
            # Shallow crouch and a small waist yaw while leaning the right arm
            # forward to the tabletop drop zone.
            arm = lerp(POSE_CARRY_OBJECT, POSE_PLACE,       t)
            leg = lerp(LEG_STAND,         LEG_HALF_SQUAT,   t)

        elif phase_label == "RELEASE":
            arm = POSE_PLACE.copy()
            leg = LEG_HALF_SQUAT.copy()

        elif phase_label == "RETURN_TO_STAND":
            arm = lerp(POSE_PLACE,      POSE_NEUTRAL, t)
            leg = lerp(LEG_HALF_SQUAT,  LEG_STAND,    t)

        else:  # DONE or unknown
            arm = POSE_NEUTRAL.copy()
            leg = LEG_STAND.copy()

        return arm, leg
