"""
Pushover template — copy this to pushover.py in the project root and implement
your own push schedule in compute_force() and pass/fail logic in evaluate().

The class name must remain `Pushover` so deploy.py can import it as-is.
See docs/pushover.md for full specification.
"""

import numpy as np


class Pushover:

        # --------- CONSTANTS ---------
    SIGMA_XY = 9.7
    SIGMA_Z = 0.8
    MAG_MIN = 5.0
    MAG_MAX = 100.0
    P_OUTLIER = 0.0044
    OUTLIER_F_MIN = 30.0
    OUTLIER_F_MAX = 100.0
    OUTLIER_FZ_STD = 3.0
    PUSH_PERIOD = 0.5
    PUSHES_PER_DAY = int(24 * 3600 / PUSH_PERIOD)
    FALL_BUDGET_PER_DAY = 1.0
    RESPAWN_GAP_SECONDS = 1.5

    def __init__(self, seed: int = 0):
        self._rng        = np.random.default_rng(seed)
        self._push_count = 0
        self._fall_count = 0
        self._last_t     = None

    def compute_force(self, t, qpos, qvel):
        """Compute external force to apply to the pelvis.

        Args:
            t: simulation time (s)
            qpos: full robot qpos (nq,)
            qvel: full robot qvel (nv,)

        Returns:
            force: np.ndarray(3,) — world-frame force [fx, fy, fz] in Newtons
        """
        if self._last_t is not None and (t - self._last_t) > self.RESPAWN_GAP_SECONDS:
            self._fall_count += 1
        self._last_t = float(t)
 
        if self._rng.random() < self.P_OUTLIER:
            theta = self._rng.uniform(0.0, 2.0 * np.pi)
            mag   = self._rng.uniform(self.OUTLIER_F_MIN, self.OUTLIER_F_MAX)
            fx, fy = mag * np.cos(theta), mag * np.sin(theta)
            fz = self._rng.normal(0.0, self.OUTLIER_FZ_STD)
        else:
            for _ in range(20):
                fx = self._rng.normal(0.0, self.SIGMA_XY)
                fy = self._rng.normal(0.0, self.SIGMA_XY)
                fz = self._rng.normal(0.0, self.SIGMA_Z)
                if np.sqrt(fx * fx + fy * fy + fz * fz) >= self.MAG_MIN:
                    break
            else:
                fx, fy, fz = self.MAG_MIN, 0.0, 0.0
 
        mag = float(np.sqrt(fx * fx + fy * fy + fz * fz))
        if mag > self.MAG_MAX:
            s = self.MAG_MAX / mag
            fx, fy, fz = fx * s, fy * s, fz * s
 
        self._push_count += 1
        return np.array([fx, fy, fz])

        # return np.random.rand(3)

    def evaluate(self):
        """Evaluate robot performance after simulation ends.

        Returns:
            bool — True if fall risk exceeds threshold, False otherwise
        """
        n, f = self._push_count, self._fall_count
        if n == 0:
            return False

        p_hat = f / n
        z = 1.96
        denom  = 1.0 + z*z / n
        center = (p_hat + z*z / (2*n)) / denom
        radius = (z / denom) * np.sqrt(p_hat*(1-p_hat)/n + z*z/(4*n*n))
        p_upper = (center + radius) * self.PUSHES_PER_DAY
        
        return bool(p_upper > self.FALL_BUDGET_PER_DAY)
        # return True