"""
pickup_ctrl.py  —  HW Final Project
=====================================
G1 Humanoid Pick-and-Place Controller

Integrates:
  - HW3 StandingCtrl  (Lagrangian dynamics + QP-based balance)
  - HW3 Pushover      (disturbance robustness)
  - HW1-style state machine via TaskStateManager
  - ArmPlanner        (pose library + smooth interpolation per phase)
  - NavController     (base steering toward waypoints)
  - MuJoCo equality-constraint weld for object grasping

Scene layout (world frame, robot spawns at origin):
  - Item 1 (red cube):   [1.5,  0.4, 0.035]
  - Item 2 (blue cube):  [1.5, -0.4, 0.035]
  - Table top:           [3.2,  0.0, 0.725]
  - Drop-zone 1 target:  [3.35, 0.15, 0.725]
  - Drop-zone 2 target:  [3.35,-0.15, 0.725]

Usage::
    ctrl = PickupCtrl("configs/g1_pickup.yaml")
    simulate(ctrl, duration=120)
"""

import mujoco
import numpy as np
import yaml
import os
from typing import Tuple, Dict, Optional

from standing_ctrl import StandingCtrl
from task_state    import TaskStateManager, Phase
from arm_planner   import ArmPlanner
from nav_controller import NavController


# ---------------------------------------------------------------------------
#  World-frame positions (must match scene_pickup.xml)
# ---------------------------------------------------------------------------

ITEM_POSITIONS = [
    np.array([1.5,  0.4, 0.035]),   # item 1 — red cube
    np.array([1.5, -0.4, 0.035]),   # item 2 — blue cube
]

TABLE_TOP_Z    = 0.725              # table surface height
TABLE_XY       = np.array([3.2, 0.0])

DROPZONE_XY    = [
    np.array([3.35,  0.15]),        # drop-zone for item 1
    np.array([3.35, -0.15]),        # drop-zone for item 2
]

# How close the robot base needs to be before transitioning from WALK_TO_*
PICKUP_APPROACH_DIST = 0.55         # metres from item
TABLE_APPROACH_DIST  = 0.60         # metres from table edge


# ---------------------------------------------------------------------------
class PickupCtrl:
    """
    Full pick-and-place controller for the G1 27-DoF humanoid.

    Implements the controller protocol expected by deploy.py:
      simulation_dt, control_decimation, num_actuators
      get_xml_path(), get_duration(), get_initial_state()
      compute_torque(qpos, qvel) -> (tau, info)
      reset()
    """

    # ------------------------------------------------------------------
    def __init__(self, config_path: str):
        self.config_path = config_path
        config_abs = os.path.abspath(config_path)
        self.config_dir = os.path.dirname(config_abs)

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # ---- Standing controller (HW3 base) ----
        self.standing = StandingCtrl(config_path)

        # Expose protocol attributes
        self.simulation_dt      = self.standing.simulation_dt
        self.control_decimation = self.standing.control_decimation
        self.control_dt         = self.standing.control_dt
        self.num_joints         = self.standing.num_joints
        self.num_arm_joints     = self.standing.num_arm_joints
        self.num_actuators      = self.standing.num_actuators

        # ---- MuJoCo model (for object/body lookups) ----
        xml_path = self.get_xml_path()
        self.sim_model = mujoco.MjModel.from_xml_path(xml_path)
        self.sim_data  = mujoco.MjData(self.sim_model)

        self._resolve_body_ids()
        self._resolve_eq_ids()

        # ---- Sub-systems ----
        self.state_mgr = TaskStateManager(num_items=len(ITEM_POSITIONS))
        self.arm_planner = ArmPlanner()
        self.nav_ctrl    = NavController()

        # ---- Internal state ----
        self._sim_time        = 0.0
        self._started         = False
        self._grasp_active    = [False, False]   # per-item weld state
        self._last_phase_log  = ""

        # Arm target (15,) — override the StandingCtrl arm_waist_target
        self._arm_target = self.standing.arm_waist_target.copy()
        # Leg target override (12,)
        self._leg_override: Optional[np.ndarray] = None

        print(f"[PickupCtrl] Initialized")
        print(f"  XML:         {xml_path}")
        print(f"  num_actuators: {self.num_actuators}")
        print(f"  Items: {len(ITEM_POSITIONS)}")

    # ------------------------------------------------------------------
    # Controller protocol
    # ------------------------------------------------------------------

    def get_xml_path(self) -> str:
        scene = self.config.get('pickup_scene',
                                '{DIR}/resources/g1/scene_pickup.xml')
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(self.config_path)))
        return scene.replace('{DIR}', base_dir)

    def get_duration(self) -> float:
        return float(self.config.get('simulation_duration', 180))

    def get_initial_state(self) -> dict:
        """Spawn robot at origin in standing pose."""
        state = self.standing.get_initial_state()
        # Place robot at world origin (items are in front)
        state['qpos'][0] = 0.0
        state['qpos'][1] = 0.0
        return state

    # ------------------------------------------------------------------
    def compute_torque(self,
                       qpos: np.ndarray,
                       qvel: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Main control loop — called at control_dt frequency.

        1. Update sim time and initialize state machine on first call.
        2. Determine current phase via state machine.
        3. Compute arm + leg targets via ArmPlanner.
        4. Apply nav bias to leg targets if in a walking phase.
        5. Override StandingCtrl's arm/standing targets.
        6. Call StandingCtrl.compute_torque() for full-body torque.
        7. Handle object weld attachment / detachment.
        8. Evaluate phase-advance conditions.
        """
        # ---- (1) Time tracking ----
        self._sim_time += self.control_dt
        t = self._sim_time

        if not self._started:
            self.state_mgr.start(t)
            self._started = True

        phase   = self.state_mgr.phase
        elapsed = self.state_mgr.phase_elapsed(t)
        item_i  = self.state_mgr.item_index

        # ---- (2) Arm + leg targets from planner ----
        arm_target, leg_target = self.arm_planner.get_targets(
            self.state_mgr.label, elapsed)

        # ---- (3) Navigation bias ----
        if phase == Phase.WALK_TO_ITEM and item_i < len(ITEM_POSITIONS):
            goal_xy = ITEM_POSITIONS[item_i][:2]
            self.nav_ctrl.set_goal(goal_xy)
        elif phase == Phase.WALK_TO_TABLE:
            goal_xy = TABLE_XY
            self.nav_ctrl.set_goal(goal_xy)

        pitch_bias, yaw_bias, arrived = self.nav_ctrl.compute_bias(
            qpos, dt=self.control_dt)
        if phase in (Phase.WALK_TO_ITEM, Phase.WALK_TO_TABLE):
            leg_target = self.nav_ctrl.apply_bias_to_leg_targets(
                leg_target, pitch_bias, yaw_bias)

        # ---- (4) Override StandingCtrl internal targets ----
        self.standing.arm_waist_target = arm_target
        self.standing.standing_angles  = leg_target.astype(np.float32)

        # ---- (5) Compute torques from HW3 StandingCtrl ----
        tau, info = self.standing.compute_torque(qpos, qvel)
        info['phase']      = self.state_mgr.label
        info['item_index'] = item_i

        # ---- (6) Object weld management ----
        self._handle_grasping(phase, item_i, qpos)

        # ---- (7) Phase-advance logic ----
        self._check_advance(phase, elapsed, arrived, qpos, item_i, t)

        return tau, info

    # ------------------------------------------------------------------
    def reset(self):
        """Reset all sub-systems (called on fall detection)."""
        self.standing.reset()
        self.state_mgr = TaskStateManager(num_items=len(ITEM_POSITIONS))
        self._started      = False
        self._sim_time     = 0.0
        self._grasp_active = [False, False]
        # Disable all welds on reset
        self._set_weld("grasp_item1", False)
        self._set_weld("grasp_item2", False)
        print("[PickupCtrl] Reset")

    # ------------------------------------------------------------------
    #  Private helpers
    # ------------------------------------------------------------------

    def _resolve_body_ids(self):
        """Pre-compute body IDs for items, table, and wrist."""
        def bid(name):
            i = mujoco.mj_name2id(self.sim_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if i < 0:
                print(f"[Warning] Body '{name}' not found")
            return i

        self.body_item1       = bid("item1")
        self.body_item2       = bid("item2")
        self.body_wrist_right = bid("right_wrist_yaw_link")

    def _resolve_eq_ids(self):
        """Pre-compute equality constraint IDs."""
        def eid(name):
            i = mujoco.mj_name2id(self.sim_model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
            if i < 0:
                print(f"[Warning] Equality constraint '{name}' not found in model")
            return i

        self.eq_grasp1 = eid("grasp_item1")
        self.eq_grasp2 = eid("grasp_item2")

    def _set_weld(self, name: str, active: bool):
        """Enable or disable a named equality constraint weld."""
        eq_id = mujoco.mj_name2id(
            self.sim_model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if eq_id >= 0:
            self.sim_model.eq_active0[eq_id] = int(active)

    def _item_body_id(self, item_i: int) -> int:
        return self.body_item1 if item_i == 0 else self.body_item2

    def _handle_grasping(self, phase: Phase, item_i: int, qpos: np.ndarray):
        """Enable/disable weld constraints based on phase."""
        if item_i >= len(ITEM_POSITIONS):
            return

        weld_name = f"grasp_item{item_i + 1}"

        if phase == Phase.GRASP_OBJECT and not self._grasp_active[item_i]:
            self._set_weld(weld_name, True)
            self._grasp_active[item_i] = True
            print(f"[{self._sim_time:.2f}s] GRASPED item {item_i + 1}")

        elif phase == Phase.RELEASE and self._grasp_active[item_i]:
            self._set_weld(weld_name, False)
            self._grasp_active[item_i] = False
            print(f"[{self._sim_time:.2f}s] RELEASED item {item_i + 1}")

    def _check_advance(self, phase: Phase, elapsed: float,
                       nav_arrived: bool, qpos: np.ndarray,
                       item_i: int, t: float):
        """Check conditions for advancing to next phase."""
        if not self.state_mgr.min_dwell_met(t):
            return

        base_xy = qpos[:2]

        if phase == Phase.STAND:
            # Advance after dwell time — robot has settled
            self.state_mgr.advance(t)
            if item_i < len(ITEM_POSITIONS):
                goal = ITEM_POSITIONS[item_i][:2]
                self.nav_ctrl.set_goal(goal)

        elif phase == Phase.WALK_TO_ITEM:
            if item_i < len(ITEM_POSITIONS):
                item_xy = ITEM_POSITIONS[item_i][:2]
                dist = np.linalg.norm(base_xy - item_xy)
                if dist < PICKUP_APPROACH_DIST or nav_arrived:
                    self.state_mgr.advance(t)

        elif phase == Phase.REACH_DOWN:
            self.state_mgr.advance(t)

        elif phase == Phase.GRASP_OBJECT:
            self.state_mgr.advance(t)

        elif phase == Phase.LIFT_OBJECT:
            self.state_mgr.advance(t)
            self.nav_ctrl.set_goal(TABLE_XY)

        elif phase == Phase.WALK_TO_TABLE:
            dist = np.linalg.norm(base_xy - TABLE_XY)
            if dist < TABLE_APPROACH_DIST or nav_arrived:
                self.state_mgr.advance(t)

        elif phase == Phase.PLACE_OBJECT:
            self.state_mgr.advance(t)

        elif phase == Phase.RELEASE:
            self.state_mgr.advance(t)

        elif phase == Phase.RETURN_TO_STAND:
            self.state_mgr.advance(t)
            # Set up next item navigation if applicable
            next_i = self.state_mgr.item_index
            if next_i < len(ITEM_POSITIONS):
                self.nav_ctrl.set_goal(ITEM_POSITIONS[next_i][:2])