"""
pickup_ctrl.py  —  HW Final Project
Full pick-and-place controller for the G1 27-DoF humanoid.

Strategy
--------
HW3 StandingCtrl (Lagrangian dynamics + QP contact-force balance + CoM
PID + ankle moment) is used in every phase.  The PickupCtrl overrides
StandingCtrl.standing_angles / arm_waist_target each control step so
the same QP solver continues to balance the robot while the *target
pose* changes from phase to phase.

Walking is implemented as a small lean: NavController computes a hip
pitch + hip yaw bias from the (goal - base_xy) error and we add it to
the standing leg targets.  The QP rebalances around the leaned target
and the robot drifts slowly toward the goal.
"""
import os
import yaml
import mujoco
import numpy as np
from typing import Tuple, Dict

from standing_ctrl  import StandingCtrl
from task_state     import TaskStateManager, Phase
from arm_planner    import ArmPlanner
from nav_controller import NavController

# ---------------------------------------------------------------------------
# World positions  (must match scene_pickup.xml)
# ---------------------------------------------------------------------------
ITEM_POSITIONS = [
    np.array([3.0,  0.0, 0.05]),   # item 1 — red cube, 3 m in front
    np.array([0.0,  5.0, 0.05]),   # item 2 — blue cube, 5 m to the left (+Y)
]
# Drop both items on the table top (z is irrelevant — only XY is used for
# the proximity check; the arm planner brings the wrist down to surface).
DROP_POSITIONS = [
    np.array([6.35,  0.15]),
    np.array([6.35, -0.15]),
]
PICKUP_DIST = 0.90      # m — switch from WALK_TO_ITEM to REACH_DOWN
DROP_DIST   = 0.70      # m — switch from WALK_TO_DROP to PLACE_OBJECT


class PickupCtrl:

    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.standing = StandingCtrl(config_path)

        self.simulation_dt      = self.standing.simulation_dt
        self.control_decimation = self.standing.control_decimation
        self.control_dt         = self.standing.control_dt
        self.num_joints         = self.standing.num_joints
        self.num_arm_joints     = self.standing.num_arm_joints
        self.num_actuators      = self.standing.num_actuators

        # Build path to scene XML for the viewer
        config_abs = os.path.abspath(config_path)
        base_dir   = os.path.dirname(os.path.dirname(config_abs))
        scene_key  = self.config.get('pickup_scene', '{DIR}/resources/g1/scene_pickup.xml')
        self._xml_path = scene_key.replace('{DIR}', base_dir)

        # Load scene model for body / equality lookups
        self.sim_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self.sim_data  = mujoco.MjData(self.sim_model)
        self._resolve_ids()

        self.state_mgr   = TaskStateManager(num_items=len(ITEM_POSITIONS))
        self.arm_planner = ArmPlanner()
        self.nav         = NavController()

        self._sim_time = 0.0
        self._started  = False
        self._grasped  = [False, False]

        print(f"[PickupCtrl] Initialized")
        print(f"  XML:           {self._xml_path}")
        print(f"  num_actuators: {self.num_actuators}")

    # ------------------------------------------------------------------
    def get_xml_path(self) -> str:  return self._xml_path
    def get_duration(self) -> float: return float(self.config.get('simulation_duration', 300))
    def get_initial_state(self) -> dict: return self.standing.get_initial_state()

    # ------------------------------------------------------------------
    def compute_torque(self, qpos: np.ndarray, qvel: np.ndarray) -> Tuple[np.ndarray, Dict]:
        self._sim_time += self.control_dt
        t = self._sim_time

        if not self._started:
            self.state_mgr.start(t)
            self._started = True

        phase   = self.state_mgr.phase
        elapsed = self.state_mgr.phase_elapsed(t)
        item_i  = self.state_mgr.item_index

        # ---- Arm + leg targets ----
        arm_tgt, leg_tgt = self.arm_planner.get_targets(phase.name, elapsed)

        # ---- Inject nav lean during walking phases ----
        if phase in (Phase.WALK_TO_ITEM, Phase.WALK_TO_DROP):
            pb, yb, _ = self.nav.compute_bias(qpos)
            leg_tgt = self.nav.apply_bias(leg_tgt, pb, yb)

        # ---- Always use QP-balanced StandingCtrl ----
        self.standing.arm_waist_target = arm_tgt
        self.standing.standing_angles  = leg_tgt.astype(np.float32)
        tau, info = self.standing.compute_torque(qpos, qvel)

        info['phase']      = phase.name
        info['item_index'] = item_i

        # ---- Grasp / release ----
        self._handle_grasping(phase, item_i)

        # ---- Phase advance ----
        self._check_advance(phase, elapsed, qpos, item_i, t)

        return tau, info

    # ------------------------------------------------------------------
    def reset(self):
        self.standing.reset()
        self.state_mgr  = TaskStateManager(num_items=len(ITEM_POSITIONS))
        self._started   = False
        self._sim_time  = 0.0
        self._grasped   = [False, False]
        self._set_weld("grasp_item1", False)
        self._set_weld("grasp_item2", False)

    # ------------------------------------------------------------------
    def _resolve_ids(self):
        def bid(n): return mujoco.mj_name2id(self.sim_model, mujoco.mjtObj.mjOBJ_BODY, n)
        def eid(n): return mujoco.mj_name2id(self.sim_model, mujoco.mjtObj.mjOBJ_EQUALITY, n)
        self.body_item1 = bid("item1")
        self.body_item2 = bid("item2")
        self.eq_ids     = [eid("grasp_item1"), eid("grasp_item2")]

    def _set_weld(self, name: str, active: bool):
        eid = mujoco.mj_name2id(self.sim_model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if eid >= 0:
            self.sim_model.eq_active0[eid] = int(active)

    def _handle_grasping(self, phase: Phase, item_i: int):
        if item_i >= len(ITEM_POSITIONS):
            return
        weld = f"grasp_item{item_i + 1}"
        if phase == Phase.GRASP_OBJECT and not self._grasped[item_i]:
            self._set_weld(weld, True)
            self._grasped[item_i] = True
            print(f"[{self._sim_time:.1f}s] GRASPED item {item_i+1}")
        elif phase == Phase.RELEASE and self._grasped[item_i]:
            self._set_weld(weld, False)
            self._grasped[item_i] = False
            print(f"[{self._sim_time:.1f}s] RELEASED item {item_i+1}")

    # ------------------------------------------------------------------
    def _check_advance(self, phase: Phase, elapsed: float,
                       qpos: np.ndarray, item_i: int, t: float):
        if not self.state_mgr.min_dwell_met(t):
            return

        base_xy = qpos[:2]

        if phase == Phase.STAND:
            self.state_mgr.advance(t)
            if item_i < len(ITEM_POSITIONS):
                self.nav.set_goal(ITEM_POSITIONS[item_i][:2])

        elif phase == Phase.WALK_TO_ITEM:
            if item_i < len(ITEM_POSITIONS):
                dist = np.linalg.norm(base_xy - ITEM_POSITIONS[item_i][:2])
                if dist < PICKUP_DIST:
                    self.state_mgr.advance(t)

        elif phase in (Phase.REACH_DOWN, Phase.GRASP_OBJECT, Phase.LIFT_OBJECT):
            self.state_mgr.advance(t)
            if phase == Phase.LIFT_OBJECT and item_i < len(DROP_POSITIONS):
                self.nav.set_goal(DROP_POSITIONS[item_i])

        elif phase == Phase.WALK_TO_DROP:
            if item_i < len(DROP_POSITIONS):
                dist = np.linalg.norm(base_xy - DROP_POSITIONS[item_i])
                if dist < DROP_DIST:
                    self.state_mgr.advance(t)

        elif phase in (Phase.PLACE_OBJECT, Phase.RELEASE):
            self.state_mgr.advance(t)

        elif phase == Phase.RETURN_TO_STAND:
            self.state_mgr.advance(t)
            next_i = self.state_mgr.item_index
            if next_i < len(ITEM_POSITIONS):
                self.nav.set_goal(ITEM_POSITIONS[next_i][:2])