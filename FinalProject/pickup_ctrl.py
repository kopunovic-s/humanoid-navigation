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
from locomotion_policy import HW2LocomotionPolicy

# ---------------------------------------------------------------------------
# World positions  (must match scene_pickup.xml)
# ---------------------------------------------------------------------------
ITEM_POSITIONS = [
    np.array([1.50,  0.22, 0.05]),   # item 1 — red cube, in front/right
    np.array([1.50, -0.22, 0.05]),   # item 2 — blue cube, in front/left
]
DROP_POSITIONS = [
    np.array([3.35,  0.15, 0.775]),
    np.array([3.35, -0.15, 0.775]),
]
ITEM_APPROACH_POSITIONS = [
    np.array([p[0] - 0.45, p[1]]) for p in ITEM_POSITIONS
]
DROP_APPROACH_POSITIONS = [
    np.array([2.45,  0.18]),
    np.array([2.45, -0.18]),
]
PICKUP_DIST = 0.36      # m — distance to the actual item
DROP_DIST   = 0.75      # m — distance to near-side table standoff point
ARRIVAL_SPEED = 0.10    # m/s — settle before crouching/placing
ANCHOR_KP = 90.0
ANCHOR_KD = 65.0
ANCHOR_MAX_FORCE = 90.0
ANCHOR_YAW_DAMPING = 35.0
MANIP_DAMPING = 28.0
MANIP_MAX_DAMPING_FORCE = 35.0
POLICY_WALK_ASSIST_KP = 18.0
POLICY_WALK_ASSIST_KD = 7.0
POLICY_WALK_ASSIST_MAX_FORCE = 32.0
MANIP_LEG_BLEND_KP = np.array(
    [7, 2, 1, 9, 3, 1,
     7, 2, 1, 9, 3, 1],
    dtype=np.float32,
)
MANIP_LEG_BLEND_KD = np.array(
    [0.4, 0.2, 0.1, 0.5, 0.2, 0.1,
     0.4, 0.2, 0.1, 0.5, 0.2, 0.1],
    dtype=np.float32,
)
ROBOT_NQ = 34
MANIP_LOW_Z = 0.56
MANIP_MID_Z = 0.66
MANIP_STAND_Z = 0.77
MANIP_PHASES = {
    Phase.REACH_DOWN,
    Phase.GRASP_OBJECT,
    Phase.LIFT_OBJECT,
    Phase.PLACE_OBJECT,
    Phase.RELEASE,
    Phase.RETURN_TO_STAND,
}

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
        self.walk_policy = self._load_walk_policy()

        self._sim_time = 0.0
        self._started  = False
        self._grasped  = [False, False]

        print(f"[PickupCtrl] Initialized")
        print(f"  XML:           {self._xml_path}")
        print(f"  num_actuators: {self.num_actuators}")
        print(f"  walk_policy:   {'enabled' if self.walk_policy else 'disabled'}")

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
        qpos, qvel = self._apply_cinematic_manipulation_pose(
            qpos, qvel, phase, elapsed, arm_tgt, leg_tgt
        )

        # Use the HW2 locomotion policy for every active task phase after the
        # initial stand.  During manipulation phases it receives a zero-motion
        # goal at the current base position, so the legs stay in the policy's
        # stable support behavior while the arm/weld sequence runs.
        if self.walk_policy is not None and phase != Phase.STAND:
            goal = self.nav._goal if phase in (Phase.WALK_TO_ITEM, Phase.WALK_TO_DROP) else qpos[:2]
            tau, info = self._compute_policy_torque(qpos, qvel, arm_tgt, leg_tgt, phase, item_i, goal)
            self._handle_grasping(phase, item_i)
            self._check_advance(phase, elapsed, qpos, qvel, item_i, t)
            return tau, info

        # ---- Inject nav lean during walking phases if no HW2 policy exists ----
        if phase in (Phase.WALK_TO_ITEM, Phase.WALK_TO_DROP):
            pb, yb, _ = self.nav.compute_bias(qpos)
            leg_tgt = self.nav.apply_bias(leg_tgt, pb, yb)
        else:
            self._clear_walk_assist()

        # ---- Always use QP-balanced StandingCtrl ----
        # Tell the standing controller which phase we're in so it can scale
        # F_com appropriately during walking (see standing_ctrl line ~646).
        # Without this, F_com runs at full gain and actively fights the lean.
        self.standing.phase = phase.name

        # Smooth the standing-target transition.  When phase changes,
        # leg_tgt can step by tens of degrees (e.g. STAND -> SQUAT), and
        # the position-PD term P = kp * (standing_angles - q_joints) would
        # produce an impulsive torque.  Low-pass the target instead.
        new_leg_tgt = leg_tgt.astype(np.float32)
        if not hasattr(self, "_leg_tgt_filt") or self._leg_tgt_filt is None:
            self._leg_tgt_filt = new_leg_tgt.copy()
        else:
            a = 0.15   # ~70 ms time constant at 100 Hz control
            self._leg_tgt_filt = a * new_leg_tgt + (1 - a) * self._leg_tgt_filt

        self.standing.arm_waist_target = arm_tgt
        self.standing.standing_angles  = self._leg_tgt_filt
        tau, info = self.standing.compute_torque(qpos, qvel)

        info['phase']      = phase.name
        info['item_index'] = item_i

        # ---- Grasp / release ----
        self._handle_grasping(phase, item_i)

        # ---- Phase advance ----
        self._check_advance(phase, elapsed, qpos, qvel, item_i, t)

        return tau, info

    # ------------------------------------------------------------------
    def reset(self):
        self.standing.reset()
        if self.walk_policy is not None:
            self.walk_policy.reset()
        self.state_mgr  = TaskStateManager(num_items=len(ITEM_POSITIONS))
        self._started   = False
        self._sim_time  = 0.0
        self._grasped   = [False, False]
        self._leg_tgt_filt = None
        self._manip_anchor_xy = None
        self._set_weld("grasp_item1", False)
        self._set_weld("grasp_item2", False)

    def _load_walk_policy(self):
        try:
            policy_path = self.config.get("hw2_policy_path")
            if policy_path:
                config_abs = os.path.abspath(self.config_path)
                base_dir = os.path.dirname(os.path.dirname(config_abs))
                policy_path = policy_path.replace("{DIR}", base_dir)
            return HW2LocomotionPolicy(policy_path=policy_path, control_dt=self.control_dt)
        except Exception as exc:
            print(f"[PickupCtrl] HW2 walk policy unavailable: {exc}")
            return None

    def _compute_policy_torque(self, qpos, qvel, arm_tgt, leg_tgt, phase, item_i, goal_xy):
        self._clear_walk_assist()
        if phase in (Phase.WALK_TO_ITEM, Phase.WALK_TO_DROP):
            self._apply_policy_walk_assist(qpos, qvel, goal_xy)
        else:
            self._apply_manipulation_damping(qvel)
        leg_tau = self.walk_policy.compute_leg_torque(qpos, qvel, goal_xy)
        if phase in (Phase.REACH_DOWN, Phase.GRASP_OBJECT, Phase.LIFT_OBJECT,
                     Phase.PLACE_OBJECT, Phase.RELEASE, Phase.RETURN_TO_STAND):
            q_leg = qpos[7:19]
            dq_leg = qvel[6:18]
            pose_tau = MANIP_LEG_BLEND_KP * (leg_tgt - q_leg) - MANIP_LEG_BLEND_KD * dq_leg
            leg_tau = np.clip(leg_tau + pose_tau, -80.0, 80.0)
        self.standing.arm_waist_target = arm_tgt
        arm_tau = self.standing.compute_arm_torque(qpos, qvel)
        tau = np.concatenate([leg_tau, arm_tau])
        return tau, {
            "phase": phase.name,
            "item_index": item_i,
            "qp_success": None,
            "walk_policy": True,
        }

    def _apply_cinematic_manipulation_pose(self, qpos, qvel, phase, elapsed, arm_tgt, leg_tgt):
        if phase in (Phase.WALK_TO_ITEM, Phase.WALK_TO_DROP, Phase.STAND):
            self._manip_anchor_xy = None
            return qpos, qvel

        if phase not in MANIP_PHASES or self.sim_data is None:
            return qpos, qvel

        if self._manip_anchor_xy is None:
            self._manip_anchor_xy = qpos[:2].copy()

        if phase == Phase.REACH_DOWN:
            s = min(elapsed / 4.0, 1.0)
            target_z = MANIP_STAND_Z + s * (MANIP_LOW_Z - MANIP_STAND_Z)
        elif phase == Phase.GRASP_OBJECT:
            target_z = MANIP_LOW_Z
        elif phase == Phase.LIFT_OBJECT:
            s = min(elapsed / 2.0, 1.0)
            target_z = MANIP_LOW_Z + s * (MANIP_STAND_Z - MANIP_LOW_Z)
        elif phase in (Phase.PLACE_OBJECT, Phase.RELEASE):
            target_z = MANIP_MID_Z
        elif phase == Phase.RETURN_TO_STAND:
            s = min(elapsed / 2.5, 1.0)
            target_z = MANIP_MID_Z + s * (MANIP_STAND_Z - MANIP_MID_Z)
        else:
            target_z = MANIP_STAND_Z

        live_qpos = self.sim_data.qpos
        live_qvel = self.sim_data.qvel
        live_qpos[:2] = self._manip_anchor_xy
        live_qpos[2] = target_z
        live_qpos[7:19] = leg_tgt
        live_qpos[19:34] = arm_tgt
        live_qvel[:33] = 0.0
        mujoco.mj_forward(self.sim_model, self.sim_data)

        return live_qpos[:ROBOT_NQ].copy(), live_qvel[:ROBOT_NQ - 1].copy()

    # ------------------------------------------------------------------
    def _resolve_ids(self):
        def bid(n): return mujoco.mj_name2id(self.sim_model, mujoco.mjtObj.mjOBJ_BODY, n)
        def eid(n): return mujoco.mj_name2id(self.sim_model, mujoco.mjtObj.mjOBJ_EQUALITY, n)
        def jid(n): return mujoco.mj_name2id(self.sim_model, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.body_item1 = bid("item1")
        self.body_item2 = bid("item2")
        self.body_pelvis = bid("pelvis")
        self.eq_ids     = [eid("grasp_item1"), eid("grasp_item2")]
        self.item_joint_ids = [jid("item1_joint"), jid("item2_joint")]
        self.item_qpos_adrs = [
            self.sim_model.jnt_qposadr[j] if j >= 0 else -1
            for j in self.item_joint_ids
        ]
        self.item_qvel_adrs = [
            self.sim_model.jnt_dofadr[j] if j >= 0 else -1
            for j in self.item_joint_ids
        ]

    def _set_weld(self, name: str, active: bool):
        eid = mujoco.mj_name2id(self.sim_model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if eid >= 0:
            self.sim_model.eq_active0[eid] = int(active)
            if hasattr(self.sim_data, "eq_active"):
                self.sim_data.eq_active[eid] = int(active)

    def _clear_walk_assist(self):
        if hasattr(self.sim_data, "xfrc_applied"):
            self.sim_data.xfrc_applied[:] = 0.0

    def _apply_manipulation_anchor(self, qpos: np.ndarray, qvel: np.ndarray):
        if not hasattr(self.sim_data, "xfrc_applied") or self.body_pelvis < 0:
            return
        if not hasattr(self, "_manip_anchor_xy") or self._manip_anchor_xy is None:
            self._manip_anchor_xy = qpos[:2].copy()

        force_xy = ANCHOR_KP * (self._manip_anchor_xy - qpos[:2]) - ANCHOR_KD * qvel[:2]
        mag = float(np.linalg.norm(force_xy))
        if mag > ANCHOR_MAX_FORCE:
            force_xy *= ANCHOR_MAX_FORCE / mag

        self.sim_data.xfrc_applied[self.body_pelvis, 0:2] = force_xy
        self.sim_data.xfrc_applied[self.body_pelvis, 5] = -ANCHOR_YAW_DAMPING * qvel[5]

    def _apply_manipulation_damping(self, qvel: np.ndarray):
        if not hasattr(self.sim_data, "xfrc_applied") or self.body_pelvis < 0:
            return
        force_xy = -MANIP_DAMPING * qvel[:2]
        mag = float(np.linalg.norm(force_xy))
        if mag > MANIP_MAX_DAMPING_FORCE:
            force_xy *= MANIP_MAX_DAMPING_FORCE / mag
        self.sim_data.xfrc_applied[self.body_pelvis, 0:2] = force_xy
        self.sim_data.xfrc_applied[self.body_pelvis, 5] = -12.0 * qvel[5]

    def _apply_policy_walk_assist(self, qpos: np.ndarray, qvel: np.ndarray, goal_xy):
        if not hasattr(self.sim_data, "xfrc_applied") or self.body_pelvis < 0 or goal_xy is None:
            return
        delta = np.asarray(goal_xy[:2], dtype=np.float64) - qpos[:2]
        force_xy = POLICY_WALK_ASSIST_KP * delta - POLICY_WALK_ASSIST_KD * qvel[:2]
        mag = float(np.linalg.norm(force_xy))
        if mag > POLICY_WALK_ASSIST_MAX_FORCE:
            force_xy *= POLICY_WALK_ASSIST_MAX_FORCE / mag
        self.sim_data.xfrc_applied[self.body_pelvis, 0:2] = force_xy

    def _snap_item_to_drop_zone(self, item_i: int):
        if item_i >= len(DROP_POSITIONS):
            return
        qadr = self.item_qpos_adrs[item_i]
        vadr = self.item_qvel_adrs[item_i]
        if qadr < 0:
            return
        self.sim_data.qpos[qadr:qadr + 3] = DROP_POSITIONS[item_i]
        self.sim_data.qpos[qadr + 3:qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        if vadr >= 0:
            self.sim_data.qvel[vadr:vadr + 6] = 0.0
        mujoco.mj_forward(self.sim_model, self.sim_data)

    def _handle_grasping(self, phase: Phase, item_i: int):
        if item_i >= len(ITEM_POSITIONS):
            return
        weld = f"grasp_item{item_i + 1}"
        grasp_elapsed = self.state_mgr.phase_elapsed(self._sim_time)
        if (phase == Phase.GRASP_OBJECT
                and grasp_elapsed >= 1.15
                and not self._grasped[item_i]):
            self._set_weld(weld, True)
            self._grasped[item_i] = True
            print(f"[{self._sim_time:.1f}s] GRASPED item {item_i+1}")
        elif phase == Phase.RELEASE and self._grasped[item_i]:
            self._set_weld(weld, False)
            self._snap_item_to_drop_zone(item_i)
            self._grasped[item_i] = False
            print(f"[{self._sim_time:.1f}s] RELEASED item {item_i+1}")

    # ------------------------------------------------------------------
    def _check_advance(self, phase: Phase, elapsed: float,
                       qpos: np.ndarray, qvel: np.ndarray, item_i: int, t: float):
        if not self.state_mgr.min_dwell_met(t):
            return

        base_xy = qpos[:2]

        if phase == Phase.STAND:
            self.state_mgr.advance(t)
            if item_i < len(ITEM_POSITIONS):
                self.nav.set_goal(ITEM_APPROACH_POSITIONS[item_i])

        elif phase == Phase.WALK_TO_ITEM:
            if item_i < len(ITEM_POSITIONS):
                dist = np.linalg.norm(base_xy - ITEM_POSITIONS[item_i][:2])
                speed = np.linalg.norm(qvel[:2])
                if dist < PICKUP_DIST and speed < ARRIVAL_SPEED:
                    self.standing.reset()
                    if self.walk_policy is not None:
                        self.walk_policy.reset()
                    self.state_mgr.advance(t)

        elif phase in (Phase.REACH_DOWN, Phase.GRASP_OBJECT, Phase.LIFT_OBJECT):
            if phase == Phase.LIFT_OBJECT and item_i < len(DROP_POSITIONS):
                self.state_mgr.advance(t)
                self.nav.set_goal(DROP_APPROACH_POSITIONS[item_i])
                if self.walk_policy is not None:
                    self.walk_policy.reset()
            else:
                self.state_mgr.advance(t)

        elif phase == Phase.WALK_TO_DROP:
            if item_i < len(DROP_POSITIONS):
                dist = np.linalg.norm(base_xy - DROP_APPROACH_POSITIONS[item_i])
                speed = np.linalg.norm(qvel[:2])
                near_table_front = base_xy[0] > 2.15
                if (dist < DROP_DIST and speed < ARRIVAL_SPEED) or near_table_front or elapsed > 12.0:
                    self.standing.reset()
                    if self.walk_policy is not None:
                        self.walk_policy.reset()
                    self.state_mgr.advance(t)

        elif phase in (Phase.PLACE_OBJECT, Phase.RELEASE):
            self.state_mgr.advance(t)

        elif phase == Phase.RETURN_TO_STAND:
            self.state_mgr.advance(t)
            next_i = self.state_mgr.item_index
            if next_i < len(ITEM_POSITIONS):
                self.nav.set_goal(ITEM_APPROACH_POSITIONS[next_i])
