"""
deploy_pickup.py  —  HW Final Project
=======================================
Entry point for the G1 humanoid pick-and-place simulation.

Run:
    python deploy_pickup.py
    python deploy_pickup.py --duration 180 --no-log
    python deploy_pickup.py --no-pushover

What it does:
  1. Loads the PickupCtrl (state machine + HW3 balance + arm planner + nav).
  2. Calls the same simulate() harness from HW3's deploy.py.
  3. Optionally applies pushover perturbations (HW3 Pushover).
  4. Logs and plots torque/joint data after the run.
"""

import mujoco
import mujoco.viewer
import numpy as np
import os
import time
import argparse
import matplotlib.pyplot as plt

from pickup_ctrl import PickupCtrl
from task_state  import Phase, PHASE_LABELS
from pushover    import Pushover   # HW3 pushover (unchanged)


# ---------------------------------------------------------------------------

class DataLogger:
    """Extended logger that also tracks task phase and CoM."""

    def __init__(self, num_actuators: int):
        self.num_actuators = num_actuators
        self.clear()

    def clear(self):
        self.time      = []
        self.qpos      = []
        self.qvel      = []
        self.tau_cmd   = []
        self.phase     = []
        self.com_err   = []

    def log(self, t, qpos, qvel, tau, phase_label="", com_err=None):
        self.time.append(t)
        self.qpos.append(qpos.copy())
        self.qvel.append(qvel.copy())
        self.tau_cmd.append(tau.copy())
        self.phase.append(phase_label)
        self.com_err.append(com_err.copy() if com_err is not None else np.zeros(3))

    def plot_summary(self, save_path: str = None):
        t       = np.array(self.time)
        qpos    = np.array(self.qpos)
        tau     = np.array(self.tau_cmd)
        com_err = np.array(self.com_err)

        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        fig.suptitle("G1 Pick-and-Place — Simulation Summary", fontsize=13)

        # Base XY trajectory
        ax = axes[0, 0]
        ax.plot(qpos[:, 0], qpos[:, 1], 'b-', lw=1.5, label='Base path')
        from pickup_ctrl import ITEM_POSITIONS, TABLE_XY, DROPZONE_XY
        for i, pos in enumerate(ITEM_POSITIONS):
            ax.plot(pos[0], pos[1], 'r^', ms=10, label=f'Item {i+1}')
        ax.plot(TABLE_XY[0], TABLE_XY[1], 'gs', ms=10, label='Table')
        for i, dz in enumerate(DROPZONE_XY):
            ax.plot(dz[0], dz[1], 'g+', ms=12, mew=2)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_title('Base XY Trajectory'); ax.legend(fontsize=7); ax.grid(True)
        ax.set_aspect('equal')

        # Base Z (height)
        ax = axes[0, 1]
        ax.plot(t, qpos[:, 2], 'b-', lw=1)
        ax.set_ylabel('Height (m)'); ax.set_title('Base Height')
        ax.set_xlabel('Time (s)'); ax.grid(True)

        # CoM error
        ax = axes[0, 2]
        ax.plot(t, com_err[:, 0], label='err_x')
        ax.plot(t, com_err[:, 1], label='err_y')
        ax.plot(t, com_err[:, 2], label='err_z')
        ax.set_ylabel('CoM error (m)'); ax.set_title('CoM Tracking Error')
        ax.set_xlabel('Time (s)'); ax.legend(fontsize=7); ax.grid(True)

        # Joint positions (first 12 — legs)
        ax = axes[1, 0]
        jp = qpos[:, 7:19]
        for i in range(12):
            ax.plot(t, jp[:, i], lw=0.8, label=f'j{i}')
        ax.set_ylabel('Rad'); ax.set_title('Leg Joint Positions')
        ax.set_xlabel('Time (s)'); ax.legend(fontsize=5, ncol=3); ax.grid(True)

        # Commanded torques (legs)
        ax = axes[1, 1]
        for i in range(min(12, tau.shape[1])):
            ax.plot(t, tau[:, i], lw=0.8, label=f'j{i}')
        ax.set_ylabel('Nm'); ax.set_title('Leg Commanded Torques')
        ax.set_xlabel('Time (s)'); ax.legend(fontsize=5, ncol=3); ax.grid(True)

        # Phase timeline
        ax = axes[1, 2]
        phase_arr = self.phase
        unique_phases = list(dict.fromkeys(phase_arr))
        phase_to_int  = {p: i for i, p in enumerate(unique_phases)}
        phase_int = np.array([phase_to_int[p] for p in phase_arr])
        ax.step(t, phase_int, 'purple', lw=1.5)
        ax.set_yticks(range(len(unique_phases)))
        ax.set_yticklabels(unique_phases, fontsize=6)
        ax.set_title('Task Phase Timeline'); ax.set_xlabel('Time (s)'); ax.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"[Plot] Saved to {save_path}")
        plt.show()


# ---------------------------------------------------------------------------

def simulate_pickup(ctrl: PickupCtrl,
                    duration: float = None,
                    log_data: bool   = True,
                    pushover         = None):
    """
    Simulation harness — mirrors HW3 deploy.py but adds:
      - Phase-aware status printing
      - Extended DataLogger
      - Gripper finger visual update (weld constraint toggle)
    """
    xml_path = ctrl.get_xml_path()
    if duration is None:
        duration = ctrl.get_duration()

    sim_model = mujoco.MjModel.from_xml_path(xml_path)
    sim_data  = mujoco.MjData(sim_model)
    sim_model.opt.timestep = ctrl.simulation_dt

    # Also link controller's internal model to sim_data for weld toggling
    ctrl.sim_model = sim_model
    ctrl.sim_data  = sim_data

    # Re-resolve IDs against the viewer's model instance
    import mujoco as mj
    def bid(name):
        return mj.mj_name2id(sim_model, mj.mjtObj.mjOBJ_BODY, name)
    def eid(name):
        return mj.mj_name2id(sim_model, mj.mjtObj.mjOBJ_EQUALITY, name)

    ctrl.body_item1       = bid("item1")
    ctrl.body_item2       = bid("item2")
    ctrl.body_wrist_right = bid("right_wrist_yaw_link")
    ctrl.eq_grasp1        = eid("grasp_item1")
    ctrl.eq_grasp2        = eid("grasp_item2")

    # Initial state
    INIT_FOOT_CLEARANCE = 0.1
    init_state = ctrl.get_initial_state()
    init_state['qpos'][2] += INIT_FOOT_CLEARANCE
    sim_data.qpos[:] = init_state['qpos']
    sim_data.qvel[:] = init_state['qvel']

    # Make sure all welds start inactive
    for name in ("grasp_item1", "grasp_item2"):
        eq_id = mj.mj_name2id(sim_model, mj.mjtObj.mjOBJ_EQUALITY, name)
        if eq_id >= 0:
            sim_model.eq_active0[eq_id] = 0

    pelvis_id = mj.mj_name2id(sim_model, mj.mjtObj.mjOBJ_BODY, "pelvis")

    print(f"\n{'='*60}")
    print(f"  G1 Humanoid Pick-and-Place Simulation")
    print(f"  XML:      {xml_path}")
    print(f"  Mass:     {np.sum(sim_model.body_mass):.1f} kg")
    print(f"  Sim dt:   {ctrl.simulation_dt*1000:.1f} ms")
    print(f"  Ctrl dt:  {ctrl.simulation_dt*ctrl.control_decimation*1000:.1f} ms")
    print(f"  Duration: {duration:.0f} s")
    print(f"  Pushover: {'enabled' if pushover else 'disabled'}")
    print(f"{'='*60}\n")

    logger = DataLogger(num_actuators=ctrl.num_actuators) if log_data else None

    control_counter = 0
    tau = np.zeros(ctrl.num_actuators)
    fall_count = 0
    spawn_xy   = init_state['qpos'][:2].copy()
    fall_radius = 0.45
    episode_start = 0.0

    PUSH_DURATION  = 0.5
    push_hold_steps = int(PUSH_DURATION / ctrl.simulation_dt)
    push_step_ctr   = 0
    current_force   = np.zeros(3)

    with mujoco.viewer.launch_passive(sim_model, sim_data) as viewer:
        sim_time = 0.0
        while viewer.is_running() and sim_time < duration:
            step_start = time.time()

            # Fall detection
            xy = sim_data.qpos[:2]
            if np.linalg.norm(xy - spawn_xy) > fall_radius:
                fall_count += 1
                print(f"\n[{sim_time:.1f}s] FALL #{fall_count} — respawning...")
                sim_data.qpos[:] = init_state['qpos']
                sim_data.qvel[:] = init_state['qvel']
                sim_data.xfrc_applied[:] = 0
                for name in ("grasp_item1", "grasp_item2"):
                    eq_id = mj.mj_name2id(sim_model, mj.mjtObj.mjOBJ_EQUALITY, name)
                    if eq_id >= 0:
                        sim_model.eq_active0[eq_id] = 0
                ctrl.reset()
                tau = np.zeros(ctrl.num_actuators)
                episode_start = sim_time
                push_step_ctr  = 0
                current_force  = np.zeros(3)
                mujoco.mj_forward(sim_model, sim_data)
                viewer.sync()
                continue

            # Control step
            if control_counter % ctrl.control_decimation == 0:
                qpos = sim_data.qpos.copy()
                qvel = sim_data.qvel.copy()

                try:
                    tau, info = ctrl.compute_torque(qpos, qvel)
                    com_err = (info.get('target_com', np.zeros(3)) -
                               info.get('current_com', np.zeros(3)))

                    if logger:
                        logger.log(sim_time, qpos, qvel, tau,
                                   phase_label=info.get('phase', ''),
                                   com_err=com_err)

                    # Status print every 2 seconds of control time
                    if control_counter % (ctrl.control_decimation * int(2.0 / ctrl.control_dt)) == 0:
                        phase = info.get('phase', '?')
                        item  = info.get('item_index', 0)
                        print(f"[{sim_time:6.1f}s] Phase: {phase:<18s}  "
                              f"item: {item}  "
                              f"base: ({qpos[0]:.2f}, {qpos[1]:.2f}, {qpos[2]:.2f})  "
                              f"com_err: ({com_err[0]:.3f}, {com_err[1]:.3f})")

                except Exception as e:
                    import traceback
                    print(f"[Warning] Control error: {e}")
                    traceback.print_exc()

            # Pushover force (5s grace period after spawn)
            ep_elapsed = sim_time - episode_start
            if pushover is not None and ep_elapsed > 5.0:
                if push_step_ctr % push_hold_steps == 0:
                    current_force = pushover.compute_force(
                        sim_time, sim_data.qpos, sim_data.qvel)
                push_step_ctr += 1
                if pelvis_id >= 0:
                    sim_data.xfrc_applied[pelvis_id, :3] = current_force

            sim_data.ctrl[:] = tau
            mujoco.mj_step(sim_model, sim_data)
            control_counter += 1
            sim_time = control_counter * ctrl.simulation_dt
            viewer.sync()

            sleep = ctrl.simulation_dt - (time.time() - step_start)
            if sleep > 0:
                time.sleep(sleep)

    print(f"\n{'='*60}")
    print(f"  Simulation finished — sim_time={sim_time:.1f}s, falls={fall_count}")
    final_phase = PHASE_LABELS.get(ctrl.state_mgr.phase, "?")
    print(f"  Final phase: {final_phase}")
    print(f"  Items placed: {ctrl.state_mgr.item_index} / {ctrl.state_mgr.num_items}")
    print(f"{'='*60}\n")

    if pushover is not None:
        risky = pushover.evaluate()
        print(f"Pushover assessment: {'RISKY' if risky else 'SAFE'}")

    if logger and len(logger.time) > 0:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(out_dir, exist_ok=True)
        plot_path = os.path.join(out_dir, 'pickup_summary.png')
        logger.plot_summary(save_path=plot_path)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G1 Pick-and-Place Deploy")
    parser.add_argument("--duration",    type=float, default=None,
                        help="Simulation duration in seconds")
    parser.add_argument("--no-log",      action="store_true",
                        help="Disable data logging and plots")
    parser.add_argument("--no-pushover", action="store_true",
                        help="Disable disturbance forces")
    args = parser.parse_args()

    # Config lives alongside this script
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configs", "g1_pickup.yaml")

    ctrl     = PickupCtrl(config_path)
    pushover = None if args.no_pushover else Pushover(seed=42)

    simulate_pickup(ctrl,
                    duration=args.duration,
                    log_data=not args.no_log,
                    pushover=pushover)