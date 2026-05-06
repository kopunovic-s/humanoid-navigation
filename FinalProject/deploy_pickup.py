"""
Run:  mjpython deploy_pickup.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import os, time, argparse
import matplotlib.pyplot as plt

from pickup_ctrl import PickupCtrl


# ---------------------------------------------------------------------------
class DataLogger:
    def __init__(self):
        self.time, self.qpos, self.phase = [], [], []

    def log(self, t, qpos, phase):
        self.time.append(t)
        self.qpos.append(qpos[:3].copy())
        self.phase.append(phase)

    def plot(self, save_path=None):
        t    = np.array(self.time)
        pos  = np.array(self.qpos)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle("G1 Pick-and-Place — Run Summary")

        ax = axes[0]
        ax.plot(pos[:, 0], pos[:, 1], 'b-', lw=1)
        from pickup_ctrl import ITEM_POSITIONS, DROP_POSITIONS
        for i, p in enumerate(ITEM_POSITIONS):
            ax.plot(p[0], p[1], 'r^', ms=10, label=f'Item {i+1}')
        for i, d in enumerate(DROP_POSITIONS):
            ax.plot(d[0], d[1], 'gs', ms=10, label=f'Drop {i+1}')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_title('Base XY path'); ax.legend(fontsize=7); ax.grid(True)
        ax.set_aspect('equal')

        axes[1].plot(t, pos[:, 2])
        axes[1].set_xlabel('Time (s)'); axes[1].set_ylabel('Z (m)')
        axes[1].set_title('Pelvis height'); axes[1].grid(True)

        unique = list(dict.fromkeys(self.phase))
        p2i = {p: i for i, p in enumerate(unique)}
        axes[2].step(t, [p2i[p] for p in self.phase], color='purple')
        axes[2].set_yticks(range(len(unique)))
        axes[2].set_yticklabels(unique, fontsize=7)
        axes[2].set_title('Phase timeline'); axes[2].grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"[Plot] saved → {save_path}")
        plt.show()


# ---------------------------------------------------------------------------
def simulate(ctrl: PickupCtrl, duration: float = None, log: bool = True):
    xml_path = ctrl.get_xml_path()
    duration = duration or ctrl.get_duration()

    sim_model = mujoco.MjModel.from_xml_path(xml_path)
    sim_data  = mujoco.MjData(sim_model)
    sim_model.opt.timestep = ctrl.simulation_dt

    # Give controller references to the viewer model so welds work
    ctrl.sim_model = sim_model
    ctrl.sim_data  = sim_data

    # Re-resolve body/equality/freejoint IDs against this model instance.
    ctrl._resolve_ids()

    # Disable all welds at start
    for name in ("grasp_item1", "grasp_item2"):
        ctrl._set_weld(name, False)

    # Set initial robot pose without disturbing object positions
    robot_qpos = ctrl.get_initial_state()['qpos']
    robot_nq   = len(robot_qpos)
    mujoco.mj_resetData(sim_model, sim_data)
    sim_data.qpos[:robot_nq] = robot_qpos
    sim_data.qvel[:] = 0.0

    print(f"\n{'='*58}")
    print(f"  G1 Pick-and-Place Simulation")
    print(f"  scene nq={sim_model.nq}  robot nq={robot_nq}")
    print(f"  mass={np.sum(sim_model.body_mass):.1f} kg  "
          f"sim_dt={ctrl.simulation_dt*1000:.1f} ms  "
          f"ctrl_dt={ctrl.control_dt*1000:.1f} ms")
    print(f"  duration={duration:.0f} s")
    print(f"{'='*58}\n")

    logger  = DataLogger() if log else None
    tau     = np.zeros(ctrl.num_actuators)
    falls   = 0
    counter = 0

    with mujoco.viewer.launch_passive(sim_model, sim_data) as viewer:
        sim_time = 0.0
        while viewer.is_running() and sim_time < duration:
            step_start = time.time()

            # Fall detection — pelvis below 0.45 m means toppled
            if sim_data.qpos[2] < 0.45:
                falls += 1
                print(f"\n[{sim_time:.1f}s] FALL #{falls} — respawning...")
                mujoco.mj_resetData(sim_model, sim_data)
                sim_data.qpos[:robot_nq] = robot_qpos
                sim_data.qvel[:] = 0.0
                for name in ("grasp_item1", "grasp_item2"):
                    ctrl._set_weld(name, False)
                ctrl.reset()
                tau = np.zeros(ctrl.num_actuators)
                mujoco.mj_forward(sim_model, sim_data)
                viewer.sync()
                continue

            if counter % ctrl.control_decimation == 0:
                try:
                    robot_nv = robot_nq - 1  # nv=33 for robot-only model
                    tau, info = ctrl.compute_torque(
                        sim_data.qpos[:robot_nq].copy(),
                        sim_data.qvel[:robot_nv].copy())
                    if logger:
                        logger.log(sim_time, sim_data.qpos, info.get('phase',''))
                    # Status every 3 s
                    if counter % (ctrl.control_decimation * int(3.0/ctrl.control_dt)) == 0:
                        print(f"[{sim_time:6.1f}s] {info.get('phase','?'):<18s}  "
                              f"item={info.get('item_index',0)}  "
                              f"base=({sim_data.qpos[0]:.2f}, {sim_data.qpos[1]:.2f})  "
                              f"pelvis_z={sim_data.qpos[2]:.3f}")
                except Exception as e:
                    import traceback; traceback.print_exc()

            sim_data.ctrl[:] = tau
            mujoco.mj_step(sim_model, sim_data)
            counter  += 1
            sim_time  = counter * ctrl.simulation_dt
            viewer.sync()

            elapsed = time.time() - step_start
            if ctrl.simulation_dt > elapsed:
                time.sleep(ctrl.simulation_dt - elapsed)

    print(f"\nDone — sim_time={sim_time:.1f}s  falls={falls}")
    print(f"Final phase: {ctrl.state_mgr.label}  "
          f"items_done={ctrl.state_mgr.item_index}/{ctrl.state_mgr.num_items}")

    if logger and logger.time:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(out, exist_ok=True)
        logger.plot(save_path=os.path.join(out, 'run_summary.png'))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--no-log",  action="store_true")
    args = p.parse_args()

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configs", "g1_pickup.yaml")

    ctrl = PickupCtrl(config_path)
    simulate(ctrl, duration=args.duration, log=not args.no_log)
