# G1 Humanoid Pick-and-Place — Final Project

## Overview

This project implements a full pick-and-place task for the Unitree G1 27-DoF humanoid
in MuJoCo, building on:

| Homework | Component Used |
|----------|----------------|
| HW1      | State-machine architecture (`TaskStateManager`) |
| HW2      | Navigation controller design (Kalman-filter + planner concepts) |
| HW3      | `StandingCtrl` (Lagrangian QP balance), `Pushover` disturbance |

---

## File Structure

```
FinalProject/
├── deploy_pickup.py        ← Entry point — run this
├── pickup_ctrl.py          ← Main controller integrating all subsystems
├── task_state.py           ← HW1-style state machine (9 task phases)
├── arm_planner.py          ← Pose library + smooth interpolation per phase
├── nav_controller.py       ← Base steering toward waypoints
├── standing_ctrl.py        ← HW3 StandingCtrl (unchanged)
├── pushover.py             ← HW3 Pushover (unchanged)
├── requirements.txt
├── configs/
│   └── g1_pickup.yaml      ← Config (extends HW3 g1.yaml)
└── resources/
    └── g1/
        ├── scene_pickup.xml ← Custom scene with items, table, weld constraints
        ├── g1_27dof.xml     ← G1 robot model (from HW3)
        └── meshes/          ← STL meshes (from HW3)
```

---

## Task Phases

The state machine cycles through these phases for **each item** in order:

```
STAND → WALK_TO_ITEM → REACH_DOWN → GRASP_OBJECT → LIFT_OBJECT
      → WALK_TO_TABLE → PLACE_OBJECT → RELEASE → RETURN_TO_STAND
      → (repeat for item 2)
      → DONE
```

| Phase            | What happens |
|------------------|-------------|
| `STAND`          | Robot settles into stable standing pose (2 s dwell) |
| `WALK_TO_ITEM`   | NavController steers base toward item; arm in carry pose |
| `REACH_DOWN`     | Knees bend (squat); right arm extends down to floor level |
| `GRASP_OBJECT`   | MuJoCo weld constraint activated — object attaches to wrist |
| `LIFT_OBJECT`    | Robot stands up, arm rises to carry height |
| `WALK_TO_TABLE`  | NavController steers base toward drop-off table |
| `PLACE_OBJECT`   | Arm extends forward to table surface height |
| `RELEASE`        | Weld constraint deactivated — object rests on table |
| `RETURN_TO_STAND`| Arm returns to neutral; leg targets return to standing |

---

## Scene Layout

All positions are in the world frame (robot spawns at origin, facing +X):

```
                Item 1 (red)   Item 2 (blue)
                  [1.5, 0.4]    [1.5, -0.4]
Robot [0,0] ──────────────────────────────── Table [3.2, 0.0]
                                              Drop-zone 1: [3.35,  0.15]
                                              Drop-zone 2: [3.35, -0.15]
```

---

## Gripper

The G1 hand joints are **not actuated** in the base 27-DoF model (commented out
in `g1_27dof.xml`).  Instead, a **MuJoCo equality weld constraint** is used to
simulate grasping:

- `grasp_item1` and `grasp_item2` welds connect each cube to `right_wrist_yaw_link`
- The controller **enables** the weld in `GRASP_OBJECT` and **disables** it in `RELEASE`
- This cleanly simulates a gripper close/open without requiring separate finger joints

---

## Running

```bash
# Basic run (180 s simulation, with pushover disturbances)
python deploy_pickup.py

# No disturbances, no log
python deploy_pickup.py --no-pushover --no-log

# Custom duration
python deploy_pickup.py --duration 240
```

### Dependencies (same as HW3)

```bash
pip install mujoco numpy scipy cvxpy pyyaml matplotlib
```

---

## Design Notes

### Balance (HW3 StandingCtrl)
The `StandingCtrl` is used **without modification**.  The `PickupCtrl` overrides
`standing.arm_waist_target` and `standing.standing_angles` each control step to
inject the phase-appropriate pose targets, while the QP-based contact-force
solver and CoM PID continue to run transparently.

### Navigation
Rather than a full locomotion policy, the `NavController` injects small hip-pitch
and hip-yaw **offsets** into the standing-angle targets.  This causes the balance
controller to lean the robot toward the goal, producing slow forward/turning motion
while maintaining stability via the HW3 QP.

### Interpolation
All pose transitions use **smoothstep** interpolation to avoid torque spikes at
phase boundaries.

### Fall Recovery
Falls are detected by the existing HW3 fall-radius mechanism.  On fall, the robot
respawns at the origin, all weld constraints are deactivated, and the task state
machine resets to `STAND`.
