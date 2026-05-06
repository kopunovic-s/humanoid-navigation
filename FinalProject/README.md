# G1 Humanoid Pick-and-Place — Final Project

## Overview

This project implements a full pick-and-place task for the Unitree G1 27-DoF humanoid
in MuJoCo, building on:

| Homework | Component Used |
|----------|----------------|
| HW1      | State-machine architecture (`TaskStateManager`) |
| HW2      | TorchScript walking policy and waypoint command pipeline |
| HW3      | `StandingCtrl` (Lagrangian QP balance), `Pushover` disturbance |

---

## File Structure

```
FinalProject/
├── deploy_pickup.py        ← Entry point — run this
├── pickup_ctrl.py          ← Main controller integrating all subsystems
├── task_state.py           ← HW1-style state machine (9 task phases)
├── arm_planner.py          ← Pose library + smooth interpolation per phase
├── nav_controller.py       ← Base waypoint tracking and fallback lean steering
├── locomotion_policy.py    ← HW2 TorchScript locomotion-policy adapter
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
      → WALK_TO_DROP → PLACE_OBJECT → RELEASE → RETURN_TO_STAND
      → (repeat for item 2)
      → DONE
```

| Phase            | What happens |
|------------------|-------------|
| `STAND`          | Robot settles into stable standing pose inside the 0.3 m spawn disk |
| `WALK_TO_ITEM`   | NavController steers base toward item; arm in carry pose |
| `REACH_DOWN`     | Knees bend (squat); right arm extends down to floor level |
| `GRASP_OBJECT`   | MuJoCo weld constraint activated — object attaches to wrist |
| `LIFT_OBJECT`    | Robot stands up, arm rises to carry height |
| `WALK_TO_DROP`   | NavController steers base toward drop-off table |
| `PLACE_OBJECT`   | Robot shallow-crouches and leans the arm toward the tabletop |
| `RELEASE`        | Weld constraint deactivated — object rests on table |
| `RETURN_TO_STAND`| Arm returns to neutral; leg targets return to standing |

---

## Scene Layout

All positions are in the world frame (robot spawns at origin, facing +X):

```
Spawn disk diameter: 0.3 m

                Item 1 (red)   Item 2 (blue)
                 [1.5, 0.22]   [1.5, -0.22]
Robot [0,0] ──────────────────────────────── Table [3.15, 0.0]
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
# Basic run (300 s simulation)
python deploy_pickup.py

# No disturbances, no log
python deploy_pickup.py --no-log

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
The `StandingCtrl` is still used for the initial stable stand and as the fallback
balance controller if the HW2 walking policy cannot be loaded.

### Navigation
Walking uses the trained HW2 TorchScript locomotion policy in
`../../HW2/HW2/policy/motion.pt`.  `PickupCtrl` converts each item/table waypoint
into conservative forward/yaw velocity commands and lets the policy control the
12 leg actuators.  During reach/grasp/place phases, the same policy receives a
zero-motion goal so the legs stay in the policy's stable support behavior while
the arm/weld sequence runs.  If the policy cannot be loaded, `NavController`
falls back to the older lean-steering approach.

### Interpolation
All pose transitions use **smoothstep** interpolation to avoid torque spikes at
phase boundaries.

### Fall Recovery
Falls are detected by the existing HW3 fall-radius mechanism.  On fall, the robot
respawns at the origin, all weld constraints are deactivated, and the task state
machine resets to `STAND`.
