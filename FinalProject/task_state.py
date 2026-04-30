"""
task_state.py  —  HW Final Project
===================================
Task phase definitions and state manager for the G1 humanoid pick-and-place task.

State machine (inspired by HW1's phased control pattern):

  STAND -> WALK_TO_ITEM -> REACH_DOWN -> GRASP_OBJECT -> LIFT_OBJECT
        -> WALK_TO_TABLE -> PLACE_OBJECT -> RELEASE -> RETURN_TO_STAND
        -> (repeat for item 2, then DONE)
"""

from enum import Enum, auto


class Phase(Enum):
    """All task phases in order."""
    STAND           = auto()   # Hold stable standing pose, settle dynamics
    WALK_TO_ITEM    = auto()   # Navigate base toward item pickup position
    REACH_DOWN      = auto()   # Bend knees + lower right arm to object height
    GRASP_OBJECT    = auto()   # Enable weld constraint (attach object to wrist)
    LIFT_OBJECT     = auto()   # Straighten up while holding object
    WALK_TO_TABLE   = auto()   # Navigate base toward table dropoff position
    PLACE_OBJECT    = auto()   # Lower arm to table surface height
    RELEASE         = auto()   # Disable weld constraint (release object)
    RETURN_TO_STAND = auto()   # Return arm to neutral, straighten posture
    DONE            = auto()   # All items placed


# Human-readable labels for logging
PHASE_LABELS = {
    Phase.STAND:           "STAND",
    Phase.WALK_TO_ITEM:    "WALK_TO_ITEM",
    Phase.REACH_DOWN:      "REACH_DOWN",
    Phase.GRASP_OBJECT:    "GRASP_OBJECT",
    Phase.LIFT_OBJECT:     "LIFT_OBJECT",
    Phase.WALK_TO_TABLE:   "WALK_TO_TABLE",
    Phase.PLACE_OBJECT:    "PLACE_OBJECT",
    Phase.RELEASE:         "RELEASE",
    Phase.RETURN_TO_STAND: "RETURN_TO_STAND",
    Phase.DONE:            "DONE",
}

# Minimum dwell time in each phase (seconds) before auto-advance
PHASE_MIN_DWELL = {
    Phase.STAND:           2.0,
    Phase.WALK_TO_ITEM:    0.5,   # advances on proximity
    Phase.REACH_DOWN:      1.5,
    Phase.GRASP_OBJECT:    0.3,
    Phase.LIFT_OBJECT:     1.2,
    Phase.WALK_TO_TABLE:   0.5,   # advances on proximity
    Phase.PLACE_OBJECT:    1.5,
    Phase.RELEASE:         0.3,
    Phase.RETURN_TO_STAND: 1.5,
    Phase.DONE:            9999,
}


class TaskStateManager:
    """
    Tracks which task phase we are in, which item is being handled,
    and provides helpers for phase transitions.

    Items are handled in order: item index 0, then item index 1.
    """

    def __init__(self, num_items: int = 2):
        self.num_items   = num_items
        self.phase       = Phase.STAND
        self.item_index  = 0            # which item we are currently working on
        self.phase_start = 0.0          # sim time when current phase began
        self.phase_count = 0            # number of completed phase transitions

    # ---------------------------------------------------------------
    def start(self, sim_time: float):
        """Call once at the beginning to record start time."""
        self.phase_start = sim_time

    def phase_elapsed(self, sim_time: float) -> float:
        """Seconds spent in the current phase."""
        return sim_time - self.phase_start

    def min_dwell_met(self, sim_time: float) -> bool:
        """True if we have stayed in the current phase long enough."""
        return self.phase_elapsed(sim_time) >= PHASE_MIN_DWELL[self.phase]

    # ---------------------------------------------------------------
    def advance(self, sim_time: float):
        """
        Move to the next phase.
        After RETURN_TO_STAND, if more items remain, restart from WALK_TO_ITEM
        with the next item; otherwise go to DONE.
        """
        if self.phase == Phase.DONE:
            return

        next_phase_map = {
            Phase.STAND:           Phase.WALK_TO_ITEM,
            Phase.WALK_TO_ITEM:    Phase.REACH_DOWN,
            Phase.REACH_DOWN:      Phase.GRASP_OBJECT,
            Phase.GRASP_OBJECT:    Phase.LIFT_OBJECT,
            Phase.LIFT_OBJECT:     Phase.WALK_TO_TABLE,
            Phase.WALK_TO_TABLE:   Phase.PLACE_OBJECT,
            Phase.PLACE_OBJECT:    Phase.RELEASE,
            Phase.RELEASE:         Phase.RETURN_TO_STAND,
            Phase.RETURN_TO_STAND: None,   # handled below
            Phase.DONE:            Phase.DONE,
        }

        if self.phase == Phase.RETURN_TO_STAND:
            self.item_index += 1
            if self.item_index < self.num_items:
                self._set_phase(Phase.WALK_TO_ITEM, sim_time)
            else:
                self._set_phase(Phase.DONE, sim_time)
        else:
            nxt = next_phase_map[self.phase]
            self._set_phase(nxt, sim_time)

        self.phase_count += 1

    # ---------------------------------------------------------------
    def _set_phase(self, phase: Phase, sim_time: float):
        old = self.phase
        self.phase       = phase
        self.phase_start = sim_time
        print(f"[{sim_time:.2f}s] Phase: {PHASE_LABELS[old]} "
              f"-> {PHASE_LABELS[phase]}  (item {self.item_index})")

    # ---------------------------------------------------------------
    @property
    def label(self) -> str:
        return PHASE_LABELS[self.phase]

    def __repr__(self):
        return (f"TaskStateManager(phase={self.label}, "
                f"item={self.item_index})")
