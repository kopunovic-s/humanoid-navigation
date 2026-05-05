"""
task_state.py  —  HW Final Project

Cycling 9-phase pick-and-place state machine.

Per-item flow:
  STAND -> WALK_TO_ITEM -> REACH_DOWN -> GRASP_OBJECT -> LIFT_OBJECT
        -> WALK_TO_DROP  -> PLACE_OBJECT -> RELEASE -> RETURN_TO_STAND
        -> (RETURN_TO_STAND advances item_index; if more items remain,
            jumps back to WALK_TO_ITEM for the next item; otherwise DONE)

NOTE: do NOT add WALK_TO_ITEM2 / REACH_DOWN2 / ... etc.  ArmPlanner only
maps the 9 phase names listed below — any unknown phase falls through to
POSE_NEUTRAL/LEG_STAND, which causes a hard pose snap mid-task.
"""
from enum import Enum, auto


class Phase(Enum):
    STAND           = auto()
    WALK_TO_ITEM    = auto()
    REACH_DOWN      = auto()
    GRASP_OBJECT    = auto()
    LIFT_OBJECT     = auto()
    WALK_TO_DROP    = auto()
    PLACE_OBJECT    = auto()
    RELEASE         = auto()
    RETURN_TO_STAND = auto()
    DONE            = auto()


PHASE_LABELS = {p: p.name for p in Phase}

# Minimum dwell time per phase (seconds) before the controller is
# allowed to advance.  Walking phases also need their proximity check
# satisfied; the dwell here is just a floor.
PHASE_MIN_DWELL = {
    Phase.STAND:           5.0,
    Phase.WALK_TO_ITEM:    1.0,
    Phase.REACH_DOWN:      2.5,
    Phase.GRASP_OBJECT:    0.6,
    Phase.LIFT_OBJECT:     2.0,
    Phase.WALK_TO_DROP:    1.0,
    Phase.PLACE_OBJECT:    2.5,
    Phase.RELEASE:         0.6,
    Phase.RETURN_TO_STAND: 2.5,
    Phase.DONE:            9999,
}

# Linear next-phase map for everything *except* RETURN_TO_STAND.
# RETURN_TO_STAND is the cycle point: it either jumps back to
# WALK_TO_ITEM (more items) or ends in DONE (handled in advance()).
_NEXT = {
    Phase.STAND:           Phase.WALK_TO_ITEM,
    Phase.WALK_TO_ITEM:    Phase.REACH_DOWN,
    Phase.REACH_DOWN:      Phase.GRASP_OBJECT,
    Phase.GRASP_OBJECT:    Phase.LIFT_OBJECT,
    Phase.LIFT_OBJECT:     Phase.WALK_TO_DROP,
    Phase.WALK_TO_DROP:    Phase.PLACE_OBJECT,
    Phase.PLACE_OBJECT:    Phase.RELEASE,
    Phase.RELEASE:         Phase.RETURN_TO_STAND,
    Phase.RETURN_TO_STAND: None,    # handled specially — see advance()
    Phase.DONE:            None,
}


class TaskStateManager:
    def __init__(self, num_items: int = 2):
        self.num_items   = num_items
        self.phase       = Phase.STAND
        self.item_index  = 0
        self.phase_start = 0.0

    # -----------------------------------------------------------------
    def start(self, sim_time: float):
        self.phase_start = sim_time

    def phase_elapsed(self, sim_time: float) -> float:
        return sim_time - self.phase_start

    def min_dwell_met(self, sim_time: float) -> bool:
        return self.phase_elapsed(sim_time) >= PHASE_MIN_DWELL[self.phase]

    # -----------------------------------------------------------------
    def advance(self, sim_time: float):
        """Move to the next phase.  RETURN_TO_STAND cycles to the next item."""
        if self.phase == Phase.DONE:
            return

        if self.phase == Phase.RETURN_TO_STAND:
            self.item_index += 1
            if self.item_index < self.num_items:
                nxt = Phase.WALK_TO_ITEM
            else:
                nxt = Phase.DONE
        else:
            nxt = _NEXT[self.phase]
            if nxt is None:
                nxt = Phase.DONE

        old = self.phase
        self.phase       = nxt
        self.phase_start = sim_time
        print(f"[{sim_time:.1f}s] {old.name} -> {nxt.name}  (item {self.item_index})")

    # -----------------------------------------------------------------
    @property
    def label(self) -> str:
        return self.phase.name

    def __repr__(self):
        return f"TaskStateManager(phase={self.label}, item={self.item_index})"