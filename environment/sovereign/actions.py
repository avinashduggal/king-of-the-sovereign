"""Action enums for the SOVEREIGN environment.

The Invader's joint action is ``(political, military, target)``.

* ``political`` (5 options): SEEK_ALLIANCE, IMPOSE_SANCTION, ISSUE_THREAT,
  NEGOTIATE, DO_NOTHING.
* ``military`` (4 options): ADVANCE, HOLD, WITHDRAW, STRIKE.
* ``target`` (territory id 0..|V|-1): used by ADVANCE, STRIKE, optionally
  WITHDRAW. Ignored for HOLD and (by convention) DO_NOTHING.
"""

from __future__ import annotations

from enum import IntEnum


class PoliticalAction(IntEnum):
    SEEK_ALLIANCE = 0
    IMPOSE_SANCTION = 1
    ISSUE_THREAT = 2
    NEGOTIATE = 3
    DO_NOTHING = 4


class MilitaryAction(IntEnum):
    ADVANCE = 0
    HOLD = 1
    WITHDRAW = 2
    STRIKE = 3


N_POLITICAL = len(PoliticalAction)
N_MILITARY = len(MilitaryAction)


def political_name(value: int) -> str:
    """Human-readable name for a political action index."""
    return PoliticalAction(value).name


def military_name(value: int) -> str:
    """Human-readable name for a military action index."""
    return MilitaryAction(value).name
