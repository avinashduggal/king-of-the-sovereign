"""Game board: territories and adjacency.

The default topology has 9 nodes (3 home + 6 contested) chosen so that:

* Invader needs to traverse at least two edges to reach the Defender home.
* C4 sits at the centre and connects four other territories (a natural
  flashpoint for resource capture).
* The Neutral home is reachable through a single contested territory (C6),
  making the Neutral nation strategically relevant without sharing a direct
  border with the main combatants.

```
    [I_HOME]──[C1]──[C3]──[D_HOME]
        │      │       │
       [C2]──[C4]────────┘
        │      │
       [C5]──[C6]──[N_HOME]
```

Switch to a different topology by constructing ``GameMap`` with custom
territories and adjacency dicts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Territory:
    id: int
    name: str
    resource_value: float
    strategic_value: float
    home_of: str | None  # "I", "D", "N", or None for contested


# ---------------------------------------------------------------------------
# Default 9-territory topology
# ---------------------------------------------------------------------------

DEFAULT_TERRITORIES: tuple[Territory, ...] = (
    Territory(0, "I_HOME", resource_value=0.30, strategic_value=0.80, home_of="I"),
    Territory(1, "C1",     resource_value=0.40, strategic_value=0.30, home_of=None),
    Territory(2, "C2",     resource_value=0.30, strategic_value=0.40, home_of=None),
    Territory(3, "C3",     resource_value=0.50, strategic_value=0.20, home_of=None),
    Territory(4, "C4",     resource_value=0.60, strategic_value=0.50, home_of=None),
    Territory(5, "C5",     resource_value=0.40, strategic_value=0.30, home_of=None),
    Territory(6, "D_HOME", resource_value=0.30, strategic_value=0.80, home_of="D"),
    Territory(7, "C6",     resource_value=0.30, strategic_value=0.40, home_of=None),
    Territory(8, "N_HOME", resource_value=0.20, strategic_value=0.50, home_of="N"),
)

DEFAULT_ADJACENCY: dict[int, list[int]] = {
    0: [1, 2],          # I_HOME -- C1, C2
    1: [0, 3, 4],       # C1     -- I_HOME, C3, C4
    2: [0, 4, 5],       # C2     -- I_HOME, C4, C5
    3: [1, 6],          # C3     -- C1, D_HOME
    4: [1, 2, 6, 7],    # C4     -- C1, C2, D_HOME, C6
    5: [2, 7],          # C5     -- C2, C6
    6: [3, 4],          # D_HOME -- C3, C4
    7: [4, 5, 8],       # C6     -- C4, C5, N_HOME
    8: [7],             # N_HOME -- C6
}


class GameMap:
    """Adjacency-graph representation of the battlefield.

    Nodes are territories indexed by integer id.  Edges are stored as a
    plain ``dict[int, list[int]]`` because the graph is tiny (≤ ~20 nodes
    in any reasonable topology) and we want zero external dependencies.
    """

    def __init__(
        self,
        territories: tuple[Territory, ...] = DEFAULT_TERRITORIES,
        adjacency: dict[int, list[int]] | None = None,
    ) -> None:
        self.territories: tuple[Territory, ...] = territories
        self.adjacency: dict[int, list[int]] = (
            DEFAULT_ADJACENCY if adjacency is None else adjacency
        )

        # Cache home-territory ids for fast lookups (must be done before
        # _validate, which checks that all three roles are present).
        self.home: dict[str, int] = {}
        for t in self.territories:
            if t.home_of is not None:
                self.home[t.home_of] = t.id

        self._validate()

    @property
    def n(self) -> int:
        """Number of territories."""
        return len(self.territories)

    def neighbors(self, tid: int) -> list[int]:
        """Adjacent territory ids."""
        return self.adjacency[tid]

    def is_adjacent(self, a: int, b: int) -> bool:
        """True iff territories ``a`` and ``b`` share an edge."""
        return b in self.adjacency.get(a, ())

    def shortest_path_length(self, a: int, b: int) -> int:
        """Edge count of the shortest path between ``a`` and ``b``.

        Returns ``-1`` if no path exists. Implemented as plain BFS — for
        a 9-node graph the constant factors do not matter.
        """
        if a == b:
            return 0
        visited = {a}
        queue: deque[tuple[int, int]] = deque([(a, 0)])
        while queue:
            node, dist = queue.popleft()
            for nb in self.adjacency[node]:
                if nb == b:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        return -1

    def connected_component(
        self, start: int, controlled: set[int]
    ) -> set[int]:
        """All territories reachable from ``start`` while staying inside
        ``controlled`` (used for supply-line connectivity)."""
        if start not in controlled:
            return set()
        seen = {start}
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for nb in self.adjacency[node]:
                if nb in controlled and nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        return seen

    # ---- validation ------------------------------------------------------

    def _validate(self) -> None:
        ids = {t.id for t in self.territories}
        expected = set(range(len(self.territories)))
        if ids != expected:
            raise ValueError(
                f"Territory ids must be 0..n-1; got {sorted(ids)}"
            )
        if set(self.adjacency.keys()) != expected:
            raise ValueError(
                "Adjacency dict must have an entry for every territory id"
            )
        for src, neighbours in self.adjacency.items():
            for dst in neighbours:
                if dst not in expected:
                    raise ValueError(
                        f"Edge {src}->{dst} references unknown territory"
                    )
                if src not in self.adjacency.get(dst, ()):
                    raise ValueError(
                        f"Edge {src}->{dst} is not symmetric "
                        f"(missing {dst}->{src})"
                    )
        if not all(role in self.home for role in ("I", "D", "N")):
            raise ValueError(
                "Map must define a home territory for I, D, and N"
            )
