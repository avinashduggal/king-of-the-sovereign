"""Adjacency / connectivity tests for the default 9-territory map."""

import pytest

from sovereign.game_map import (
    DEFAULT_ADJACENCY,
    DEFAULT_TERRITORIES,
    GameMap,
    Territory,
)


def test_default_map_constructs():
    m = GameMap()
    assert m.n == 9
    assert "I" in m.home and "D" in m.home and "N" in m.home
    assert m.home["I"] == 0
    assert m.home["D"] == 6
    assert m.home["N"] == 8


def test_adjacency_is_symmetric():
    for src, neighbours in DEFAULT_ADJACENCY.items():
        for dst in neighbours:
            assert src in DEFAULT_ADJACENCY[dst], (
                f"Edge {src}->{dst} is not symmetric"
            )


def test_invader_to_defender_path_is_at_least_two_edges():
    m = GameMap()
    d = m.shortest_path_length(m.home["I"], m.home["D"])
    assert d >= 2, f"Expected I_HOME->D_HOME distance >= 2, got {d}"


def test_neutral_reachable_through_c6():
    m = GameMap()
    # N_HOME (id 8) only connects to C6 (id 7)
    assert m.neighbors(8) == [7]


def test_invalid_topology_rejected():
    bad_terrs = (
        Territory(0, "A", 0.5, 0.5, "I"),
        Territory(2, "B", 0.5, 0.5, "D"),  # gap: missing id 1
        Territory(3, "C", 0.5, 0.5, "N"),
    )
    with pytest.raises(ValueError):
        GameMap(territories=bad_terrs, adjacency={0: [], 2: [], 3: []})


def test_connected_component_simple():
    m = GameMap()
    controlled = {0, 1}  # I_HOME + C1
    cc = m.connected_component(0, controlled)
    assert cc == {0, 1}


def test_connected_component_disconnected():
    m = GameMap()
    # Owns I_HOME and D_HOME but not the territories between them.
    controlled = {0, 6}
    cc = m.connected_component(0, controlled)
    assert cc == {0}, "Disconnected territory should not be in component"
