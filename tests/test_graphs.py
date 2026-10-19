"""Tests for dashboard.graphs"""

import polars as pl

from dashboard.graphs import build_cluster_graph


def test_build_cluster_graph_nodes(synth):
    g = build_cluster_graph(synth.sessions)

    assert set(g.nodes) == {0, 1, 2}
    assert g.nodes[0] == {'name': 'A', 'weight': 3}
    assert g.nodes[1] == {'name': 'B', 'weight': 2}
    assert g.nodes[2] == {'name': 'C', 'weight': 1}


def test_build_cluster_graph_edges(synth):
    g = build_cluster_graph(synth.sessions)

    # session 2 has a single play and must not add a self-loop
    assert set(g.edges) == {(0, 1), (1, 0), (2, 0)}
    assert g[0][1]['weight'] == 1


def test_build_cluster_graph_accepts_lazy(synth):
    assert build_cluster_graph(synth.sessions.lazy()).number_of_nodes() == 3


def test_build_cluster_graph_ignores_single_play_sessions():
    df = pl.DataFrame(
        [
            {
                'session': 1,
                'cluster': 0,
                'cluster_label': 'A',
                'prev_cluster': None,
                'ses_len': 1,
            },
            {
                'session': 1,
                'cluster': 1,
                'cluster_label': 'B',
                'prev_cluster': None,
                'ses_len': 1,
            },
        ]
    )

    g = build_cluster_graph(df)

    assert set(g.nodes) == {0, 1}
    assert g.number_of_edges() == 0
