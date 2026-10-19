"""Network graph tools"""

import networkx as nx
import plotly.colors
import polars as pl
import polars.selectors as cs
from streamlit_agraph import Config, Edge, Node, agraph


def build_cluster_graph(df_sessions: pl.DataFrame | pl.LazyFrame) -> nx.DiGraph:
    """
    Build the cluster-transition graph from session-level plays.

    Nodes are clusters, weighted by play count.
    Edges are transitions between consecutive plays, weighted by frequency within sessions.

    Args:
        df_sessions: session-level plays with `cluster`, `cluster_label`,
            `prev_cluster`, and `ses_len` columns

    Returns: directed cluster-transition graph
    """
    df_sessions = df_sessions.lazy()
    g = nx.DiGraph()
    g.add_nodes_from(
        [
            (c, {'name': n, 'weight': w})
            for c, n, w in (
                df_sessions.drop_nulls('cluster')
                .group_by(['cluster', 'cluster_label'])
                .len()
                .sort('cluster')
                .collect()
                .iter_rows()
            )
        ]
    )
    g.add_weighted_edges_from(
        df_sessions.filter(pl.col('ses_len') > 1)
        .select('prev_cluster', 'cluster')
        .drop_nulls()
        .group_by(cs.all())
        .len()
        .collect()
        .rows()
    )
    return g


def plot_network_agraph(
    g: nx.Graph, images: dict[int, str] | None = None, palette=None
):
    if not palette:
        palette = plotly.colors.qualitative.D3

    node_labels = g.nodes(data='name')
    edges = g.edges(data='weight')  # list of (src, dst, weight) tuples
    nodes = g.nodes(data='weight')  # dict of {node: weight}
    nodes = [
        Node(
            id=i,
            label=node_labels[i],
            title=f'Cluster {node_labels[i]}',
            image=images.get(i, '') if images else '',
            shape='circularImage',
            borderWidth=8,
            color=palette[i],
            value=nodes[i],  # was previously out_degrees[i],
            chosen=False,
        )
        for i in g.nodes
    ]

    edges = [
        Edge(
            e[0],
            e[1],
            value=e[2],
        )
        for e in edges
    ]

    config = Config(
        directed=isinstance(g, nx.DiGraph),
        physics=True,
        solver='repulsion',
        interaction={'selectable': False},
    )

    return agraph(nodes, edges, config)
