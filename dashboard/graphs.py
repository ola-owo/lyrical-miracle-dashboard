"""Network graph tools"""

import networkx as nx
import plotly.colors
from streamlit_agraph import Config, Edge, Node, agraph


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
