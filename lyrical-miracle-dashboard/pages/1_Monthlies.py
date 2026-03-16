import gzip
import pickle

import streamlit as st
import pendulum as pn
import networkx as nx
import polars as pl
import polars.selectors as cs
import plotly.express as px
import plotly.graph_objects as go

from graphs import plot_network_agraph
from albumart import get_genius_img, get_lastfm_img
from database import db_read_query
from common import (
    DATA_DIR,
    PALETTE,
    BIG5_TRAITS_SHORT,
    make_df_cluster_labels,
    timeout_popup,
)


timeout_popup()
st.set_page_config(page_title='Your Monthly Music Breakdown', page_icon='📅')
st.header('Your Monthly Music Breakdown')
st.sidebar.header('The Lyrical Miracle')


###
### Read data
###

df_sessions = pl.scan_parquet(DATA_DIR / 'df_sessions.parquet')
df_embeddings_clustered = pl.scan_parquet(DATA_DIR / 'df_embeddings_clustered.parquet')
df_cluster_stats = pl.scan_parquet(DATA_DIR / 'df_cluster_stats.parquet')
plays_expanded = pl.scan_parquet(DATA_DIR / 'plays_expanded.parquet')
plays_clustered = pl.scan_parquet(DATA_DIR / 'plays_clustered.parquet')
big5 = pl.scan_parquet(DATA_DIR / 'big5.parquet')


@st.cache_data
def read_cluster_graph():
    with gzip.open(DATA_DIR / 'ses_graph_full.pkl.gz') as f:
        return pickle.load(f)


@st.cache_data
def read_kmeans():
    with gzip.open(DATA_DIR / 'kmeans.pkl.gz') as f:
        return pickle.load(f)


km = read_kmeans()


###
### DATE INPUT
###

date_min = pn.instance(df_sessions.select('ses_start').min().collect().item()).date()
date_max = pn.instance(df_sessions.select('ses_end').max().collect().item()).date()
with st.sidebar:
    all_months = list(
        pn.interval(date_max.start_of('month'), date_min.start_of('month')).range(
            'months'
        )
    )
    st.session_state.selected_date = st.selectbox(
        'Listening period',
        all_months,
        index=None,
        format_func=lambda dt: dt.format('MMMM YYYY'),
    )
    print('selected date', st.session_state.selected_date)


###
### DATA WRANGLING
###
n_clusters = df_cluster_stats.select(pl.len()).collect().item()
df_cluster_labels = make_df_cluster_labels(n_clusters)


def filter_df_by_month(df, date: pn.Date):
    if not date:
        return df
    return df.filter((pl.col('year') == date.year) & (pl.col('month') == date.month))


df_centroids = (
    pl.DataFrame(pl.Series('centroid', km.cluster_centers_))
    .with_row_index('cluster')
    .join(df_cluster_labels, 'cluster', 'left')
)


def get_cluster_stats(df: pl.DataFrame | pl.LazyFrame):
    """Get aggregate stats per cluster"""
    return (
        df.lazy()
        .group_by('cluster')
        .agg(
            n_plays=pl.len(),
            n_unique_plays=pl.col('g_id').unique().len(),
            n_sessions=pl.col('session').unique().len(),
            top_song_id=pl.col('g_id').mode().first(),
        )
        .sort('cluster')
        .with_columns((pl.col('n_plays') / pl.col('n_plays').sum()).alias('freq'))
        .join(df_centroids.lazy(), on='cluster', how='full', coalesce=True)
        .with_columns(
            pl.col('freq', 'n_plays', 'n_unique_plays', 'n_sessions').replace(None, 0)
        )
        .select(
            'cluster',
            'cluster_label',
            'freq',
            'n_plays',
            'n_unique_plays',
            'n_sessions',
            'top_song_id',
            'centroid',
        )
    )


cluster_stats_this_period = get_cluster_stats(
    filter_df_by_month(df_sessions, st.session_state.selected_date)
).collect()


@st.cache_data
def get_cluster_examples(date: pn.Date) -> pl.DataFrame:
    # get intermediate df with ids of songs to query from db
    df = (
        df_embeddings_clustered.lazy()
        .filter(pl.col('centroid_dist').rank('dense').over('cluster') <= 3)
        .join(filter_df_by_month(df_sessions, date), on='g_id')
        .collect()
    )
    # get song ids from db
    g_ids_to_query = df['g_id'].unique()
    if g_ids_to_query.is_empty():
        genius_song_matches = pl.DataFrame(schema={'id': pl.String, 'g_id': pl.Int32})
    else:
        g_ids_to_query_str = '(' + ','.join(g_ids_to_query.cast(str)) + ')'
        g_ids_query = (
            'SELECT id, g_id FROM "genius"."song_matches" WHERE g_id in '
            + g_ids_to_query_str
        )
        genius_song_matches = db_read_query(g_ids_query)

    return (
        df.lazy()
        .join(genius_song_matches.lazy(), on='g_id')
        .unique(['cluster', 'g_id'])
        .drop(cs.ends_with('_right'))
        .join(
            df_cluster_stats.lazy().select('cluster', 'cluster_label'),
            on='cluster',
            how='left',
        )
        .sort('cluster', 'centroid_dist')
        .rename({'g_id': 'id', 'id': 'spotify_id'})
        .drop([cs.starts_with('g_'), 'timediff'])
        .collect()
    )


cluster_similar_tracks_this_month = get_cluster_examples(st.session_state.selected_date)


@st.cache_data
def get_album_art(cluster_similar_tracks_this_month: pl.DataFrame) -> dict[int, str]:
    """
    get album art urls,
    return dictionary of g_id->url
    """
    image_meta = (
        cluster_similar_tracks_this_month.lazy()
        # top (closest to centroid) result per cluster
        .filter(
            pl.col('centroid_dist') == pl.col('centroid_dist').min().over('cluster')
        )
        .unique('cluster')
        # get song names/ids
        .join(
            plays_expanded.lazy().select('song', 'artist', 'id'),
            ['song', 'artist'],
            how='left',
        )
        .rename({'id': 'g_id', 'id_right': 'song_id'})
        .select('cluster', 'cluster_label', 'song', 'artist', 'song_id', 'g_id')
        .sort('cluster')
        .collect()
    )

    # album art image urls
    image_urls = pl.Series([None] * image_meta.height, dtype=pl.String)

    # 1st pass: get Genius thumbnails from "genius_full" table
    for i, g_id in enumerate(image_meta['g_id']):
        if g_id:
            image_urls[i] = get_genius_img(g_id)

    # 2nd pass: get lastfm images (uses lastfm API)
    for i, songinfo in enumerate(image_meta.to_dicts()):
        if image_urls[i]:
            continue
        image_urls[i] = get_lastfm_img(
            artist=songinfo['artist'], song=songinfo['song'], mbid=songinfo['song_id']
        )

    image_meta = image_meta.with_columns(image=image_urls.replace(None, ''))
    image_dict = image_meta.select('cluster', 'image').rows_by_key('cluster')
    image_dict = {k: v[0][0] for k, v in image_dict.items()}
    return image_dict


# monthly graphs
@st.cache_data
def get_cluster_graph(date: pn.Date):
    if not date:
        return read_cluster_graph()

    df = filter_df_by_month(df_sessions, date)
    g = nx.DiGraph()
    g.add_nodes_from(
        [
            (c, {'name': n, 'weight': w})
            for c, n, w in (
                df.group_by(['cluster', 'cluster_label'])
                .len()
                .sort('cluster')
                .collect()
                .iter_rows()
            )
        ]
    )
    g.add_weighted_edges_from(
        df.filter(pl.col('ses_len') > 1)
        .select('prev_cluster', 'cluster')
        .drop_nulls()
        .group_by(cs.all())
        .len()
        .collect()
        .rows()
    )
    return g


###
### VISUALS
###


@st.fragment
def plot_network(date: pn.Date):
    """Show graph of listening sessions with song lyric clusters"""
    cluster_graph = get_cluster_graph(date)
    image_dict = get_album_art(cluster_similar_tracks_this_month)
    st.write(plot_network_agraph(cluster_graph, image_dict))


st.header(
    'Clustered listening sessions',
    help="""This graph shows how often you jump between clusters during listening sessions.
        A session is defined as a period of time with no more than 24h between plays.
        Nodes are weighted according to the amount of plays of each cluster.
        Edges are weighted according to the transition frequency between clusters.
        """,
    divider=True,
)
plot_network(st.session_state.selected_date)


@st.fragment
def plot_big5(date):
    """Plot Big 5 traits of all songs listened to during this period"""
    if date:
        big5_this_month = big5.filter(pl.col('date').dt.month_start() == date).collect()
    else:
        big5_this_month = big5.collect()
    fig_bar_big5 = px.bar(
        big5_this_month,
        x='trait_short',
        y='score',
        labels={
            'trait_short': 'Trait',
            'score': 'Score',
        },
        color='trait_short',
        custom_data=['score_pct', 'trait_desc'],
        category_orders={'trait_short': BIG5_TRAITS_SHORT},
    )
    fig_bar_big5.update_traces(
        hovertemplate='%{customdata[0]:.0f}% <b>%{customdata[1]}</b><extra></extra>'
    )
    fig_bar_big5.update_layout(
        xaxis_visible=False,
        yaxis_tickformat='.0%',
        hovermode='x',
    )
    st.plotly_chart(fig_bar_big5)


st.header(
    'Music personality',
    help='Average Big Five personality scores of your listened-to music lyrics',
    divider=True,
)
plot_big5(st.session_state.selected_date)


# pie chart of plays per cluster
@st.fragment
def plot_cluster_piechart():
    st.plotly_chart(
        go.Figure(
            [
                go.Pie(
                    labels=cluster_stats_this_period.select(
                        pl.format('Cluster {}', pl.col('cluster_label'))
                    ),
                    values=cluster_stats_this_period['n_plays'],
                    textinfo='label+value',
                    hoverinfo='percent',
                    sort=False,
                    marker=dict(colors=PALETTE),
                )
            ]
        )
    )


st.header('Cluster frequency', divider=True)
plot_cluster_piechart()


@st.fragment
def plot_cluster_times_polar(date: pn.Date):
    scrobbles_clustered_timebins = (
        pl.LazyFrame({'hour': list(range(24))})
        .join(pl.LazyFrame({'cluster': list(range(n_clusters))}), how='cross')
        .with_columns(pl.lit(0).alias('count'))
    )
    scrobbles_clustered_timebins = (
        filter_df_by_month(plays_clustered.lazy(), date)
        .group_by('cluster', pl.col('time').dt.hour().alias('hour'))
        .len('count')
        .join(scrobbles_clustered_timebins, on=['hour', 'cluster'], how='right')
        .join(df_cluster_labels.lazy(), 'cluster', 'left')
        .with_columns(pl.coalesce('count', 'count_right'))
        .sort('hour', 'cluster')
        .collect()
    )
    polar_hist_scrobble_times = px.bar_polar(
        scrobbles_clustered_timebins,
        r='count',
        theta='hour',
        color='cluster_label',
        color_discrete_sequence=PALETTE,
    )
    polar_hist_scrobble_times.update_layout(
        polar=dict(
            angularaxis=dict(
                type='category',
                tickvals=list(range(24)),
                direction='clockwise',
                rotation=90,
                period=24,  # Forces the circle to represent 24 units
            )
        )
    )
    st.plotly_chart(polar_hist_scrobble_times)


st.header(
    'Time of day listening frequency',
    help='Your listening habits at each hour of the day',
    divider=True,
)
plot_cluster_times_polar(st.session_state.selected_date)


# table of top tracks per cluster, this period
st.header(
    'Cluster-representative songs',
    help='Songs from this time period that are close to the cluster centroid',
    divider=True,
)
st.write(
    cluster_similar_tracks_this_month.select('cluster_label', 'song', 'artist', 'album')
)
