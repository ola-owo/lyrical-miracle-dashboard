from string import ascii_uppercase as ALPHABET
import json

import streamlit as st

import polars as pl
import polars.selectors as cs

import numpy as np
import polars as pl
import polars.selectors as cs
import networkx as nx
from scipy import sparse

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# import markov_clustering as mc
from plotly import graph_objects as go
from plotly import express as px
import plotly.colors

from lyric_analyzer_base.database import *
from lyric_analyzer_base.lastfm import get_image_url

###
### Globals
###
PALETTE = plotly.colors.qualitative.D3
RANDOM_SEED = 1738
TIME_ZONE = 'US/Eastern' # for converting scrobbles timestamps from UTC


###
### Stats stuff
###
def get_quantile(df: pl.DataFrame, i: int) -> pl.DataFrame:
    '''
    Find the quantiles of each column of row `i` of dataframe `df`.

    All dtypes of `df` must be numeric/comparable
    '''
    return df.select(cs.all().get(i).ge(cs.all()).mean())


###
### Read data
###
duckdb_read_table_cached = st.cache_data(duckdb_read_table)

# scrobbles = duckdb_read_table_cached('scrobbles')
df_lyrics = duckdb_read_table_cached('lyrics')
df_lyrics_embed = duckdb_read_table_cached('lyrics_embed')
df_genius = duckdb_read_table_cached('genius_song_matches')

df_lyrics_big5 = pl.read_parquet('data/genius_lyrics_big5.parquet')


###
### Get album art
###
@st.cache_data
def get_lastfm_img(artist, song, mbid=None) -> str|None:
    resp = get_image_url(artist, song, mbid)
    resp_js = json.loads(resp.content)
    try:
        # imgs are ordered from smallest to largest
        img_url = resp_js['track']['album']['image'][-1]['#text']
    except KeyError:
        img_url = None
    return img_url


@st.cache_data
def get_genius_img(g_id) -> str|None:
    q = f'SELECT song_art_image_thumbnail_url FROM "genius_full" WHERE id = ? LIMIT 1'
    params = [g_id]

    g = duckdb_query(q, params, read_only=True)
    if g.is_empty():
        return None

    # use song_art_image or header_image if they exist
    for imgcol in ['song_art_image_thumbnail_url', 'header_image_thumbnail_url']:
        # if image is missing, genius uses this default:
        # https://assets.genius.com/images/default_cover_image.png?[TIMESTAMP]
        if 'default_cover_image' not in g[0, imgcol]:
            return g[0, imgcol]

    return None

###
### Run clustering
###
N_CLUSTERS = 7
assert N_CLUSTERS <= len(PALETTE)

@st.cache_data
def run_kmeans():
    return KMeans(N_CLUSTERS, random_state=RANDOM_SEED).fit(df_lyrics_embed['embedding'])


@st.cache_data
def run_spectral_clustering():
    '''
    (EXPERIMENTAL) Run spectral clustering.
    For now, using symmetric normalized Laplacian bc its the only one that works well
    '''
    emb = df_lyrics_embed['embedding'].to_numpy()
    assert N_CLUSTERS < emb.shape[0]
    
    def selfsim(mat):
        '''
        Compute pairwise similarity between embeddings matrix.
        Input matrix should have one row per embedding.
        '''
        x = mat / np.linalg.norm(mat, axis=1).reshape((-1, 1))
        return x @ x.T
    
    
    def threshold(mat, thresh_pct):
        '''
        Keep the top `thresh_pct` most similar connections in `mat` (zero the rest),
        and convert it to a sparse matrix.
        '''
        #print(np.quantile(mat.flatten(), np.linspace(0, 1, 10, endpoint=False)))
        out = mat.copy()
        thresh = np.quantile(mat, 1 - thresh_pct)
        out[out < thresh] = 0
        return sparse.bsr_array(out)
    
    
    # build sparse graph
    SIM_THRESH = 0.8 # proportion of connections to keep
    emb_sim = selfsim(emb)
    np.fill_diagonal(emb_sim, 0.0) # remove similarity of nodes to themselves
    adj = threshold(emb_sim, SIM_THRESH)
    
    # compute symmetric normalized laplacian
    # NOTE: when normed, degree matrix 0s are set to 1 and nonzeros are square-rooted
    lap, deg = sparse.csgraph.laplacian(adj, normed=True, return_diag=True, copy=True)
    
    # get eigenvectors of laplacian
    deg = sparse.diags_array(np.square(deg))
    egvals, egvecs = sparse.linalg.eigsh(lap, k=N_CLUSTERS, which='SA')
    egvecs_normalizer = np.linalg.norm(egvecs, axis=1).reshape((-1, 1))
    egvecs_normalizer[egvecs_normalizer == 0] = 1
    egvecs = egvecs / egvecs_normalizer
    print('Laplacian eigenvalues:', egvals)
    
    # cluster the rows of the eigenvector matrix
    return KMeans(N_CLUSTERS, random_state=RANDOM_SEED).fit(egvecs), egvecs


km = run_kmeans()
# km, spectra = run_spectral_clustering = run_spectral_clustering()
df_cluster_labels = pl.DataFrame({
    'cluster': range(N_CLUSTERS),
    'cluster_label': [ALPHABET[i] for i in range(N_CLUSTERS)]
})
df_centroids = (
    pl.DataFrame(pl.Series('centroid', km.cluster_centers_))
    .with_row_index('cluster')
    .join(df_cluster_labels, 'cluster', 'left')
)

@st.cache_data
def get_df_embeddings_clustered():
    return (
        df_lyrics_embed
        .with_columns(pl.Series(km.labels_).alias('cluster'))
        .join(df_centroids, on='cluster')
        # EXPERIMENTAL start
        # .with_columns(pl.col('centroid').sub(pl.Series(spectra)).alias('centroid_dist'))
        .with_columns(pl.col('centroid').sub(pl.col('embedding')).alias('centroid_dist'))
        # EXPERIMENTAL end
        .with_columns(
            pl.col('centroid_dist').map_batches(
                lambda vec: np.linalg.norm(vec, axis=1),
                returns_scalar=True, return_dtype=pl.Float64
            )
        )
        .drop('centroid')
    )

df_embeddings_clustered = get_df_embeddings_clustered()


###
### Get and transform scrobbles
###
# Time bins are 6h bins that reset at 3am:
# [3:00 - 9:00) = morning
# [9:00 - 15:00) = midday
# [15:00 - 21:00) = evening
# [21:00 - 3:00) = night
TIME_BIN_BOUNDARIES = (6, 12, 18)
TIME_BIN_LABELS = ('morning', 'midday', 'evening', 'night')
TIME_BIN_PALETTE = ('#e2e38b', '#e7a553', '#7e4b68', '#292965')
@st.cache_data
def get_scrobbles_expanded() -> pl.DataFrame:
    return (
        duckdb_read_table_cached('scrobbles').lazy()
        .with_columns(pl.from_epoch(pl.col('ts'))
                      .dt.convert_time_zone(TIME_ZONE).alias('dt'))
        .with_columns(
            pl.col('dt').dt.year().alias('year'),
            pl.col('dt').dt.month().alias('month'),
            pl.col('dt').dt.day().alias('day'),
            pl.col('dt').dt.weekday().sub(1).alias('weekday'),
            pl.col('dt').dt.time().alias('time'),
        )
        .with_columns(
            pl.col('weekday').ge(5).alias('is_weekend'),
            (pl.col('dt').dt.hour() + pl.col('dt').dt.minute()/60 + pl.col('dt').dt.second()/3600).alias('dectime')
        )
        .with_columns(pl.col('dectime').sub(3).mod(24)
                      .cut(TIME_BIN_BOUNDARIES, labels=TIME_BIN_LABELS, left_closed=True)
                      .cast(pl.Enum(TIME_BIN_LABELS))
                      .alias('timebin'))
        .collect()
    )

@st.cache_data
def get_scrobbles_clustered() -> pl.DataFrame:
    return (
        df_lyrics_embed.lazy()
        .with_columns(pl.Series(km.labels_).alias('cluster'))
        .join(df_lyrics.lazy(), on='g_id', how='inner')
        # merge with genius song metadata
        .join(df_genius.lazy(), on='g_id', how='inner')
        # merge with scrobbles
        .join(scrobbles_expanded.lazy(), on=('artist','song'))
        # group into sessions of at most 1 day between scrobbles
        .sort('dt')
        .with_columns(
            pl.col('dt').diff().alias('timediff'),
            pl.col('dt').dt.month().alias('month'),
            pl.col('dt').dt.year().alias('year'),
            )
        .collect()
    )

scrobbles_expanded = get_scrobbles_expanded()
scrobbles_clustered = get_scrobbles_clustered()


def plot_network_agraph(g: nx.Graph, images: dict[int, str] = None):
    from streamlit_agraph import agraph, Node, Edge, Config
    import plotly
    
    node_labels = g.nodes(data='name')
    edges = g.edges(data='weight') # list of (src, dst, weight) tuples
    nodes = g.nodes(data='weight') # dict of {node: weight}
    nodes = [Node(
            id=i,
            label=node_labels[i],
            title=f'Cluster {node_labels[i]}',
            image=images.get(i, '') if images else '',
            shape='circularImage',
            borderWidth=8,
            color=PALETTE[i],
            value=nodes[i], # was previously out_degrees[i],
            chosen=False,
        ) for i in g.nodes]

    edges = [
        Edge(
            e[0], e[1],
            value=e[2],
        ) for e in edges]

    config = Config(
        directed=isinstance(g, nx.DiGraph),
        physics=True,
        solver='repulsion',
        interaction=dict(selectable=False),
    )

    return agraph(nodes, edges, config)

###
### Group scrobbles into contiguous sessions
###

# maximum gap for 2 songs to be grouped into the same session
SES_MAX_GAP = pl.duration(days=1)

df_sessions = (
    scrobbles_clustered.lazy()
    .with_columns(pl.col('timediff').fill_null(pl.duration()).gt(SES_MAX_GAP).cum_sum().alias('session'))
    # add session-level stats
    .with_columns(
        pl.col('cluster').shift(1).over('session').alias('prev_cluster'),
        (pl.col('embedding') - pl.col('embedding').shift(1).over('session')).alias('latent_dist'),
        pl.col('dt').min().over('session').alias('ses_start'),
        pl.col('dt').max().over('session').alias('ses_end'),
        pl.len().over('session').alias('ses_len'),
        pl.row_index().over('session').alias('n_within_ses')
        )
    .with_columns(
        pl.col('ses_end').sub(pl.col('ses_start')).alias('ses_dur'),
        pl.col('latent_dist').map_batches(
                lambda vec: np.linalg.norm(vec, axis=1),
                returns_scalar=True, return_dtype=pl.Float64
            )
        )
    .join(df_cluster_labels.lazy(), 'cluster', 'left')
    .select(['session', 'ses_start', 'ses_end', 'ses_dur', 'ses_len', 'n_within_ses',
             'dt', 'timebin', 'year', 'month', 'timediff', 'cluster', 'prev_cluster',
             'cluster_label', 'latent_dist', 'g_id', 'song', 'artist', 'album'])
    .collect()
)


###
### Group scrobbles by month
###

df_months = df_sessions.select(['year','month']).unique().sort(['year','month'])
n_months = df_months.height

@st.cache_data
def get_month_ix(year: int, month: int) -> int|None:
    '''
    Get the index corresponding to the given year/month.
    This is used to index `df_sessions_parts` and other lists based on it
    (`ses_graphs`, `df_cluster_monthly_stats`, ...)

    Return None if there's no match in `df_sessions_parts`.
    '''
    if not (year and month):
        return None
    
    rowix = df_months.select(
        pl.row_index()
        .get(((pl.col('year') == year) & (pl.col('month') == month)).arg_true())
        )
    
    return rowix.item() if not rowix.is_empty() else None


@st.cache_data
def get_df_sessions_this_month(month_ix: int) -> pl.DataFrame:
    # return df_sessions.join(df_months[month_ix].lazy(), on=['year','month'], how='semi')
    return df_sessions.filter((pl.col('year')==df_months[month_ix, 'year']) &
                              (pl.col('month')==df_months[month_ix, 'month']))


@st.cache_data
def get_cluster_stats(df: pl.DataFrame|pl.LazyFrame) -> pl.DataFrame:
    return (
        df.lazy()
        .group_by('cluster').agg(
            pl.len().alias('n_plays'),
            pl.col('g_id').unique().len().alias('n_unique_plays'),
            pl.col('session').unique().len().alias('n_sessions'),
            pl.col('g_id').mode().first().alias('top_song_id'),
        )
        .sort('cluster')
        .with_columns((pl.col('n_plays') / pl.col('n_plays').sum()).alias('freq'))
        .join(df_centroids.lazy(), 'cluster', 'full', coalesce=True)
        .with_columns(pl.col('freq', 'n_plays', 'n_unique_plays', 'n_sessions').replace(None, 0))
        .select(['cluster', 'cluster_label', 'freq', 'n_plays', 'n_unique_plays',
                'n_sessions', 'top_song_id', 'centroid'])
        .collect()
    )

@st.cache_data
def get_df_monthly_stats(month_ix: int) -> pl.DataFrame:
    return (
        get_df_sessions_this_month(month_ix)
        .select(
            'year', 'month',
            pl.len().alias('n_plays'),
            pl.col('session').max().add(1).alias('n_sessions'),
            pl.col('ses_len').median().alias('ses_len_median'),
            pl.col('ses_len').max().alias('ses_len_max'),
            pl.col('cluster').mode().first().alias('cluster_mode'),
            pl.col('cluster')
                .eq(pl.col('cluster').mode().first())
                .sum().alias('cluster_mode_count'),
            )
        .with_columns(
            pl.date(pl.col('year'), pl.col('month'), 1).alias('date'),
            pl.col('cluster_mode_count').truediv(pl.col('n_plays')).alias('cluster_mode_freq'),
            )
        .sort('date')
        .collect()
    )


@st.cache_data
def get_df_stats_all_months():
    # compute gini and shannon diversity:
    # https://en.wikipedia.org/wiki/Diversity_index
    group_agg = (
        df_sessions.lazy()
        .group_by(['year', 'month', 'cluster']).agg(pl.len())
        .with_columns((pl.col('len') / pl.col('len').sum().over(['year','month'])).alias('p'))
        .group_by(['year', 'month']).agg(
            pl.col('p').pow(2).sum().alias('gini'),
            (-pl.col('p') * pl.col('p').log(2)).sum().alias('shannon'),
            pl.col('p').max().pow(-1).alias('berger'),
            )
    )

    return (
        df_sessions.lazy()
        .group_by(['year', 'month']).agg(
            pl.len().alias('n_plays'),
            pl.col('session').max().add(1).alias('n_sessions'),
            pl.col('ses_len').median().alias('ses_len_median'),
            pl.col('ses_len').max().alias('ses_len_max'),
            pl.col('cluster').mode().first().alias('cluster_mode'),
            # pl.col('freq').get(pl.arg_where(pl.col('cluster') == pl.col('cluster').mode().first()))
            #     .alias('cluster_mode_freq'),
            pl.col('cluster').eq(pl.col('cluster').mode().first())
                .sum().alias('cluster_mode_count'),
            pl.col('latent_dist').drop_nans().mean().alias('latent_dist_mean'),
            pl.col('latent_dist').drop_nans().median().alias('latent_dist_med'),
        )
        .with_columns(
            pl.date(pl.col('year'), pl.col('month'), 1),
            pl.col('cluster_mode_count').truediv(pl.col('n_plays')).alias('cluster_mode_freq'),
            )
        .join(group_agg, on=['year', 'month'])
        .sort(['year', 'month'])
        .collect()
    )

df_stats_all_months = get_df_stats_all_months()
df_cluster_stats = get_cluster_stats(df_sessions)

@st.cache_data
def get_df_cluster_monthly_stats(month_ix):
    return get_cluster_stats(get_df_sessions_this_month(month_ix))

top_cluster_per_month = (
    scrobbles_clustered
    .group_by(['cluster', 'year', 'month']).len('n_plays')
    .group_by(['year', 'month'])
        .agg(pl.all().top_k_by('n_plays', 1)).explode(['cluster', 'n_plays'])
    .with_columns(pl.date(pl.col('year'), pl.col('month'), 1))
    .sort(['date'])
)

###
### Get representative songs per cluster
###
@st.cache_data
def get_cluster_similar_tracks(month_ix=None) -> pl.DataFrame:
    if month_ix:
        session_data = get_df_sessions_this_month(month_ix)
    else:
        session_data = df_sessions

    top_tracks = (
        df_embeddings_clustered.lazy()
        .join(session_data.lazy(), on='g_id')
        .filter(pl.col('centroid_dist').rank('dense').over('cluster') <= 3)
        .join(df_genius.lazy(), on='g_id')
        .unique(['cluster', 'g_id'])
        .drop(cs.ends_with('_right'))
        .join(df_cluster_stats.lazy().select(['cluster', 'cluster_label']), 'cluster', 'left')
        .sort('cluster', 'centroid_dist')
        .rename({'g_id': 'id'})
        .drop([cs.starts_with('g_'), 'timediff', 'searchtext'])
        .collect()
    )

    return top_tracks


###
### Songs per cluster per month
###
@st.cache_data
def get_df_centroid_similar_tracks(month_ix=None):
    if month_ix is None:
        sessions_with_embeddings = df_embeddings_clustered.join(get_df_sessions_this_month(month_ix), on='g_id')
    else:
        sessions_with_embeddings = df_embeddings_clustered

    return (
        sessions_with_embeddings
        .filter(pl.col('centroid_dist').rank('dense').over('cluster') <= 3)
        .unique(['cluster', 'g_id'])
        .join(df_genius, on='g_id')
        .drop(cs.ends_with('_right'))
        .sort('cluster', 'centroid_dist')
        .rename({'g_id': 'id'})
        .drop([cs.starts_with('g_'), 'timediff', 'searchtext'])
    )


###
### Summarize cluster frequencies per month
###
df_cluster_per_month = (
    scrobbles_clustered
    .group_by(['year', 'month', 'cluster'])
    .agg(pl.len().alias('n_cluster_plays'))
    .join(df_stats_all_months, on=['year', 'month'])
    .join(df_cluster_labels, on='cluster')
    .with_columns(
        pl.date(pl.col('year'), pl.col('month'), 1).alias('date'),
        pl.col('n_cluster_plays').truediv(pl.col('n_plays')).alias('cluster_freq'),
        )
    .select(['year', 'month', 'date', 'cluster', 'cluster_label', 'n_cluster_plays', 'cluster_freq'])
    .sort(['year', 'month', 'cluster'])
)


###
### Build cluster traversal graphs
### 

# full graph
ses_graph_full = nx.DiGraph()
ses_graph_full.add_nodes_from([
    (c, {'name': n, 'weight': w}) for c,n,w in
    df_sessions.group_by(['cluster', 'cluster_label']).len().sort('cluster').rows()
    ])
ses_graph_full.add_weighted_edges_from(
    df_sessions.filter(pl.col('ses_len') > 1).select(['prev_cluster', 'cluster'])
    .drop_nulls()
    .group_by(cs.all()).len().rows()
)

# monthly graphs
@st.cache_data
def get_ses_graph(month_ix):
    df = get_df_sessions_this_month(month_ix)
    g = nx.DiGraph()
    g.add_nodes_from([
        (c, {'name': n, 'weight': w}) for c,n,w in
        df_sessions.group_by(['cluster', 'cluster_label']).len().sort('cluster').rows()
        ])
    g.add_weighted_edges_from(
        df.filter(pl.col('ses_len') > 1).select(['prev_cluster', 'cluster'])
        .drop_nulls()
        .group_by(cs.all()).len().rows()
    )
    return g


###
### USER INPUT
###

# Date selection
date_min = df_sessions['ses_start'].min()
date_max = df_sessions['ses_end'].max()
st.session_state.selected_date = st.sidebar.date_input(
    "Choose a listening date:",
    value=None,
    min_value=date_min,
    max_value=date_max,
)

# this month's top songs per cluster (or all months)
st.session_state.month_ix = get_month_ix(st.session_state.selected_date.year, st.session_state.selected_date.month) \
    if st.session_state.selected_date else None

cluster_similar_tracks_this_month = get_cluster_similar_tracks(st.session_state.month_ix)

# this month's (or all months) cluster graph
cluster_graph = get_ses_graph(st.session_state.month_ix) \
    if st.session_state.month_ix else ses_graph_full

cluster_stats_this_period = get_cluster_stats(
    get_df_sessions_this_month(st.session_state.month_ix)
) if st.session_state.month_ix else df_cluster_stats

# album art metadata
image_meta = (
    cluster_similar_tracks_this_month.lazy()
    # top (closest to centroid) result per cluster
    .filter(pl.col('centroid_dist') == pl.col('centroid_dist').min().over('cluster'))
    .unique('cluster')
    # get song names/ids
    .join(scrobbles_expanded.lazy().select(['song', 'artist', 'song_id']),
          ['song','artist'], how='left')
    .select(['cluster', 'cluster_label', 'song', 'artist', 'song_id', pl.col('id').alias('g_id')])
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
    image_urls[i] = get_lastfm_img(artist=songinfo['artist'], song=songinfo['song'],
                                    mbid=songinfo['song_id'])

image_meta = image_meta.with_columns(image_urls.replace(None, '').alias('image'))
image_dict = image_meta.select(['cluster', 'image']).rows_by_key('cluster')
image_dict = {k: v[0][0] for k,v in image_dict.items()}


###
### VISUALS START HERE
###

st.title('The Lyrical Miracle')

# left_side, right_side = st.columns(2)

st.header('Graph of listening sessions')
st.write(plot_network_agraph(cluster_graph, image_dict))


@st.fragment
def plot_big5(month_ix=None):
    if month_ix is not None:
        big5 = (
            get_df_sessions_this_month(month_ix)
            .join(df_lyrics_big5, on='g_id', how='left')
            .select(df_lyrics_big5.columns)
        )
    else:
        big5 = df_lyrics_big5
    st.subheader('Your Music Personality')
    BIG5_TRAITS = (
        'Openness to Experience',
        'Conscientiousness',
        'Extraversion',
        'Agreeableness',
        'Neuroticism',
    )
    BIG5_TRAITS_SHORT = list('OCEAN')
    BIG5_TRAITS_POS = (
        'Open to experience',
        'Organized',
        'Extraverted',
        'Agreeable',
        'Neurotic',
    )
    BIG5_TRAITS_NEG = (
        'Closed to experince',
        'Messy',
        'Introverted',
        'Disagreeable',
        'Emotionally stable',
    )
    big5 = (
        big5
        .with_columns(trait = pl.lit(BIG5_TRAITS_SHORT))
        .select(
            trait = pl.col('trait').list.explode(),
            logit = pl.col('logits').arr.explode(),
        )
        .group_by('trait', maintain_order=True).mean()
        .with_columns(
            trait_desc = pl.when(pl.col('logit') >= 0)
                .then(pl.Series(BIG5_TRAITS_POS))
                .otherwise(pl.Series(BIG5_TRAITS_NEG)),
            score = pl.col('logit').tanh(),
        )
        .with_columns(score_pct = pl.col('score').abs() * 100)
    )
    fig_bar_big5 = px.bar(
        big5,
        x=BIG5_TRAITS_SHORT,
        y='score',
        labels={
            'trait': 'Trait',
            'score': 'Score',
        },
        color='trait',
        custom_data=['score_pct', 'trait_desc'],
    )
    fig_bar_big5.update_traces(
        hovertemplate="%{customdata[0]:.0f}% <b>%{customdata[1]}</b><extra></extra>"
    )
    fig_bar_big5.update_layout(
        yaxis=dict(  
            tickformat='.0%',
            # title='Score',
        ),
        hovermode='x',
    )  
    st.plotly_chart(fig_bar_big5)
plot_big5()


# pie chart of plays per cluster
st.header('Song plays per cluster')
st.plotly_chart(go.Figure([go.Pie(
    labels=cluster_stats_this_period.select(pl.format('Cluster {}', pl.col('cluster_label'))),
    values=cluster_stats_this_period['n_plays'],
    textinfo='label+value',
    hoverinfo='percent',
    sort=False,
    marker=dict(colors=PALETTE),
)]))


scrobbles_clustered_timebins = (
    pl.LazyFrame({'hour': list(range(24))})
    .join(pl.LazyFrame({'cluster': list(range(N_CLUSTERS))}), how='cross')
    .with_columns(pl.lit(0).alias('count'))
)
scrobbles_clustered_timebins = (
    scrobbles_clustered.lazy()
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
            period=24  # Forces the circle to represent 24 units
        )
    )
)
st.plotly_chart(polar_hist_scrobble_times)


# show table of top tracks per cluster, this period
st.write(cluster_similar_tracks_this_month.select(['cluster_label', 'song', 'artist', 'album']))

# Bar graphs of listening stats per month
st.header('Monthly Listening Stats')

# NOTE: `on_select` seems to be broken -
# the returned selection is always blank
# https://github.com/streamlit/streamlit/issues/8760
# bar_select = st.plotly_chart(
#     px.bar(
#         songs_per_month,
#         x='date',
#         y='len',
#         labels={
#             'date': 'Date',
#             'len': 'Songs played',
#         },
#     ),
#     on_select='rerun',
#     selection_mode='points',
# )
# st.write(bar_select)

st.subheader('Songs played per cluster')

barchart_plays_per_month = px.bar(
    df_cluster_per_month,
    x='date',
    y='n_cluster_plays',
    color='cluster_label',
    color_discrete_sequence=PALETTE,
    labels={
        'date': 'Date',
        'n_cluster_plays': 'Songs played',
        'cluster_label': 'Cluster'
    },
)

# TODO:highlight the bars where x = selected_date.replace(day=1)
# i tried plotly update_traces() but that updates all bars of the same cluster,
# instead we should highlight all bars of the same date
if st.session_state.selected_date:
    pass # TODO
st.plotly_chart(
    barchart_plays_per_month
    # on_select='rerun',
    # selection_mode='points',
    # key='bar_select2',
)
# st.write(st.session_state['bar_select2'])

st.subheader('Listening at each time of day')

songs_per_timebin_month = (
    df_sessions.group_by(['year', 'month', 'timebin']).len()
    .with_columns(
        pl.date(pl.col('year'), pl.col('month'), 1),
    )
    .sort(['timebin', 'date'])
)
fig_bar_timebin = px.bar(
    songs_per_timebin_month,
    x='date',
    y='len',
    color='timebin',
    color_discrete_sequence=TIME_BIN_PALETTE,
    labels={
        'date': 'Date',
        'len': 'Songs played',
        'timebin': 'Time of day'
    },
)
st.plotly_chart(fig_bar_timebin)


st.subheader('Latent distance between songs')
fig_bar_timebin = px.bar(
    df_stats_all_months,
    x='date',
    y='latent_dist_mean',
    barmode='group',
    labels={
        'date': 'Date',
        'latent_dist_mean': 'Latent-space distance',
    },
)
st.plotly_chart(fig_bar_timebin)


@st.fragment
def plot_diversity():
    st.subheader('Cluster diversity')
    div_cols = ('gini', 'shannon', 'berger')
    div_labels = ('Gini index', 'Shannon index', 'Inverse Berger-Parker index')
    st.session_state.diversity_type_ix = st.selectbox(
        'Diversity index:',
        list(range(len(div_cols))),
        format_func=div_labels.__getitem__
        )
    div_col = div_cols[st.session_state.diversity_type_ix]
    div_label = div_labels[st.session_state.diversity_type_ix]
    
    fig_bar_gini = px.bar(
        df_stats_all_months,
        x='date',
        y=div_col,
        barmode='group',
        labels={
            'date': 'Date',
            div_col: div_label,
        },
    )
    st.plotly_chart(fig_bar_gini)

plot_diversity()
