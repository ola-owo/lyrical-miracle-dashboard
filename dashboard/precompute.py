"""
Precompute and save data for the dashboard

Files saved:
    lyrics_embed: song ids and lyric-embeddings
    kmeans: embedding cluster assignments centroids
    df_embeddings_clustered: song ids and their assigned cluster
    plays_expanded: song plays table with extra computed cols
    plays_clustered: plays_expanded with cluster assignments
    df_sessions: session-level aggregates
    df_stats_all_months: month-level session aggregates
    df_cluster_stats: cluster-level aggregates
    df_cluster_per_month: cluster/month level aggregates
    ses_graph_full: cluster graph spanning the entire timespan
    big5: month-level average big5 scores
"""

import pickle
import gzip

import numpy as np
import polars as pl
import polars.selectors as cs
import networkx as nx

from dashboard.clustering import run_kmeans
from dashboard.database import db_read_table
from dashboard.common import (
    TIME_ZONE,
    RANDOM_SEED,
    DATA_DIR,
    TIME_BIN_BOUNDARIES,
    TIME_BIN_LABELS,
    BIG5_TRAITS,
    BIG5_TRAITS_SHORT,
    BIG5_TRAITS_POS,
    BIG5_TRAITS_NEG,
    KMEANS_FILE,
    SES_MAX_GAP,
    N_CLUSTERS,
    EMBEDDING_DIM,
    CLUSTER_VECTORS_PATH,
    SEARCH_VECTORS_PATH,
    make_df_cluster_labels,
)

DATA_DIR.mkdir(exist_ok=True)

###
### Read data
###
df_plays = db_read_table(
    'spotify.streams',
    [
        'ts',
        'spotify_track_uri',
        'master_metadata_track_name',
        'master_metadata_album_album_name',
        'master_metadata_album_artist_name',
    ],
).select(
    dt=pl.col('ts'),
    id=pl.col('spotify_track_uri').str.split_exact(':', 2).struct.field('field_2'),
    song=pl.col('master_metadata_track_name'),
    album=pl.col('master_metadata_album_album_name'),
    artist=pl.col('master_metadata_album_artist_name'),
)
df_scrobbles = db_read_table('lastfm.scrobbles', ['dt', 'song', 'artist', 'album'])
df_lyrics_cluster_vecs = db_read_table('genius.lyrics_embed_clustering').with_columns(
    pl.col('embedding').cast(pl.Array(pl.Float64, EMBEDDING_DIM))
)
df_lyrics_search_vecs = db_read_table('genius.lyrics_embed').with_columns(
    pl.col('embedding').cast(pl.Array(pl.Float32, EMBEDDING_DIM))
)
df_lyrics_big5 = db_read_table('genius.lyrics_big5').with_columns(
    pl.col('outputs').cast(pl.Array(pl.Float64, 5))
)
df_spotify_genius_matches = db_read_table('genius.song_matches')
df_lastfm_genius_matches = db_read_table('lastfm.genius_matches').drop_nulls('g_id')


###
### Clustering
###
df_lyrics_search_vecs.write_parquet(SEARCH_VECTORS_PATH)
(
    df_lyrics_cluster_vecs.lazy()
    .select('id', 'embedding')
    .with_columns(pl.col('embedding').cast(pl.Array(pl.Float32, EMBEDDING_DIM)))
    .sink_parquet(CLUSTER_VECTORS_PATH)
)

print('running kmeans...')
km = run_kmeans(df_lyrics_cluster_vecs['embedding'], N_CLUSTERS, RANDOM_SEED)

with gzip.open(KMEANS_FILE, 'wb') as f:
    pickle.dump(km, f)

# df_centroids: cluster labels and centroid vectors
df_cluster_labels = make_df_cluster_labels(N_CLUSTERS)
df_centroids = (
    pl.DataFrame(pl.Series('centroid', km.cluster_centers_))
    .with_row_index('cluster')
    .join(df_cluster_labels, 'cluster', 'left')
)

# df_embeddings_clustered: song ids, cluster labels, and centroid distances
# (dropping embedding/centroid vectors to save space)
df_embeddings_clustered: pl.LazyFrame = (
    df_lyrics_cluster_vecs.lazy()
    .rename({'id': 'g_id'})
    .with_columns(pl.Series('cluster', km.labels_))
    .join(df_centroids.lazy(), on='cluster')
    # EXPERIMENTAL spectral-clustering centroids
    # .with_columns(centroid_dist = pl.col('centroid').sub(pl.Series(spectra)))
    # EXPERIMENTAL end
    .with_columns(centroid_dist=pl.col('centroid') - pl.col('embedding'))
    .with_columns(
        pl.col('centroid_dist').map_batches(
            lambda vec: np.linalg.norm(vec, axis=1),
            returns_scalar=True,
            return_dtype=pl.Float64,
        )
    )
    .select('g_id', 'cluster', 'cluster_label', 'centroid_dist')
)

df_embeddings_clustered.sink_parquet(DATA_DIR / 'df_embeddings_clustered.parquet')


###
### Get and transform scrobbles
###

# plays_matched: lastfm and spotify plays with matching genius song ids
scrobbles_matched = df_scrobbles.lazy().join(
    df_lastfm_genius_matches.lazy().select('song', 'artist', 'g_id'),
    on=('song', 'artist'),
    how='left',
)
spotify_matched = (
    df_plays.lazy()
    .join(df_spotify_genius_matches.lazy().select('id', 'g_id'), on='id', how='left')
    .drop('id')
)
plays_matched = pl.concat((spotify_matched, scrobbles_matched))

# plays_expanded: plays_matched with expanded date/time info
#
# Time bins are 6h bins that reset at 3am:
# [3:00 - 9:00) = morning
# [9:00 - 15:00) = midday
# [15:00 - 21:00) = evening
# [21:00 - 3:00) = night
plays_expanded: pl.LazyFrame = (
    plays_matched.with_columns(pl.col('dt').dt.convert_time_zone(TIME_ZONE))
    .with_columns(
        year=pl.col('dt').dt.year(),
        month=pl.col('dt').dt.month(),
        day=pl.col('dt').dt.day(),
        weekday=pl.col('dt').dt.weekday() - 1,
        time=pl.col('dt').dt.time(),
        dectime=(
            pl.col('dt').dt.hour()
            + pl.col('dt').dt.minute() / 60
            + pl.col('dt').dt.second() / 3600
        ),
    )
    .with_columns(is_weekend=pl.col('weekday') >= 5)
    .with_columns(
        timebin=pl.col('dectime')
        .sub(3)
        .mod(24)
        .cut(TIME_BIN_BOUNDARIES, labels=TIME_BIN_LABELS, left_closed=True)
        .cast(pl.Enum(TIME_BIN_LABELS))
    )
    .select(
        'dt',
        'g_id',
        'song',
        'album',
        'artist',
        'year',
        'month',
        'day',
        'weekday',
        'time',
        'is_weekend',
        'dectime',
        'timebin',
    )
)

# plays_clustered: plays_expanded with cluster labels
plays_clustered: pl.LazyFrame = (
    df_lyrics_cluster_vecs.lazy()
    .select(id=pl.col('id'), cluster=pl.Series(km.labels_, dtype=pl.Int64))
    # merge with song plays
    .join(plays_expanded.lazy(), left_on='id', right_on='g_id', how='right')
    # group into sessions of at most 1 day between scrobbles
    .sort('dt')
    .with_columns(
        timediff=pl.col('dt').diff(),
        month=pl.col('dt').dt.month(),
        year=pl.col('dt').dt.year(),
    )
)


print('wrangling song plays...')
plays_expanded.sink_parquet(DATA_DIR / 'plays_expanded.parquet')
plays_clustered.sink_parquet(DATA_DIR / 'plays_clustered.parquet')


###
### Group scrobbles into contiguous sessions
###

# df_sessions: plays_clustered, grouped into sessions
# includes session number, start, end, duration, and length
# "session" means a continuous listening period with at most 24h between songs
df_sessions = (
    plays_clustered.with_columns(
        session=pl.col('timediff').fill_null(pl.duration()).gt(SES_MAX_GAP).cum_sum()
    )
    # add session-level stats
    .with_columns(
        prev_cluster=pl.col('cluster').shift(1).over('session'),
        ses_start=pl.col('dt').min().over('session'),
        ses_end=pl.col('dt').max().over('session'),
        ses_len=pl.len().over('session'),
        n_within_ses=pl.row_index().over('session'),
    )
    .with_columns(ses_dur=pl.col('ses_end') - pl.col('ses_start'))
    .join(df_cluster_labels.lazy(), 'cluster', 'left')
    .select(
        'session',
        'ses_start',
        'ses_end',
        'ses_dur',
        'ses_len',
        'n_within_ses',
        'dt',
        'timebin',
        'year',
        'month',
        'timediff',
        'cluster',
        'prev_cluster',
        'cluster_label',
        'g_id',
        'song',
        'artist',
        'album',
    )
)

# df_cluster_stats: cluster-level stats
df_cluster_stats = (
    df_sessions.lazy()
    .group_by('cluster')
    .agg(
        n_plays=pl.len(),
        n_unique_plays=pl.col('g_id').unique().len(),
        n_sessions=pl.col('session').unique().len(),
        top_song_id=pl.col('g_id').mode().first(),
    )
    .sort('cluster')
    .with_columns(freq=(pl.col('n_plays') / pl.col('n_plays').sum()))
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

# df_stats_all_months: month-level stats
# includes:
#   year and month
#   play count
#   session count
#   session aggregates
#   cluster mode
#   cluster diversity
_df_session_group_agg = (
    df_sessions.lazy()
    .group_by(['year', 'month', 'cluster'])
    .agg(pl.len())
    .with_columns(p=pl.col('len') / pl.col('len').sum().over(['year', 'month']))
    .group_by(['year', 'month'])
    .agg(
        gini=1.0 - pl.col('p').pow(2).sum(),
        shannon=(-pl.col('p') * pl.col('p').log(2)).sum(),
        berger=pl.col('p').max().pow(-1),
    )
)
df_stats_all_months = (
    df_sessions.lazy()
    .group_by(['year', 'month'])
    .agg(
        n_plays=pl.len(),
        n_sessions=pl.col('session').max().add(1),
        ses_len_median=pl.col('ses_len').median(),
        ses_len_max=pl.col('ses_len').max(),
        cluster_mode=pl.col('cluster').mode().first(),
        cluster_mode_count=(
            pl.col('cluster').eq(pl.col('cluster').mode().first()).sum()
        ),
    )
    .with_columns(
        pl.date(pl.col('year'), pl.col('month'), 1),
        cluster_mode_freq=pl.col('cluster_mode_count').truediv(pl.col('n_plays')),
    )
    .join(_df_session_group_agg, on=['year', 'month'])
    .sort(['year', 'month'])
    .select(
        'date',
        'year',
        'month',
        'n_plays',
        'n_sessions',
        'ses_len_median',
        'ses_len_max',
        'cluster_mode',
        'cluster_mode_count',
        'cluster_mode_freq',
        'gini',
        'shannon',
        'berger',
    )
)

# df_cluster_per_month: cluster distribution per year/month
df_cluster_per_month = (
    plays_clustered.group_by(['year', 'month', 'cluster'])
    .agg(pl.len().alias('n_cluster_plays'))
    .join(df_stats_all_months.lazy(), on=['year', 'month'])
    .join(df_cluster_labels.lazy(), on='cluster')
    .with_columns(
        pl.date(pl.col('year'), pl.col('month'), 1).alias('date'),
        pl.col('n_cluster_plays').truediv(pl.col('n_plays')).alias('cluster_freq'),
    )
    .select(
        'year',
        'month',
        'date',
        'cluster',
        'cluster_label',
        'n_cluster_plays',
        'cluster_freq',
    )
    .sort(['year', 'month', 'cluster'])
)

print('grouping plays into sessions...')
df_stats_all_months.sink_parquet(DATA_DIR / 'df_stats_all_months.parquet')
df_sessions.sink_parquet(DATA_DIR / 'df_sessions.parquet')
df_cluster_stats.sink_parquet(DATA_DIR / 'df_cluster_stats.parquet')
df_cluster_per_month.sink_parquet(DATA_DIR / 'df_cluster_per_month.parquet')


###
### Build cluster traversal graphs
###

# ses_graph_full: session graph spanning the full timeline
# nodes are clusters,
# edges are weighted by within-session transition frequency
print('building cluster graph...')
ses_graph_full = nx.DiGraph()
ses_graph_full.add_nodes_from(
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
ses_graph_full.add_weighted_edges_from(
    df_sessions.filter(pl.col('ses_len') > 1)
    .select('prev_cluster', 'cluster')
    .drop_nulls()
    .group_by(cs.all())
    .len()
    .collect()
    .rows()
)
with gzip.open(DATA_DIR / 'ses_graph_full.pkl.gz', 'wb') as f:
    pickle.dump(ses_graph_full, f)


###
### Big 5 scores
###

# big5: trait scores per year/month
# one row per trait, per year/month
# also includes neutral, postiive, and negative trait labels
# logit = raw output from model
# score = tanh[logit] (range -1 to 1)
# score_pct = score * 100
print('getting big-5 scores...')
big5 = (
    df_sessions.lazy()
    .join(df_lyrics_big5.lazy(), left_on='g_id', right_on='id')
    .with_columns(date=pl.date(pl.col('year'), pl.col('month'), 1))
    .select('outputs', 'date')
)
df_traits = pl.LazyFrame(
    dict(
        trait=BIG5_TRAITS,
        trait_short=BIG5_TRAITS_SHORT,
        trait_pos=BIG5_TRAITS_POS,
        trait_neg=BIG5_TRAITS_NEG,
    )
).with_row_index('n')
big5 = (
    big5.with_columns(n=range(5))
    .explode('n', 'outputs')
    .rename({'outputs': 'logit'})
    .group_by('date', 'n', maintain_order=True)
    .mean()
    .join(df_traits, on='n')
    .with_columns(
        trait_desc=pl.when(pl.col('logit') >= 0)
        .then(pl.col('trait_pos'))
        .otherwise(pl.col('trait_neg')),
        score=pl.col('logit').tanh(),
    )
    .with_columns(score_pct=pl.col('score').abs() * 100)
    .select(
        'date',
        'logit',
        'trait',
        'trait_short',
        'trait_pos',
        'trait_neg',
        'trait_desc',
        'score',
        'score_pct',
    )
)
big5.sink_parquet(DATA_DIR / 'big5.parquet')
