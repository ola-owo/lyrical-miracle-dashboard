"""
Precompute and save data for the dashboard
"""

import pickle
import gzip

import numpy as np
import polars as pl
import polars.selectors as cs
import networkx as nx

from clustering import run_kmeans
from database import db_read_table
from common import (
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
    make_df_cluster_labels,
)

DATA_DIR.mkdir(exist_ok=True)

###
### Read data
###
df_plays = db_read_table('spotify.streams')
df_lyrics = db_read_table('genius.lyrics')
df_lyrics_embed = db_read_table('genius.lyrics_embed').with_columns(
    pl.col('embedding').cast(pl.Array(pl.Float64, EMBEDDING_DIM))
)
df_lyrics_big5 = db_read_table('genius.lyrics_big5').with_columns(
    pl.col('outputs').cast(pl.Array(pl.Float64, 5))
)
df_genius = db_read_table('genius.song_matches')


###
### Clustering
###
(
    df_lyrics_embed.lazy()
    .select('id', 'embedding')
    .cast(pl.col('embedding').cast(pl.Array(pl.Float32, EMBEDDING_DIM)))
    .sink_parquet(DATA_DIR / 'lyrics_embed.parquet')
)

print('running kmeans...')
km = run_kmeans(df_lyrics_embed['embedding'], N_CLUSTERS, RANDOM_SEED)

with gzip.open(KMEANS_FILE, 'wb') as f:
    pickle.dump(km, f)


df_cluster_labels = make_df_cluster_labels(N_CLUSTERS)
df_centroids = (
    pl.DataFrame(pl.Series('centroid', km.cluster_centers_))
    .with_row_index('cluster')
    .join(df_cluster_labels, 'cluster', 'left')
)

df_embeddings_clustered: pl.LazyFrame = (
    df_lyrics_embed.lazy()
    .rename({'id': 'g_id'})
    .with_columns(pl.Series('cluster', km.labels_))
    .join(df_centroids.lazy(), on='cluster')
    # EXPERIMENTAL spectral-clustering centroids
    # .with_columns(centroid_dist = pl.col('centroid').sub(pl.Series(spectra)))
    # EXPERIMENTAL end
    .with_columns(centroid_dist=pl.col('centroid').sub(pl.col('embedding')))
    .with_columns(
        pl.col('centroid_dist').map_batches(
            lambda vec: np.linalg.norm(vec, axis=1),
            returns_scalar=True,
            return_dtype=pl.Float64,
        )
    )
    .drop('centroid', 'embedding')  # dropping embedding/centroid vectors to save space
)

df_embeddings_clustered.sink_parquet(DATA_DIR / 'df_embeddings_clustered.parquet')


###
### Get and transform scrobbles
###
# Time bins are 6h bins that reset at 3am:
# [3:00 - 9:00) = morning
# [9:00 - 15:00) = midday
# [15:00 - 21:00) = evening
# [21:00 - 3:00) = night
plays_expanded: pl.LazyFrame = (
    df_plays.lazy()
    .with_columns(dt=pl.col('ts').dt.convert_time_zone(TIME_ZONE))
    .with_columns(
        year=pl.col('dt').dt.year(),
        month=pl.col('dt').dt.month(),
        day=pl.col('dt').dt.day(),
        weekday=pl.col('dt').dt.weekday().sub(1),
        time=pl.col('dt').dt.time(),
    )
    .with_columns(
        is_weekend=pl.col('weekday').ge(5),
        dectime=(
            pl.col('dt').dt.hour()
            + pl.col('dt').dt.minute() / 60
            + pl.col('dt').dt.second() / 3600
        ),
    )
    .with_columns(
        timebin=pl.col('dectime')
        .sub(3)
        .mod(24)
        .cut(TIME_BIN_BOUNDARIES, labels=TIME_BIN_LABELS, left_closed=True)
        .cast(pl.Enum(TIME_BIN_LABELS))
    )
    .select(
        'dt',
        (
            pl.col('spotify_track_uri')
            .str.split_exact(':', 2)
            .struct.field('field_2')
            .alias('id')
        ),
        pl.col('master_metadata_track_name').alias('song'),
        pl.col('master_metadata_album_album_name').alias('album'),
        pl.col('master_metadata_album_artist_name').alias('artist'),
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

plays_clustered: pl.LazyFrame = (
    df_lyrics_embed.lazy()
    .drop('embedding')
    .with_columns(cluster=pl.Series(km.labels_, dtype=pl.Int64))
    .join(df_lyrics.lazy(), on='id', how='inner')
    # merge with genius song metadata
    .join(df_genius.lazy(), left_on='id', right_on='g_id', how='inner')
    .rename({'id': 'g_id', 'id_right': 'id'})
    # merge with song plays
    .join(plays_expanded.lazy(), on='id')
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

df_sessions = (
    plays_clustered.with_columns(
        session=pl.col('timediff').fill_null(pl.duration()).gt(SES_MAX_GAP).cum_sum()
    )
    # add session-level stats
    .with_columns(
        prev_cluster=pl.col('cluster').shift(1).over('session'),
        # skipping latent_dist bc there's not much variation there
        # latent_dist=(
        #     pl.col('embedding') - pl.col('embedding').shift(1).over('session')
        # ),
        ses_start=pl.col('dt').min().over('session'),
        ses_end=pl.col('dt').max().over('session'),
        ses_len=pl.len().over('session'),
        n_within_ses=pl.row_index().over('session'),
    )
    .with_columns(
        # pl.col('latent_dist').map_batches(
        #     lambda vec: np.linalg.norm(vec, axis=1),
        #     returns_scalar=True,
        #     return_dtype=pl.Float64,
        # ),
        ses_dur=pl.col('ses_end') - pl.col('ses_start'),
    )
    .join(df_cluster_labels.lazy(), 'cluster', 'left')
    .select(
        [
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
            # 'latent_dist',
            'g_id',
            'song',
            'artist',
            'album',
        ]
    )
)


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
        [
            'cluster',
            'cluster_label',
            'freq',
            'n_plays',
            'n_unique_plays',
            'n_sessions',
            'top_song_id',
            'centroid',
        ]
    )
)

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
        # latent_dist_mean=pl.col('latent_dist').drop_nans().mean(),
        # latent_dist_med=pl.col('latent_dist').drop_nans().median(),
    )
    .with_columns(
        pl.date(pl.col('year'), pl.col('month'), 1),
        cluster_mode_freq=pl.col('cluster_mode_count').truediv(pl.col('n_plays')),
    )
    .join(_df_session_group_agg, on=['year', 'month'])
    .sort(['year', 'month'])
)

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
        [
            'year',
            'month',
            'date',
            'cluster',
            'cluster_label',
            'n_cluster_plays',
            'cluster_freq',
        ]
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
print('building cluster graph...')
ses_graph_full = nx.DiGraph()
ses_graph_full.add_nodes_from(
    [
        (c, {'name': n, 'weight': w})
        for c, n, w in (
            df_sessions.group_by(['cluster', 'cluster_label'])
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
print('getting big-5 scores...')
big5 = (
    df_sessions.lazy()
    .join(df_lyrics_big5.lazy(), left_on='g_id', right_on='id', how='left')
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
)
big5.sink_parquet(DATA_DIR / 'big5.parquet')
