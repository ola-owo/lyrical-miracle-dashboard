"""
Precompute and save data for the dashboard
"""

from string import ascii_uppercase as ALPHABET
import pickle
import gzip

import duckdb
import numpy as np
import polars as pl
import polars.selectors as cs
import networkx as nx

from lyric_analyzer_base.database import DUCKDB_FILE

from clustering import run_kmeans
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
)

# maximum gap for 2 songs to be grouped into the same session
SES_MAX_GAP = pl.duration(days=1)


###
### Read data
###
df_plays = pl.scan_parquet('data/Streaming_History_Song.parquet')
df_lyrics_big5 = pl.scan_parquet('data/genius_lyrics_big5.parquet')

with duckdb.connect(DUCKDB_FILE, read_only=True) as cxn:
    df_lyrics = cxn.table('genius.lyrics').pl().lazy()
    df_lyrics_embed = cxn.table('genius.lyrics_embed').pl().lazy()
    df_genius = cxn.table('genius.song_matches').pl().lazy()


###
### Clustering
###
N_CLUSTERS = 7
with duckdb.connect(DUCKDB_FILE, read_only=True) as cxn:
    df_lyrics_embed = cxn.table('genius.lyrics_embed').pl()

print('running kmeans...')
km = run_kmeans(df_lyrics_embed['embedding'], N_CLUSTERS, RANDOM_SEED)
# km, spectra = run_spectral_clustering = run_spectral_clustering(df_lyrics_embed.collect()['embedding'], N_CLUSTERS, RANDOM_SEED)

with gzip.open(KMEANS_FILE, 'wb') as f:
    pickle.dump(km, f)


df_cluster_labels = pl.DataFrame(
    {
        'cluster': range(N_CLUSTERS),
        'cluster_label': [ALPHABET[i] for i in range(N_CLUSTERS)],
    }
)
df_centroids = (
    pl.DataFrame(pl.Series('centroid', km.cluster_centers_))
    .with_row_index('cluster')
    .join(df_cluster_labels, 'cluster', 'left')
)


def get_df_embeddings_clustered():
    return (
        df_lyrics_embed.lazy()
        .rename({'id': 'g_id'})
        .with_columns(pl.Series(km.labels_).alias('cluster'))
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
        .drop('centroid')
    )


df_embeddings_clustered = get_df_embeddings_clustered()
df_embeddings_clustered.sink_parquet(DATA_DIR / 'df_embeddings_clustered.parquet')


###
### Get and transform scrobbles
###
# Time bins are 6h bins that reset at 3am:
# [3:00 - 9:00) = morning
# [9:00 - 15:00) = midday
# [15:00 - 21:00) = evening
# [21:00 - 3:00) = night
def get_plays_expanded():
    return (
        df_plays.with_columns(dt=pl.col('ts').dt.convert_time_zone(TIME_ZONE))
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
            pl.col('spotify_track_uri')
            .str.split_exact(':', 2)
            .struct.field('field_2')
            .alias('id'),
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


def get_plays_clustered():
    return (
        df_lyrics_embed.lazy()
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
            pl.col('dt').diff().alias('timediff'),
            pl.col('dt').dt.month().alias('month'),
            pl.col('dt').dt.year().alias('year'),
        )
    )


print('wrangling song plays...')
plays_expanded = get_plays_expanded()
plays_clustered = get_plays_clustered()
plays_expanded.sink_parquet(DATA_DIR / 'plays_expanded.parquet')
plays_clustered.sink_parquet(DATA_DIR / 'plays_clustered.parquet')


###
### Group scrobbles into contiguous sessions
###

df_sessions = (
    plays_clustered.with_columns(
        pl.col('timediff')
        .fill_null(pl.duration())
        .gt(SES_MAX_GAP)
        .cum_sum()
        .alias('session')
    )
    # add session-level stats
    .with_columns(
        prev_cluster=pl.col('cluster').shift(1).over('session'),
        latent_dist=(
            pl.col('embedding') - pl.col('embedding').shift(1).over('session')
        ),
        ses_start=pl.col('dt').min().over('session'),
        ses_end=pl.col('dt').max().over('session'),
        ses_len=pl.len().over('session'),
        n_within_ses=pl.row_index().over('session'),
    )
    .with_columns(
        pl.col('latent_dist').map_batches(
            lambda vec: np.linalg.norm(vec, axis=1),
            returns_scalar=True,
            return_dtype=pl.Float64,
        ),
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
            'latent_dist',
            'g_id',
            'song',
            'artist',
            'album',
        ]
    )
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


def get_df_stats_all_months():
    # compute gini and shannon diversity:
    # https://en.wikipedia.org/wiki/Diversity_index
    group_agg = (
        df_sessions.lazy()
        .group_by(['year', 'month', 'cluster'])
        .agg(pl.len())
        .with_columns(
            (pl.col('len') / pl.col('len').sum().over(['year', 'month'])).alias('p')
        )
        .group_by(['year', 'month'])
        .agg(
            gini=1.0 - pl.col('p').pow(2).sum(),
            shannon=(-pl.col('p') * pl.col('p').log(2)).sum(),
            berger=pl.col('p').max().pow(-1),
        )
    )

    return (
        df_sessions.lazy()
        .group_by(['year', 'month'])
        .agg(
            pl.len().alias('n_plays'),
            n_sessions=pl.col('session').max().add(1),
            ses_len_median=pl.col('ses_len').median(),
            ses_len_max=pl.col('ses_len').max(),
            cluster_mode=pl.col('cluster').mode().first(),
            # pl.col('freq').get(pl.arg_where(pl.col('cluster') == pl.col('cluster').mode().first()))
            #     .alias('cluster_mode_freq'),
            cluster_mode_count=pl.col('cluster')
            .eq(pl.col('cluster').mode().first())
            .sum(),
            latent_dist_mean=pl.col('latent_dist').drop_nans().mean(),
            latent_dist_med=pl.col('latent_dist').drop_nans().median(),
        )
        .with_columns(
            pl.date(pl.col('year'), pl.col('month'), 1),
            cluster_mode_freq=pl.col('cluster_mode_count').truediv(pl.col('n_plays')),
        )
        .join(group_agg, on=['year', 'month'])
        .sort(['year', 'month'])
    )


print('grouping plays into sessions...')
df_stats_all_months = get_df_stats_all_months()
df_cluster_stats = get_cluster_stats(df_sessions)

df_stats_all_months.sink_parquet(DATA_DIR / 'df_stats_all_months.parquet')
df_sessions.sink_parquet(DATA_DIR / 'df_sessions.parquet')
df_cluster_stats.sink_parquet(DATA_DIR / 'df_cluster_stats.parquet')


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
