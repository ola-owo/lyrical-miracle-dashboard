"""Precompute the dashboard's derived data files from BigQuery.

Writes one parquet/pickle file per dataset into a target directory
(`data/` by default), and optionally uploads them to a GCS bucket.
Each step is also importable on its own, for use from a notebook:

- `lyrics_search_vecs`: song ids and lyric-embeddings
- `plays_expanded`: song plays table with extra computed cols
- `plays_clustered`: plays_expanded with cluster assignments
- `df_sessions`: session-level aggregates
- `df_stats_all_months`: month-level session aggregates
- `df_cluster_stats`: cluster-level aggregates
- `df_cluster_per_month`: cluster/month level aggregates
- `ses_graph_full`: cluster graph spanning the entire timespan
- `big5`: month-level average big5 scores
"""

import argparse
import gzip
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import polars as pl

from dashboard.common import (
    BIG5_TRAITS,
    BIG5_TRAITS_NEG,
    BIG5_TRAITS_POS,
    BIG5_TRAITS_SHORT,
    DATA_DIR,
    EMBEDDING_DIM,
    SES_MAX_GAP,
    TIME_BIN_BOUNDARIES,
    TIME_BIN_LABELS,
    TIME_ZONE,
    make_df_cluster_labels,
)
from dashboard.database import db_read_table
from dashboard.graphs import build_cluster_graph
from dashboard.storage import upload_dir


@dataclass(frozen=True)
class Tables:
    """The BigQuery tables this pipeline reads from."""

    plays: pl.DataFrame
    scrobbles: pl.DataFrame
    lyrics_search_vecs: pl.DataFrame
    lyrics_big5: pl.DataFrame
    spotify_genius_matches: pl.DataFrame
    lastfm_genius_matches: pl.DataFrame
    lyrics_clusters: pl.DataFrame
    cluster_centroids: pl.DataFrame
    cluster_labels: pl.DataFrame

    @property
    def centroids(self) -> pl.DataFrame:
        """Cluster centroids joined onto their labels."""
        return self.cluster_centroids.join(self.cluster_labels, 'cluster', 'left')


def load_tables() -> Tables:
    """Read every BigQuery table needed by the pipeline."""
    plays = db_read_table(
        'spotify.streams',
        cols=[
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
    scrobbles = db_read_table(
        'lastfm.scrobbles', cols=['dt', 'song', 'artist', 'album']
    )
    lyrics_search_vecs = db_read_table('genius.lyrics_embed').with_columns(
        pl.col('embedding').cast(pl.Array(pl.Float32, EMBEDDING_DIM))
    )
    lyrics_big5 = db_read_table('genius.lyrics_big5').select(
        pl.col('id'), pl.concat_arr('opn', 'con', 'ext', 'agr', 'neu').alias('outputs')
    )
    spotify_genius_matches = db_read_table('spotify.genius_matches')
    lastfm_genius_matches = db_read_table('lastfm.genius_matches').drop_nulls('g_id')
    lyrics_clusters = db_read_table('genius.lyrics_clusters')
    cluster_centroids = db_read_table('genius.lyrics_cluster_centroids').sort('cluster')
    cluster_labels = make_df_cluster_labels(cluster_centroids.height)

    return Tables(
        plays=plays,
        scrobbles=scrobbles,
        lyrics_search_vecs=lyrics_search_vecs,
        lyrics_big5=lyrics_big5,
        spotify_genius_matches=spotify_genius_matches,
        lastfm_genius_matches=lastfm_genius_matches,
        lyrics_clusters=lyrics_clusters,
        cluster_centroids=cluster_centroids,
        cluster_labels=cluster_labels,
    )


def build_plays_matched(tables: Tables) -> pl.LazyFrame:
    """Last.fm and Spotify plays, with matching genius song ids."""
    scrobbles_matched = (
        tables.scrobbles.lazy()
        .join(
            tables.lastfm_genius_matches.lazy().select('song', 'artist', 'g_id'),
            on=('song', 'artist'),
            how='left',
        )
        .select('dt', 'song', 'album', 'artist', 'g_id')
    )
    spotify_matched = (
        tables.plays.lazy()
        .join(
            tables.spotify_genius_matches.lazy().select('id', 'g_id'),
            on='id',
            how='left',
        )
        .select('dt', 'song', 'album', 'artist', 'g_id')
    )
    return pl.concat((spotify_matched, scrobbles_matched))


def build_plays_expanded(tables: Tables) -> pl.LazyFrame:
    """`plays_matched` with expanded date/time info.

    Time bins are 6h bins that reset at 3am:
    - [3:00 - 9:00) = morning
    - [9:00 - 15:00) = midday
    - [15:00 - 21:00) = evening
    - [21:00 - 3:00) = night
    """
    return (
        build_plays_matched(tables)
        .with_columns(pl.col('dt').dt.convert_time_zone(TIME_ZONE))
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
            .bin_intervals(TIME_BIN_BOUNDARIES, labels=TIME_BIN_LABELS)
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


def build_plays_clustered(tables: Tables, plays_expanded: pl.LazyFrame) -> pl.LazyFrame:
    """`plays_expanded` with cluster labels and time deltas between plays."""
    return (
        tables.lyrics_clusters.lazy()
        .select('id', 'cluster')
        # merge with song plays
        .join(plays_expanded, left_on='id', right_on='g_id', how='right')
        .sort('dt')
        .with_columns(
            timediff=pl.col('dt').diff(),
            month=pl.col('dt').dt.month(),
            year=pl.col('dt').dt.year(),
        )
    )


def build_sessions(tables: Tables, plays_clustered: pl.LazyFrame) -> pl.LazyFrame:
    """`plays_clustered` grouped into sessions.

    A **session** is a continuous listening period with at most 24h
    between songs.
    """
    return (
        plays_clustered.with_columns(
            session=pl.col('timediff')
            .fill_null(pl.duration())
            .gt(SES_MAX_GAP)
            .cum_sum()
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
        .join(tables.cluster_labels.lazy(), 'cluster', 'left')
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


def build_cluster_stats(tables: Tables, df_sessions: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate stats per cluster."""
    return (
        df_sessions.group_by('cluster')
        .agg(
            n_plays=pl.len(),
            n_unique_plays=pl.col('g_id').unique().len(),
            n_sessions=pl.col('session').unique().len(),
            top_song_id=pl.col('g_id').mode().first(),
        )
        .sort('cluster')
        .with_columns(freq=(pl.col('n_plays') / pl.col('n_plays').sum()))
        .join(tables.centroids.lazy(), on='cluster', how='full', coalesce=True)
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


def build_stats_all_months(df_sessions: pl.LazyFrame) -> pl.LazyFrame:
    """Month-level stats: play/session counts, session aggregates,
    cluster mode, and cluster diversity."""
    session_group_agg = (
        df_sessions.group_by(['year', 'month', 'cluster'])
        .agg(pl.len())
        .with_columns(p=pl.col('len') / pl.col('len').sum().over(['year', 'month']))
        .group_by(['year', 'month'])
        .agg(
            gini=1.0 - pl.col('p').pow(2).sum(),
            shannon=(-pl.col('p') * pl.col('p').log(2)).sum(),
            berger=pl.col('p').max().pow(-1),
        )
    )
    return (
        df_sessions.group_by(['year', 'month'])
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
        .join(session_group_agg, on=['year', 'month'])
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


def build_cluster_per_month(
    tables: Tables,
    plays_clustered: pl.LazyFrame,
    df_stats_all_months: pl.LazyFrame,
) -> pl.LazyFrame:
    """Each cluster's frequency distribution per year/month."""
    return (
        plays_clustered.group_by(['year', 'month', 'cluster'])
        .agg(pl.len().alias('n_cluster_plays'))
        .join(df_stats_all_months, on=['year', 'month'])
        .join(tables.cluster_labels.lazy(), on='cluster')
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


def build_big5(tables: Tables, df_sessions: pl.LazyFrame) -> pl.LazyFrame:
    """Trait scores per year/month.

    - one row per trait, per year/month
    - logit = raw output from model
    - score = tanh[logit] (range -1 to 1)
    - score_pct = score * 100
    """
    big5 = (
        df_sessions.join(tables.lyrics_big5.lazy(), left_on='g_id', right_on='id')
        .with_columns(date=pl.date(pl.col('year'), pl.col('month'), 1))
        .select('outputs', 'date')
    )
    df_traits = pl.LazyFrame(
        {
            'trait': BIG5_TRAITS,
            'trait_short': BIG5_TRAITS_SHORT,
            'trait_pos': BIG5_TRAITS_POS,
            'trait_neg': BIG5_TRAITS_NEG,
        }
    ).with_row_index('n')
    return (
        big5.with_columns(n=range(5))
        .explode('n', 'outputs', empty_as_null=True)
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


def build_graph(df_sessions: pl.LazyFrame) -> nx.DiGraph:
    """Session graph spanning the full timeline.

    Nodes are clusters, and edges are weighted by within-session
    transition frequency.
    """
    return build_cluster_graph(df_sessions)


def _log(message: str) -> None:
    print(message, flush=True)


def run(out_dir: Path = DATA_DIR) -> list[Path]:
    """Compute every derived data file, and write it to `out_dir`.

    Args:
        out_dir: directory to write the parquet/pickle files into

    Returns: the paths written, in creation order
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    _log('loading BigQuery tables...')
    tables = load_tables()

    _log('writing lyrics embedding vectors...')
    search_vecs_path = out_dir / 'lyrics_search_vecs.parquet'
    tables.lyrics_search_vecs.write_parquet(search_vecs_path)

    _log('building plays tables...')
    plays_expanded = build_plays_expanded(tables)
    plays_clustered = build_plays_clustered(tables, plays_expanded)
    plays_expanded_path = out_dir / 'plays_expanded.parquet'
    plays_clustered_path = out_dir / 'plays_clustered.parquet'
    plays_expanded.sink_parquet(plays_expanded_path)
    plays_clustered.sink_parquet(plays_clustered_path)

    _log('grouping plays into sessions...')
    df_sessions = build_sessions(tables, plays_clustered)
    df_cluster_stats = build_cluster_stats(tables, df_sessions)
    df_stats_all_months = build_stats_all_months(df_sessions)
    df_cluster_per_month = build_cluster_per_month(
        tables, plays_clustered, df_stats_all_months
    )
    sessions_path = out_dir / 'df_sessions.parquet'
    cluster_stats_path = out_dir / 'df_cluster_stats.parquet'
    stats_path = out_dir / 'df_stats_all_months.parquet'
    cluster_month_path = out_dir / 'df_cluster_per_month.parquet'
    df_stats_all_months.sink_parquet(stats_path)
    df_sessions.sink_parquet(sessions_path)
    df_cluster_stats.sink_parquet(cluster_stats_path)
    df_cluster_per_month.sink_parquet(cluster_month_path)

    _log('building cluster graph...')
    graph_path = out_dir / 'ses_graph_full.pkl.gz'
    with gzip.open(graph_path, 'wb') as graph_file:
        pickle.dump(build_graph(df_sessions), graph_file)

    _log('computing big5 scores...')
    big5_path = out_dir / 'big5.parquet'
    build_big5(tables, df_sessions).sink_parquet(big5_path)

    return [
        search_vecs_path,
        plays_expanded_path,
        plays_clustered_path,
        sessions_path,
        cluster_stats_path,
        stats_path,
        cluster_month_path,
        graph_path,
        big5_path,
    ]


def main(argv: list[str] | None = None) -> None:
    """Run the pipeline from the command line."""
    parser = argparse.ArgumentParser(description='Precompute the dashboard data files.')
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path(os.environ.get('DATA_DIR', str(DATA_DIR))),
        help='directory to write the data files into (default: $DATA_DIR or data/)',
    )
    parser.add_argument(
        '--bucket',
        default=os.environ.get('GCS_BUCKET'),
        help='GCS bucket to upload to (default: $GCS_BUCKET; skips upload if unset)',
    )
    parser.add_argument(
        '--prefix',
        default=os.environ.get('GCS_PREFIX', ''),
        help='object-name prefix within the bucket (default: $GCS_PREFIX)',
    )
    parser.add_argument(
        '--no-upload',
        action='store_true',
        help='write files locally, without uploading',
    )
    args = parser.parse_args(argv)

    paths = run(args.out_dir)
    _log(f'wrote {len(paths)} files to {args.out_dir}')

    if args.no_upload or not args.bucket:
        if not args.no_upload:
            _log('no bucket configured, skipping upload')
        return

    _log(f'uploading to gs://{args.bucket}/{args.prefix}...')
    for uri in upload_dir(args.out_dir, args.bucket, args.prefix):
        _log(f'uploaded {uri}')


if __name__ == '__main__':
    main()
