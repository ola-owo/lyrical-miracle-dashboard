"""Shared fixtures and synthetic data builders for the test suite"""

import datetime
import gzip
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pendulum as pn
import polars as pl
import pytest
import streamlit

from dashboard.common import (
    BIG5_TRAITS,
    BIG5_TRAITS_NEG,
    BIG5_TRAITS_POS,
    BIG5_TRAITS_SHORT,
    EMBEDDING_DIM,
    TIME_BIN_BOUNDARIES,
    TIME_BIN_LABELS,
    make_df_cluster_labels,
)


@pytest.fixture(autouse=True)
def clear_streamlit_caches():
    """Clear Streamlit caches around every test, so mock results don't leak"""
    streamlit.cache_data.clear()
    streamlit.cache_resource.clear()
    yield
    streamlit.cache_data.clear()
    streamlit.cache_resource.clear()


@pytest.fixture
def fake_secrets(monkeypatch):
    """
    Replace `st.secrets` with dummy values.

    Tests must never read the real `.streamlit/secrets.toml`.
    """
    secrets = {
        'gcp_service_account': {
            'type': 'service_account',
            'project_id': 'test-project',
        },
        'lastfm': {'user': 'test-user', 'key': 'test-key', 'secret': 'test-secret'},
        'gemini': {'api_key': 'test-api-key'},
        'connections': {'duckdb': {'database': ':memory:'}},
    }
    monkeypatch.setattr(streamlit, 'secrets', secrets)
    return secrets


###
### Synthetic app data
###
# Mirrors the columns and types that notebooks/precompute.py writes to `data/`

N_CLUSTERS = 3

_SONG_INFO = [
    {'g_id': 101, 'song': 'Song One', 'artist': 'Artist X', 'album': 'Album I'},
    {'g_id': 102, 'song': 'Song Two', 'artist': 'Artist X', 'album': 'Album I'},
    {'g_id': 103, 'song': 'Song Three', 'artist': 'Artist Y', 'album': 'Album II'},
    {'g_id': 104, 'song': 'Song Four', 'artist': 'Artist Y', 'album': 'Album II'},
    {'g_id': 105, 'song': 'Song Five', 'artist': 'Artist Z', 'album': 'Album III'},
]

# session 1 (Jan 5): clusters 0 -> 1 -> 0
# session 2 (Jan 20): a single play, so it must not create graph edges
# session 3 (Feb 10): clusters 2 -> 0
_PLAYS = [
    {'session': 1, 'dt': pn.datetime(2024, 1, 5, 8, 0), 'cluster': 0, 'g_id': 101},
    {'session': 1, 'dt': pn.datetime(2024, 1, 5, 8, 30), 'cluster': 1, 'g_id': 102},
    {'session': 1, 'dt': pn.datetime(2024, 1, 5, 9, 0), 'cluster': 0, 'g_id': 101},
    {'session': 2, 'dt': pn.datetime(2024, 1, 20, 21, 0), 'cluster': 1, 'g_id': 102},
    {'session': 3, 'dt': pn.datetime(2024, 2, 10, 14, 0), 'cluster': 2, 'g_id': 103},
    {'session': 3, 'dt': pn.datetime(2024, 2, 10, 14, 30), 'cluster': 0, 'g_id': 101},
]

# songs 104/105 are never played, to exercise joins that must drop unmatched rows
_EMBEDDINGS = [
    {'g_id': 101, 'cluster': 0, 'centroid_dist': 0.10},
    {'g_id': 102, 'cluster': 1, 'centroid_dist': 0.20},
    {'g_id': 103, 'cluster': 2, 'centroid_dist': 0.30},
    {'g_id': 104, 'cluster': 1, 'centroid_dist': 0.05},
    {'g_id': 105, 'cluster': 2, 'centroid_dist': 0.40},
]

_BIG5_LOGITS = {
    pn.Date(2024, 1, 1): [0.5, -0.2, 0.1, 0.8, -0.6],
    pn.Date(2024, 2, 1): [-0.3, 0.4, -0.5, 0.2, 0.7],
}


@dataclass(frozen=True)
class SynthData:
    """Synthetic stand-ins for the app's precomputed and BigQuery tables"""

    sessions: pl.DataFrame
    plays_expanded: pl.DataFrame
    plays_clustered: pl.DataFrame
    big5: pl.DataFrame
    stats_months: pl.DataFrame
    cluster_per_month: pl.DataFrame
    centroids: pl.DataFrame
    embeddings_clustered: pl.DataFrame
    song_matches: pl.DataFrame
    cluster_graph: nx.DiGraph
    search_vecs: pl.DataFrame
    search_results: pl.DataFrame
    search_song_info: pl.DataFrame


def _plays_frame() -> pl.DataFrame:
    plays = (
        pl.DataFrame(_PLAYS, schema_overrides={'dt': pl.Datetime('us')})
        .join(pl.DataFrame(_SONG_INFO), on='g_id', how='left')
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
        .with_columns(
            is_weekend=pl.col('weekday') >= 5,
            timebin=(
                pl.col('dectime')
                .sub(3)
                .mod(24)
                .bin_intervals(TIME_BIN_BOUNDARIES, labels=TIME_BIN_LABELS)
                .cast(pl.Enum(TIME_BIN_LABELS))
            ),
        )
    )
    return plays.select(
        'session',
        'dt',
        'g_id',
        'song',
        'album',
        'artist',
        'cluster',
        'year',
        'month',
        'day',
        'weekday',
        'time',
        'is_weekend',
        'dectime',
        'timebin',
    )


def _sessions_frame(plays: pl.DataFrame) -> pl.DataFrame:
    return (
        plays.with_columns(timediff=pl.col('dt').diff())
        .with_columns(
            prev_cluster=pl.col('cluster').shift(1).over('session'),
            ses_start=pl.col('dt').min().over('session'),
            ses_end=pl.col('dt').max().over('session'),
            ses_len=pl.len().over('session'),
            n_within_ses=pl.col('dt').cum_count().over('session'),
        )
        .with_columns(ses_dur=pl.col('ses_end') - pl.col('ses_start'))
        .join(make_df_cluster_labels(N_CLUSTERS), on='cluster', how='left')
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


def _big5_frame() -> pl.DataFrame:
    rows = []
    for date, logits in _BIG5_LOGITS.items():
        for logit, trait, short, pos, neg in zip(
            logits, BIG5_TRAITS, BIG5_TRAITS_SHORT, BIG5_TRAITS_POS, BIG5_TRAITS_NEG
        ):
            score = math.tanh(logit)
            rows.append(
                {
                    'date': date,
                    'logit': logit,
                    'trait': trait,
                    'trait_short': short,
                    'trait_pos': pos,
                    'trait_neg': neg,
                    'trait_desc': pos if logit >= 0 else neg,
                    'score': score,
                    'score_pct': abs(score) * 100,
                }
            )
    return pl.DataFrame(rows)


def _stats_months_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            'date': pl.Series(
                [datetime.date(2024, 1, 1), datetime.date(2024, 2, 1)], dtype=pl.Date
            ),
            'year': [2024, 2024],
            'month': [1, 2],
            'n_plays': [4, 2],
            'n_sessions': [2, 1],
            'ses_len_median': [2.0, 2.0],
            'ses_len_max': [3, 2],
            'cluster_mode': [0, 2],
            'cluster_mode_count': [2, 1],
            'cluster_mode_freq': [0.5, 0.5],
            'gini': [0.5, 0.5],
            'shannon': [1.0, 1.0],
            'berger': [2.0, 2.0],
        }
    )


def _cluster_per_month_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            'year': [2024, 2024, 2024, 2024],
            'month': [1, 1, 2, 2],
            'date': pl.Series(
                [datetime.date(2024, 1, 1)] * 2 + [datetime.date(2024, 2, 1)] * 2,
                dtype=pl.Date,
            ),
            'cluster': [0, 1, 0, 2],
            'cluster_label': ['A', 'B', 'A', 'C'],
            'n_cluster_plays': [2, 2, 1, 1],
            'cluster_freq': [0.5, 0.5, 0.5, 0.5],
        }
    )


def _cluster_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(
        [
            (0, {'name': 'A', 'weight': 3}),
            (1, {'name': 'B', 'weight': 2}),
            (2, {'name': 'C', 'weight': 1}),
        ]
    )
    g.add_weighted_edges_from([(0, 1, 1), (1, 0, 1), (2, 0, 1)])
    return g


def _search_vecs() -> pl.DataFrame:
    """Three orthogonal unit vectors, one per song id"""
    emb = np.zeros((3, EMBEDDING_DIM), dtype='float32')
    emb[0, 0] = 1.0
    emb[1, 1] = 1.0
    emb[2, 2] = 1.0
    embedding = pl.Series(emb.tolist(), dtype=pl.Array(pl.Float32, EMBEDDING_DIM))
    return pl.DataFrame({'id': [101, 102, 103], 'embedding': embedding})


def build_synth_data() -> SynthData:
    plays = _plays_frame()

    return SynthData(
        sessions=_sessions_frame(plays),
        plays_expanded=plays.drop('cluster', 'session'),
        plays_clustered=plays.drop('session'),
        big5=_big5_frame(),
        stats_months=_stats_months_frame(),
        cluster_per_month=_cluster_per_month_frame(),
        centroids=pl.DataFrame(
            {'cluster': [0, 1, 2], 'centroid': [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]}
        ),
        embeddings_clustered=pl.DataFrame(_EMBEDDINGS),
        song_matches=pl.DataFrame(_SONG_INFO),
        cluster_graph=_cluster_graph(),
        search_vecs=_search_vecs(),
        search_results=pl.DataFrame(
            {'search_num': [0], 'id': [102], 'dist': [0.9]}
        ).with_row_index('rank'),
        search_song_info=pl.DataFrame(
            {
                'song': ['Song Two'],
                'artist': ['Artist X'],
                'album': ['Album I'],
                'release_date': ['2024-01-01'],
                'url': ['https://genius.example/song-two'],
            }
        ),
    )


def write_synth_data(data_dir: Path, synth: SynthData) -> None:
    """Write the synthetic data to `data_dir`, with the same filenames as the app"""
    data_dir.mkdir(parents=True, exist_ok=True)
    synth.sessions.write_parquet(data_dir / 'df_sessions.parquet')
    synth.plays_expanded.write_parquet(data_dir / 'plays_expanded.parquet')
    synth.plays_clustered.write_parquet(data_dir / 'plays_clustered.parquet')
    synth.big5.write_parquet(data_dir / 'big5.parquet')
    synth.stats_months.write_parquet(data_dir / 'df_stats_all_months.parquet')
    synth.cluster_per_month.write_parquet(data_dir / 'df_cluster_per_month.parquet')
    synth.search_vecs.write_parquet(data_dir / 'lyrics_search_vecs.parquet')
    with gzip.open(data_dir / 'ses_graph_full.pkl.gz', 'wb') as f:
        pickle.dump(synth.cluster_graph, f)


@pytest.fixture(scope='session')
def synth() -> SynthData:
    return build_synth_data()


@pytest.fixture
def synth_data_dir(tmp_path: Path, synth: SynthData) -> Path:
    data_dir = tmp_path / 'data'
    write_synth_data(data_dir, synth)
    return data_dir
