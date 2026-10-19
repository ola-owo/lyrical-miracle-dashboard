"""
Smoke tests running the pages against the real precomputed files in `data/`.

These catch schema drift between `notebooks/precompute.py` and the pages, which
the hand-built synthetic fixtures can't. They need no secrets or network: the
BigQuery seams are faked with frames derived from the real session data.
"""

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / 'data'

REQUIRED_FILES = [
    'df_sessions.parquet',
    'plays_expanded.parquet',
    'plays_clustered.parquet',
    'big5.parquet',
    'df_stats_all_months.parquet',
    'df_cluster_per_month.parquet',
    'ses_graph_full.pkl.gz',
]

pytestmark = pytest.mark.real_data


@pytest.fixture
def real_data_harness(monkeypatch, install_app_patches):
    missing = [name for name in REQUIRED_FILES if not (DATA_DIR / name).is_file()]
    if missing:
        pytest.skip(f'missing precomputed data files: {", ".join(missing)}')

    monkeypatch.chdir(REPO_ROOT)

    sessions = pl.read_parquet(DATA_DIR / 'df_sessions.parquet')
    clusters = sessions['cluster'].drop_nulls().unique().sort().to_list()
    g_ids = sessions['g_id'].drop_nulls().unique()
    synth = SimpleNamespace(
        centroids=pl.DataFrame(
            {'cluster': clusters, 'centroid': [[0.0, 0.0]] * len(clusters)}
        ),
        embeddings_clustered=pl.DataFrame(
            {'g_id': g_ids, 'cluster': 0, 'centroid_dist': 0.1}
        ),
        song_matches=pl.DataFrame(
            {
                'g_id': g_ids,
                'song': [f'Song {g_id}' for g_id in g_ids],
                'artist': 'Artist',
                'album': 'Album',
            }
        ),
        search_results=pl.DataFrame(
            {'search_num': [0], 'id': [g_ids[0]], 'dist': [0.9]}
        ).with_row_index('rank'),
        search_song_info=pl.DataFrame(
            {
                'song': ['Song'],
                'artist': ['Artist'],
                'album': ['Album'],
                'release_date': ['2024-01-01'],
                'url': ['https://genius.example/song'],
            }
        ),
    )
    return install_app_patches(synth, timeout=30.0)


EXPECTED_HEADERS = {
    'Home.py': 'The Lyrical Miracle',
    'pages/1_Monthlies.py': 'Clustered listening sessions',
    'pages/2_Timelines.py': 'Your Listening History',
    'pages/3_Search.py': "What's that song...? 🤔",
}


@pytest.mark.parametrize('page', list(EXPECTED_HEADERS))
def test_page_runs_with_real_data(real_data_harness, page):
    at = real_data_harness.run_page(page)

    assert not at.exception, [e.value for e in at.exception]
    rendered = [t.value for t in at.title] + [h.value for h in at.header]
    assert EXPECTED_HEADERS[page] in rendered
