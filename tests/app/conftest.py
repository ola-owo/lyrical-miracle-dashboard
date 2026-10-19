"""Fixtures for Streamlit AppTest-based page tests"""

from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from dashboard import albumart, common, database, vector_search

REPO_ROOT = Path(__file__).resolve().parents[2]


class AppHarness:
    """Runs app pages through AppTest with all I/O seams patched"""

    def __init__(self, *, synth, timeout_popup: Mock, text_search: Mock):
        self.synth = synth
        self.timeout_popup = timeout_popup
        self.text_search = text_search

    def run_page(self, page: str, timeout: float = 15.0) -> AppTest:
        at = AppTest.from_file(REPO_ROOT / page, default_timeout=timeout)
        at.run()
        return at


@pytest.fixture
def install_app_patches(monkeypatch):
    """
    Patch the app's BigQuery/LastFM/Gemini seams and return an `AppHarness`.

    `synth` only needs the frame attributes that stand in for queried tables:
    `centroids`, `embeddings_clustered`, `song_matches`, `search_results`,
    and `search_song_info`.
    """

    def _install(synth, timeout: float = 15.0) -> AppHarness:
        def fake_db_read_table(tbl, dset=None, cols=None):
            if 'lyrics_cluster_centroids' in tbl:
                return synth.centroids
            raise AssertionError(f'unexpected db_read_table call: {tbl}')

        def fake_db_read_query(query, params=None):
            if 'genius.lyrics_clusters' in query:
                return synth.embeddings_clustered
            if 'genius.songs' in query:
                return synth.song_matches
            raise AssertionError(f'unexpected db_read_query call: {query}')

        monkeypatch.setattr(database, 'db_read_table', fake_db_read_table)
        monkeypatch.setattr(database, 'db_read_query', fake_db_read_query)
        monkeypatch.setattr(
            database,
            'get_bq_client',
            Mock(side_effect=AssertionError('BigQuery must not be called in tests')),
        )
        monkeypatch.setattr(
            albumart, 'get_genius_img', lambda g_id: f'https://img.example/{g_id}'
        )
        text_search = Mock(return_value=synth.search_results)
        monkeypatch.setattr(vector_search, 'text_search', text_search)
        monkeypatch.setattr(
            vector_search,
            'transform_search_res',
            Mock(return_value=synth.search_song_info),
        )
        monkeypatch.setattr(
            vector_search,
            'make_client',
            Mock(side_effect=AssertionError('Gemini must not be called in tests')),
        )
        timeout_popup = Mock()
        monkeypatch.setattr(common, 'timeout_popup', timeout_popup)

        return AppHarness(
            synth=synth, timeout_popup=timeout_popup, text_search=text_search
        )

    return _install


@pytest.fixture
def app_harness(monkeypatch, synth, synth_data_dir, install_app_patches):
    # pages read `Path('data/')`, so run them from a directory with synthetic data
    monkeypatch.chdir(synth_data_dir.parent)
    return install_app_patches(synth)
