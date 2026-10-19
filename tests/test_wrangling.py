"""Tests for dashboard.wrangling"""

import pendulum as pn
import polars as pl
import pytest

from dashboard.common import make_df_cluster_labels
from dashboard.wrangling import filter_df_by_month, get_cluster_stats

JAN = pn.Date(2024, 1, 1)
FEB = pn.Date(2024, 2, 1)


@pytest.fixture
def centroids(synth):
    return synth.centroids.join(
        make_df_cluster_labels(synth.centroids.height), on='cluster', how='left'
    )


###
### filter_df_by_month
###
def test_filter_df_by_month_none_returns_input(synth):
    assert filter_df_by_month(synth.sessions, None).equals(synth.sessions)


def test_filter_df_by_month_filters_on_year_and_month(synth):
    jan = filter_df_by_month(synth.sessions, JAN)
    assert jan.height == 4
    assert jan['month'].unique().to_list() == [1]
    assert jan['year'].unique().to_list() == [2024]


def test_filter_df_by_month_preserves_laziness(synth):
    jan = filter_df_by_month(synth.sessions.lazy(), JAN)
    assert isinstance(jan, pl.LazyFrame)
    assert jan.collect().height == 4


def test_filter_df_by_month_no_matches(synth):
    assert filter_df_by_month(synth.sessions, pn.Date(2024, 3, 1)).is_empty()


###
### get_cluster_stats
###
def test_get_cluster_stats_january(synth, centroids):
    jan = filter_df_by_month(synth.sessions, JAN)
    # NOTE: sort here because the full join in get_cluster_stats doesn't preserve order
    stats = get_cluster_stats(jan, centroids).collect().sort('cluster')

    assert stats['cluster'].to_list() == [0, 1, 2]
    by_cluster = {row['cluster']: row for row in stats.iter_rows(named=True)}

    # cluster 0: two plays of one song, all in session 1
    assert by_cluster[0]['n_plays'] == 2
    assert by_cluster[0]['n_unique_plays'] == 1
    assert by_cluster[0]['n_sessions'] == 1
    assert by_cluster[0]['top_song_id'] == 101
    assert by_cluster[0]['cluster_label'] == 'A'
    assert by_cluster[0]['centroid'] == [1.0, 0.0]
    assert by_cluster[0]['freq'] == pytest.approx(0.5)

    # cluster 1: one play in each of sessions 1 and 2
    assert by_cluster[1]['n_plays'] == 2
    assert by_cluster[1]['n_sessions'] == 2
    assert by_cluster[1]['freq'] == pytest.approx(0.5)

    # cluster 2: no plays this month, but kept by the full join
    assert by_cluster[2]['n_plays'] == 0
    assert by_cluster[2]['n_unique_plays'] == 0
    assert by_cluster[2]['n_sessions'] == 0
    assert by_cluster[2]['freq'] == 0
    assert by_cluster[2]['top_song_id'] is None
    assert by_cluster[2]['cluster_label'] == 'C'


def test_get_cluster_stats_february(synth, centroids):
    stats = get_cluster_stats(filter_df_by_month(synth.sessions, FEB), centroids)
    by_cluster = {row['cluster']: row for row in stats.collect().iter_rows(named=True)}

    assert by_cluster[0]['n_plays'] == 1
    assert by_cluster[2]['n_plays'] == 1
    assert by_cluster[1]['n_plays'] == 0
    assert by_cluster[0]['freq'] == pytest.approx(0.5)
