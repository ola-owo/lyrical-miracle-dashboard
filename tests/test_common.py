"""Tests for dashboard.common"""

import polars as pl

from dashboard.common import make_df_cluster_labels


def test_make_df_cluster_labels_columns():
    labels = make_df_cluster_labels(3)
    assert labels.columns == ['cluster', 'cluster_label']
    assert labels['cluster'].to_list() == [0, 1, 2]
    assert labels['cluster_label'].to_list() == ['A', 'B', 'C']
    assert labels.schema == {'cluster': pl.Int64, 'cluster_label': pl.String}


def test_make_df_cluster_labels_empty():
    assert make_df_cluster_labels(0).is_empty()


def test_make_df_cluster_labels_full_alphabet():
    labels = make_df_cluster_labels(26)
    assert labels['cluster_label'].first() == 'A'
    assert labels['cluster_label'].last() == 'Z'
