"""Tests for dashboard.precompute"""

import datetime

import polars as pl
import pytest

from dashboard import precompute

_EMPTY_PLAYS = pl.DataFrame(
    schema={
        'dt': pl.Datetime('us', 'UTC'),
        'id': pl.String,
        'song': pl.String,
        'album': pl.String,
        'artist': pl.String,
    }
)


def _tables(**overrides) -> precompute.Tables:
    """Build `Tables` with empty frames for everything a test doesn't need"""
    base = {
        'plays': _EMPTY_PLAYS,
        'scrobbles': pl.DataFrame(
            schema={
                'dt': pl.Datetime('us', 'UTC'),
                'song': pl.String,
                'artist': pl.String,
                'album': pl.String,
            }
        ),
        'lyrics_search_vecs': pl.DataFrame(),
        'lyrics_big5': pl.DataFrame(),
        'spotify_genius_matches': pl.DataFrame(
            schema={'id': pl.String, 'g_id': pl.Int64}
        ),
        'lastfm_genius_matches': pl.DataFrame(
            schema={'song': pl.String, 'artist': pl.String, 'g_id': pl.Int64}
        ),
        'lyrics_clusters': pl.DataFrame(schema={'id': pl.Int64, 'cluster': pl.Int64}),
        'cluster_centroids': pl.DataFrame(schema={'cluster': pl.Int64}),
        'cluster_labels': pl.DataFrame(
            schema={'cluster': pl.Int64, 'cluster_label': pl.String}
        ),
    }
    base.update(overrides)
    return precompute.Tables(**base)


def test_build_plays_expanded_converts_timezone_and_bins_time():
    plays = pl.DataFrame(
        {
            'dt': [
                datetime.datetime(2024, 1, 6, 7, 30, tzinfo=datetime.UTC),
                datetime.datetime(2024, 1, 6, 14, 0, tzinfo=datetime.UTC),
            ],
            'id': ['111', '222'],
            'song': ['Song One', 'Song Two'],
            'album': ['Album I', 'Album I'],
            'artist': ['Artist X', 'Artist X'],
        }
    )
    matches = pl.DataFrame({'id': ['111', '222'], 'g_id': [101, 102]})

    df = precompute.build_plays_expanded(
        _tables(plays=plays, spotify_genius_matches=matches)
    ).collect()
    df = df.sort('dt')  # joins don't preserve row order

    assert df.columns == [
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
    ]
    assert df['g_id'].to_list() == [101, 102]
    # 07:30 UTC is 02:30 EST, so it falls in the night bin that resets at 3am
    assert df['time'].to_list() == [datetime.time(2, 30), datetime.time(9, 0)]
    assert df['dectime'].to_list() == [2.5, 9.0]
    assert df['timebin'].to_list() == ['night', 'midday']
    # 2024-01-06 is a Saturday
    assert df['weekday'].to_list() == [5, 5]
    assert df['is_weekend'].to_list() == [True, True]


def _plays_clustered() -> pl.DataFrame:
    return pl.DataFrame(
        {
            'dt': [
                datetime.datetime(2024, 1, 5, 8, 0),
                datetime.datetime(2024, 1, 5, 8, 30),
                datetime.datetime(2024, 1, 7, 9, 0),
            ],
            'g_id': [101, 102, 101],
            'song': ['Song One', 'Song Two', 'Song One'],
            'album': ['Album I', 'Album I', 'Album I'],
            'artist': ['Artist X', 'Artist X', 'Artist X'],
            'year': [2024, 2024, 2024],
            'month': [1, 1, 1],
            'timebin': ['morning', 'midday', 'midday'],
            'timediff': [
                None,
                datetime.timedelta(minutes=30),
                datetime.timedelta(hours=48, minutes=30),
            ],
            'cluster': [0, 1, 0],
        },
        schema_overrides={'timediff': pl.Duration('us')},
    )


def test_build_sessions_splits_on_long_gaps():
    labels = pl.DataFrame({'cluster': [0, 1], 'cluster_label': ['A', 'B']})

    df = (
        precompute.build_sessions(
            _tables(cluster_labels=labels), _plays_clustered().lazy()
        )
        .collect()
        .sort('dt')  # the cluster-label join doesn't preserve row order
    )

    # the 48h gap starts a new session
    assert df['session'].to_list() == [0, 0, 1]
    assert df['ses_len'].to_list() == [2, 2, 1]
    assert df['cluster_label'].to_list() == ['A', 'B', 'A']
    # no previous cluster within a session
    assert df['prev_cluster'].to_list() == [None, 0, None]


def test_main_skips_upload_without_bucket(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(precompute, 'run', lambda out_dir: seen.append(out_dir) or [])
    monkeypatch.setattr(precompute, 'upload_dir', lambda *args: seen.append(args))

    precompute.main(['--out-dir', str(tmp_path)])

    assert seen == [tmp_path]


def test_main_skips_upload_with_no_upload_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(precompute, 'run', lambda out_dir: [])
    monkeypatch.setattr(precompute, 'upload_dir', lambda *args: pytest.fail('uploaded'))

    precompute.main(
        ['--out-dir', str(tmp_path), '--bucket', 'my-bucket', '--no-upload']
    )


def test_main_uploads_when_bucket_given(tmp_path, monkeypatch):
    uploads = []
    monkeypatch.setattr(precompute, 'run', lambda out_dir: [out_dir / 'a.parquet'])
    monkeypatch.setattr(
        precompute,
        'upload_dir',
        lambda out_dir, bucket, prefix: uploads.append((out_dir, bucket, prefix)) or [],
    )

    precompute.main(
        ['--out-dir', str(tmp_path), '--bucket', 'my-bucket', '--prefix', 'p']
    )

    assert uploads == [(tmp_path, 'my-bucket', 'p')]
