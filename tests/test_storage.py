"""Tests for dashboard.storage"""

from pathlib import Path

import pytest

from dashboard import storage


class _FakeBlob:
    def __init__(self, name: str, store: dict[str, str]):
        self.name = name
        self._store = store

    def upload_from_filename(self, filename):
        self._store[self.name] = Path(filename).read_text()


class _FakeBucket:
    def __init__(self, store: dict[str, str]):
        self._store = store

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(name, self._store)


class _FakeClient:
    def __init__(self, store: dict[str, str]):
        self.store = store
        self.bucket_name = None

    def bucket(self, name: str) -> _FakeBucket:
        self.bucket_name = name
        return _FakeBucket(self.store)


@pytest.fixture
def fake_client(monkeypatch) -> _FakeClient:
    store: dict[str, str] = {}
    client = _FakeClient(store)
    monkeypatch.setattr(storage.storage, 'Client', lambda: client)
    return client


@pytest.fixture
def local_dir(tmp_path) -> Path:
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    (data_dir / 'a.parquet').write_text('a')
    (data_dir / 'b.parquet').write_text('b')
    (data_dir / 'nested').mkdir()
    (data_dir / 'nested' / 'c.parquet').write_text('c')
    return data_dir


def test_upload_dir_uploads_top_level_files(fake_client, local_dir):
    uris = storage.upload_dir(local_dir, 'my-bucket', 'dashboard-data')

    assert fake_client.bucket_name == 'my-bucket'
    assert fake_client.store == {
        'dashboard-data/a.parquet': 'a',
        'dashboard-data/b.parquet': 'b',
    }
    assert uris == [
        'gs://my-bucket/dashboard-data/a.parquet',
        'gs://my-bucket/dashboard-data/b.parquet',
    ]


def test_upload_dir_without_prefix(fake_client, local_dir):
    uris = storage.upload_dir(local_dir, 'my-bucket')

    assert sorted(fake_client.store) == ['a.parquet', 'b.parquet']
    assert uris[0] == 'gs://my-bucket/a.parquet'


@pytest.mark.parametrize(
    'prefix', ['dashboard-data/', '/dashboard-data/', '/dashboard-data']
)
def test_upload_dir_strips_slashes_from_prefix(fake_client, local_dir, prefix):
    uris = storage.upload_dir(local_dir, 'my-bucket', prefix)

    assert uris[0] == 'gs://my-bucket/dashboard-data/a.parquet'


def test_upload_dir_empty_dir(fake_client, tmp_path):
    empty = tmp_path / 'empty'
    empty.mkdir()

    assert storage.upload_dir(empty, 'my-bucket') == []
