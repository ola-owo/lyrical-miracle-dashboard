"""Tests for dashboard.database"""

import datetime
from unittest.mock import Mock

import pendulum as pn
import polars as pl
import pyarrow as pa
import pytest
from google.cloud.bigquery import ArrayQueryParameter, ScalarQueryParameter
from streamlit.errors import StreamlitSecretNotFoundError
from streamlit.runtime.secrets import AttrDict

from dashboard import database


class _FakeQueryJob:
    def __init__(self, table: pa.Table):
        self._table = table

    def to_arrow(self) -> pa.Table:
        return self._table


class _FakeBQClient:
    """Records queries and returns a canned arrow table"""

    def __init__(self, table: pa.Table):
        self._table = table
        self.calls = []

    def query(self, query, job_config=None):
        self.calls.append((query, job_config))
        return _FakeQueryJob(self._table)


@pytest.fixture
def fake_bq_client(monkeypatch):
    client = _FakeBQClient(pa.table({'a': [1, 2], 'b': ['x', 'y']}))
    monkeypatch.setattr(database, 'get_bq_client', lambda: client)
    return client


###
### Query parameter builders
###
@pytest.mark.parametrize(
    ('value', 'bq_type'),
    [
        (True, 'BOOL'),
        (1, 'INT64'),
        (1.5, 'FLOAT64'),
        ('a', 'STRING'),
        (b'a', 'BYTES'),
        (pn.Date(2024, 1, 5), 'DATE'),
        (pn.DateTime(2024, 1, 5, 8, 30), 'TIMESTAMP'),
        (pn.Time(8, 30), 'TIME'),
    ],
)
def test_bq_scalar_param_types(value, bq_type):
    param = database._bq_scalar_param('p', value)
    assert isinstance(param, ScalarQueryParameter)
    assert param.name == 'p'
    assert param.type_ == bq_type
    assert param.value == value


def test_bq_scalar_param_unmapped_type():
    # NOTE: only exact types are mapped, so stdlib dates are rejected
    with pytest.raises(TypeError, match='No BigQuery type mapping'):
        database._bq_scalar_param('p', datetime.date(2024, 1, 5))


def test_bq_array_param():
    param = database._bq_array_param('p', ['x', 'y'])
    assert isinstance(param, ArrayQueryParameter)
    assert param.name == 'p'
    assert param.array_type == 'STRING'
    assert param.values == ['x', 'y']


def test_bq_array_param_accepts_tuple():
    param = database._bq_array_param('p', (1, 2))
    assert param.array_type == 'INT64'
    assert param.values == [1, 2]


def test_bq_array_param_empty_values_raises():
    with pytest.raises(ValueError, match='empty list'):
        database._bq_array_param('ids', [])


def test_bq_array_param_mixed_types_raises():
    with pytest.raises(TypeError, match='mixed element types'):
        database._bq_array_param('ids', [1, 'a'])


def test_bq_array_param_unmapped_element_type_raises():
    with pytest.raises(TypeError, match='array element type'):
        database._bq_array_param('ids', [datetime.date(2024, 1, 1)])


def test_bq_param_dispatches_lists_to_arrays():
    assert isinstance(database._bq_param('p', [1]), ArrayQueryParameter)
    assert isinstance(database._bq_param('p', (1,)), ArrayQueryParameter)
    assert isinstance(database._bq_param('p', 1), ScalarQueryParameter)


###
### BigQuery table/query reads
###
def test_db_read_table_two_part_name(fake_bq_client):
    df = database.db_read_table('genius.songs')
    assert df.equals(pl.DataFrame({'a': [1, 2], 'b': ['x', 'y']}))
    assert fake_bq_client.calls[0][0] == 'SELECT * FROM `genius.songs`'


def test_db_read_table_with_dset(fake_bq_client):
    database.db_read_table('songs', dset='genius')
    assert fake_bq_client.calls[0][0] == 'SELECT * FROM `genius.songs`'


def test_db_read_table_with_cols(fake_bq_client):
    database.db_read_table('genius.songs', cols=['title', 'url'])
    assert fake_bq_client.calls[0][0] == 'SELECT `title`, `url` FROM `genius.songs`'


@pytest.mark.parametrize('tbl', ['songs', 'a.b.c.d'])
def test_db_read_table_bad_name_raises(tbl):
    with pytest.raises(ValueError, match='dataset'):
        database.db_read_table(tbl)


def test_db_read_table_dset_with_qualified_name_raises():
    with pytest.raises(ValueError, match='dataset'):
        database.db_read_table('genius.songs', dset='other')


def test_db_read_table_three_part_name(fake_bq_client):
    database.db_read_table('my-project.genius.songs')
    assert fake_bq_client.calls[0][0] == 'SELECT * FROM `my-project.genius.songs`'


def test_db_read_query_without_params(fake_bq_client):
    df = database.db_read_query('SELECT 1')
    query, job_config = fake_bq_client.calls[0]
    assert query == 'SELECT 1'
    assert job_config is None
    assert df.height == 2


def test_db_read_query_with_params(fake_bq_client):
    database.db_read_query(
        'SELECT * FROM t WHERE id = @id AND name IN UNNEST(@names)',
        {'id': 5, 'names': ['a', 'b']},
    )
    _, job_config = fake_bq_client.calls[0]
    params = {p.name: p for p in job_config.query_parameters}
    assert isinstance(params['id'], ScalarQueryParameter)
    assert params['id'].value == 5
    assert isinstance(params['names'], ArrayQueryParameter)
    assert params['names'].values == ['a', 'b']


def test_db_read_query_unsupported_param_type_raises():
    with pytest.raises(TypeError, match='No BigQuery type mapping'):
        database.db_read_query('SELECT 1', {'x': object()})


def test_get_bq_client_uses_service_account(fake_secrets, monkeypatch):
    creds = object()
    creds_factory = Mock()
    creds_factory.from_service_account_info.return_value = creds
    client_factory = Mock()
    monkeypatch.setattr(database.service_account, 'Credentials', creds_factory)
    monkeypatch.setattr(database.bigquery, 'Client', client_factory)

    client = database.get_bq_client()

    assert client is client_factory.return_value
    creds_factory.from_service_account_info.assert_called_once_with(
        fake_secrets['gcp_service_account']
    )
    client_factory.assert_called_once_with(credentials=creds)


def test_get_bq_client_falls_back_to_adc_without_secret(monkeypatch):
    creds_factory = Mock()
    client_factory = Mock()
    monkeypatch.setattr(database.st, 'secrets', {})
    monkeypatch.setattr(database.service_account, 'Credentials', creds_factory)
    monkeypatch.setattr(database.bigquery, 'Client', client_factory)

    client = database.get_bq_client()

    assert client is client_factory.return_value
    creds_factory.from_service_account_info.assert_not_called()
    client_factory.assert_called_once_with()


class _MissingSecrets:
    def __getitem__(self, key):
        raise StreamlitSecretNotFoundError('no secrets found')


def test_get_bq_client_falls_back_to_adc_when_secrets_missing(monkeypatch):
    creds_factory = Mock()
    client_factory = Mock()
    monkeypatch.setattr(database.st, 'secrets', _MissingSecrets())
    monkeypatch.setattr(database.service_account, 'Credentials', creds_factory)
    monkeypatch.setattr(database.bigquery, 'Client', client_factory)

    client = database.get_bq_client()

    assert client is client_factory.return_value
    creds_factory.from_service_account_info.assert_not_called()
    client_factory.assert_called_once_with()


###
### DuckDB connection
###
def test_duckdb_connection_connects_with_database_kwarg():
    cxn = database.DuckDBConnection('duckdb', database=':memory:')
    result = cxn.query('SELECT 42 AS answer')
    assert result.equals(pl.DataFrame({'answer': [42]}))


def test_duckdb_connection_reads_database_from_secrets(monkeypatch):
    secrets = AttrDict(
        {'connections': AttrDict({'duckdb': AttrDict({'database': ':memory:'})})}
    )
    singleton = Mock()
    singleton.load_if_toml_exists.return_value = True
    singleton.get.side_effect = secrets.get
    singleton.to_dict.return_value = {}
    monkeypatch.setattr(
        'streamlit.connections.base_connection.secrets_singleton', singleton
    )

    cxn = database.DuckDBConnection('duckdb')

    assert cxn.query('SELECT 1 AS x').equals(pl.DataFrame({'x': [1]}))


def test_duckdb_connection_reads_tables():
    cxn = database.DuckDBConnection('duckdb', database=':memory:')
    with cxn.cursor() as cur:
        cur.execute('CREATE TABLE t AS SELECT * FROM (VALUES (1), (2)) AS v(x)')

    assert cxn.table('t').equals(pl.DataFrame({'x': [1, 2]}))
