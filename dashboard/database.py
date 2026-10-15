"""Database I/O tools"""

from typing import Any

import duckdb
import pendulum as pn
import polars as pl
import streamlit as st
from google.cloud import bigquery
from google.cloud.bigquery import ArrayQueryParameter, ScalarQueryParameter
from google.oauth2 import service_account
from streamlit.connections import ExperimentalBaseConnection
from streamlit.runtime.caching import cache_data

BQ_TYPE_MAP = {
    bool: 'BOOL',
    int: 'INT64',
    float: 'FLOAT64',
    str: 'STRING',
    bytes: 'BYTES',
    pn.Date: 'DATE',
    pn.DateTime: 'TIMESTAMP',
    pn.Time: 'TIME',
}


def _bq_param(name, value) -> ScalarQueryParameter | ArrayQueryParameter:
    if isinstance(value, (list, tuple)):
        return _bq_array_param(name, value)
    else:
        return _bq_scalar_param(name, value)


def _bq_scalar_param(name, value) -> ScalarQueryParameter:
    """Build a ScalarQueryParameter, inferring BQ type from the Python value."""
    py_type = type(value)
    bq_type = BQ_TYPE_MAP.get(py_type)
    if not bq_type:
        raise TypeError(f'No BigQuery type mapping for {py_type}')
    return ScalarQueryParameter(name, bq_type, value)


def _bq_array_param(name, values) -> ArrayQueryParameter:
    """Build an ArrayQueryParameter, inferring BQ type from the Python value."""
    if not values:
        raise ValueError(
            f"Cannot infer array element type for empty list param '{name}'; "
            "pass the type explicitly, e.g. ArrayQueryParameter(name, 'STRING', [])"
        )

    element_types = {type(v) for v in values}
    if len(element_types) > 1:
        raise TypeError(
            f"Array param '{name}' has mixed element types: {element_types}. "
            'BigQuery arrays must be homogeneous.'
        )

    py_type = element_types.pop()
    bq_type = BQ_TYPE_MAP.get(py_type)
    if not bq_type:
        raise TypeError(f'No BigQuery type mapping for array element type {py_type}')

    return ArrayQueryParameter(name, bq_type, list(values))


@st.cache_resource
def get_bq_client():
    return bigquery.Client(
        credentials=service_account.Credentials.from_service_account_info(
            st.secrets['gcp_service_account']
        )
    )


def db_read_table(
    tbl: str, dset: str | None = None, cols: list[str] | None = None
) -> pl.DataFrame:
    """
    Read a bigquery table

    Args:
        tbl: table name (with optional dataset or project name)
        cols: columns to select (all columns if not specified)

    Returns: query results

    Examples:

    ```
    db_read_table('my_dset.my_tbl')
    db_read_table('my_tbl', dset='my_dset')
    db_read_table('my_tbl', dset='my_dset', cols=['id'])
    ```
    """
    if dset:
        if '.' in tbl:
            raise ValueError("can't determine dataset name!")
        tbl_full = f'{dset}.{tbl}'
    else:
        tbl_split = tbl.split('.')
        if len(tbl_split) not in (2, 3):
            raise ValueError("can't determine dataset name!")
        tbl_full = f'{tbl_split[0]}.{tbl_split[1]}'

    cols_str = ', '.join(f'`{col}`' for col in cols) if cols else '*'
    query = f'SELECT {cols_str} FROM `{tbl_full}`'

    client = get_bq_client()
    query_job = client.query(query)
    return pl.DataFrame(query_job.to_arrow())


def db_read_query(query: str, params: dict[str, Any] | None = None) -> pl.DataFrame:
    """
    Run a bigquery query and return the result.

    Args:
        query: query to run
        params: query parameter names and values

    Returns: query results

    Example:

    ```
    query = "select * from tbl where id = @my_id"
    params = {"my_id": 1234}
    db_read_query(query, params)
    ```
    """
    job_config = (
        bigquery.QueryJobConfig(
            # ScalarQueryParameter.from_type_default(k, v)
            query_parameters=[_bq_param(k, v) for k, v in params.items()]
        )
        if params
        else None
    )

    client = get_bq_client()
    query_job = client.query(query, job_config=job_config)
    return pl.DataFrame(query_job.to_arrow())


def duckdb_read_table(tbl: str):
    cxn = st.connection(
        'duckdb',
        type=DuckDBConnection,
        database=st.secrets['connections']['duckdb']['database'],
    )
    return cxn.table(tbl)


def duckdb_read_query(q: str):
    cxn = st.connection(
        'duckdb',
        type=DuckDBConnection,
        database=st.secrets['connections']['duckdb']['database'],
    )
    return cxn.query(q)


class DuckDBConnection(ExperimentalBaseConnection[duckdb.DuckDBPyConnection]):
    """
    Custom duckdb connection
    source: https://github.com/streamlit/release-demos/blob/master/1.22/st-experimental-connection/duckdb_connection/connection.py
    """

    def _connect(self, **kwargs) -> duckdb.DuckDBPyConnection:
        if 'database' in kwargs:
            db = kwargs.pop('database')
        else:
            db = self._secrets['database']
        return duckdb.connect(database=db, **kwargs)

    def cursor(self) -> duckdb.DuckDBPyConnection:
        return self._instance.cursor()

    def query(self, query: str, ttl='1w', **kwargs) -> pl.DataFrame:
        @cache_data(ttl=ttl)
        def _query(query: str, **kwargs) -> pl.DataFrame:
            with self.cursor() as cur:
                cur.execute(query, **kwargs)
                return cur.pl()

        return _query(query, **kwargs)

    def table(self, table: str, ttl='1w') -> pl.DataFrame:
        @cache_data(ttl=ttl)
        def _table(table: str) -> pl.DataFrame:
            with self.cursor() as cur:
                return cur.table(table).pl()

        return _table(table)
