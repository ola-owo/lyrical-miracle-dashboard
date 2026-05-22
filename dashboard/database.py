"""Database I/O tools"""

from streamlit.connections import ExperimentalBaseConnection
from streamlit.runtime.caching import cache_data

import duckdb
import polars as pl
from polars import selectors as cs
import streamlit as st


def db_read_table(tbl: str, cols=None):
    tbl = '.'.join(f'"{part}"' for part in tbl.strip('"').split('.'))
    if cols:
        cols_str = ', '.join(f'"{col}"' for col in cols)
    else:
        cols_str = '*'
    return pl.read_database_uri(
        f'SELECT {cols_str} FROM {tbl}', st.secrets['connections']['neon']['url']
    ).drop(cs.starts_with('_dlt_'))


def db_read_query(q: str):
    return pl.read_database_uri(q, st.secrets['connections']['neon']['url'])


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
