"""
Custom duckdb connection
source: https://github.com/streamlit/release-demos/blob/master/1.22/st-experimental-connection/duckdb_connection/connection.py
"""

from streamlit.connections import ExperimentalBaseConnection
from streamlit.runtime.caching import cache_data

import duckdb
import polars as pl


class DuckDBConnection(ExperimentalBaseConnection[duckdb.DuckDBPyConnection]):
    """Basic st.experimental_connection implementation for DuckDB"""

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
