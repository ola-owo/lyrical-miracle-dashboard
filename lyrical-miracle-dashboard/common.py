from string import ascii_uppercase as ALPHABET
from pathlib import Path

import duckdb
import plotly
import polars as pl
import polars.selectors as cs
import streamlit as st

from lyric_analyzer_base.database import DUCKDB_FILE


###
### Globals
###

# data
DATA_DIR = Path('apps/dashboard/data/')

# plotting
PALETTE = plotly.colors.qualitative.D3

# date/time management
TIME_ZONE = 'US/Eastern'  # for converting scrobbles timestamps from UTC

# clustering
RANDOM_SEED = 1738
KMEANS_FILE = 'apps/dashboard/data/kmeans.pkl.gz'

# time-of-day binning
TIME_BIN_BOUNDARIES = (6, 12, 18)
TIME_BIN_LABELS = ('morning', 'midday', 'evening', 'night')
TIME_BIN_PALETTE = ('#e2e38b', '#e7a553', '#7e4b68', '#292965')

# big5
BIG5_TRAITS = (
    'Openness to Experience',
    'Conscientiousness',
    'Extraversion',
    'Agreeableness',
    'Neuroticism',
)
BIG5_TRAITS_SHORT = list('OCEAN')
BIG5_TRAITS_POS = (
    'Open to experience',
    'Organized',
    'Extraverted',
    'Agreeable',
    'Neurotic',
)
BIG5_TRAITS_NEG = (
    'Closed to experince',
    'Messy',
    'Introverted',
    'Disagreeable',
    'Emotionally stable',
)


###
### Helper functions
###


# data
@st.cache_data
def duckdb_read_table_cached(tbl):
    with duckdb.connect(DUCKDB_FILE, read_only=True) as cxn:
        return cxn.table(tbl).pl()


# stats
def get_quantile(df: pl.DataFrame, i: int) -> pl.DataFrame:
    """
    Find the quantiles of each column of row `i` of dataframe `df`.

    All dtypes of `df` must be numeric/comparable
    """
    return df.select(cs.all().get(i).ge(cs.all()).mean())


# clustering
def make_df_cluster_labels(n_clusters):
    return pl.DataFrame(
        {
            'cluster': range(n_clusters),
            'cluster_label': [ALPHABET[i] for i in range(n_clusters)],
        }
    )
