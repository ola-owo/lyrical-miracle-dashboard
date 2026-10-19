"""Data wrangling helpers shared by the dashboard pages"""

import pendulum as pn
import polars as pl


def filter_df_by_month(df: pl.DataFrame | pl.LazyFrame, date: pn.Date | None):
    """
    Filter a dataframe by year/month, using the `year` and `month` columns.

    Args:
        df: dataframe with `year` and `month` columns
        date: date whose year and month are used for filtering (no filtering if None)

    Returns: the filtered dataframe, of the same type (eager/lazy) as the input
    """
    if not date:
        return df
    return df.filter((pl.col('year') == date.year) & (pl.col('month') == date.month))


def get_cluster_stats(
    df: pl.DataFrame | pl.LazyFrame, df_centroids: pl.DataFrame | pl.LazyFrame
) -> pl.LazyFrame:
    """
    Get aggregate stats per cluster.

    Args:
        df: play or session-level dataframe with `cluster` and `g_id` columns
        df_centroids: cluster table with `cluster`, `cluster_label`, and `centroid` columns

    Returns: lazy frame with one row per cluster, including clusters with no plays
    """
    return (
        df.lazy()
        .group_by('cluster')
        .agg(
            n_plays=pl.len(),
            n_unique_plays=pl.col('g_id').unique().len(),
            n_sessions=pl.col('session').unique().len(),
            top_song_id=pl.col('g_id').mode().first(),
        )
        .sort('cluster')
        .with_columns((pl.col('n_plays') / pl.col('n_plays').sum()).alias('freq'))
        .join(df_centroids.lazy(), on='cluster', how='full', coalesce=True)
        .with_columns(
            pl.col('freq', 'n_plays', 'n_unique_plays', 'n_sessions').replace(None, 0)
        )
        .select(
            'cluster',
            'cluster_label',
            'freq',
            'n_plays',
            'n_unique_plays',
            'n_sessions',
            'top_song_id',
            'centroid',
        )
    )
