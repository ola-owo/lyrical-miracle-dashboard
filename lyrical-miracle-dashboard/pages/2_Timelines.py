import streamlit as st
import polars as pl
import plotly.express as px

from common import (
    DATA_DIR,
    PALETTE,
    TIME_BIN_PALETTE,
    make_df_cluster_labels,
    timeout_popup,
)

timeout_popup()
st.set_page_config(page_title='Timelines', page_icon='📈')
st.header('Your Listening History')
st.sidebar.header('Your Listening History')


plays_clustered = pl.scan_parquet(DATA_DIR / 'plays_clustered.parquet')
df_stats_all_months = pl.read_parquet(DATA_DIR / 'df_stats_all_months.parquet')
df_sessions = pl.scan_parquet(DATA_DIR / 'df_sessions.parquet')
df_cluster_per_month = pl.scan_parquet(DATA_DIR / 'df_cluster_per_month.parquet')

n_clusters = plays_clustered.select(pl.col('cluster').n_unique()).collect().item()
df_cluster_labels = make_df_cluster_labels(n_clusters)


@st.fragment
def plot_plays_per_cluster(df_cluster_per_month):
    # NOTE: `on_select` seems to be broken -
    # the returned selection is always blank
    # https://github.com/streamlit/streamlit/issues/8760
    # bar_select = st.plotly_chart(
    #     px.bar(
    #         songs_per_month,
    #         x='date',
    #         y='len',
    #         labels={
    #             'date': 'Date',
    #             'len': 'Songs played',
    #         },
    #     ),
    #     on_select='rerun',
    #     selection_mode='points',
    # )
    # st.write(bar_select)

    barchart_plays_per_month = px.bar(
        df_cluster_per_month,
        x='date',
        y='n_cluster_plays',
        color='cluster_label',
        color_discrete_sequence=PALETTE,
        labels={
            'date': 'Date',
            'n_cluster_plays': 'Songs played',
            'cluster_label': 'Cluster',
        },
    )

    # TODO:highlight the bars where x = selected_date.replace(day=1)
    # i tried plotly update_traces() but that updates all bars of the same cluster,
    # instead we should highlight all bars of the same date
    # if st.session_state.selected_date:
    #     pass # TODO
    st.plotly_chart(
        barchart_plays_per_month
        # on_select='rerun',
        # selection_mode='points',
        # key='bar_select2',
    )
    # st.write(st.session_state['bar_select2'])


st.header('Songs played per cluster')
plot_plays_per_cluster(df_cluster_per_month.collect())


@st.fragment
def plot_cluster_time_bins(df_sessions):
    songs_per_timebin_month = (
        df_sessions.group_by(['year', 'month', 'timebin'])
        .len()
        .with_columns(
            pl.date(pl.col('year'), pl.col('month'), 1),
        )
        .sort(['timebin', 'date'])
        .collect()
    )
    fig_bar_timebin = px.bar(
        songs_per_timebin_month,
        x='date',
        y='len',
        color='timebin',
        color_discrete_sequence=TIME_BIN_PALETTE,
        labels={'date': 'Date', 'len': 'Songs played', 'timebin': 'Time of day'},
    )
    st.plotly_chart(fig_bar_timebin)


st.header('Listening at each time of day')
plot_cluster_time_bins(df_sessions)


@st.fragment
def plot_diversity(df_stats_all_months):
    div_cols = ('gini', 'shannon', 'berger')
    div_labels = ('Gini', 'Shannon', 'Inverse Berger-Parker')
    st.session_state.diversity_type_ix = st.selectbox(
        'Index:', list(range(len(div_cols))), format_func=div_labels.__getitem__
    )
    div_col = div_cols[st.session_state.diversity_type_ix]
    div_label = div_labels[st.session_state.diversity_type_ix]

    fig_bar_gini = px.bar(
        df_stats_all_months,
        x='date',
        y=div_col,
        barmode='group',
        labels={
            'date': 'Date',
            div_col: div_label,
        },
    )
    st.plotly_chart(fig_bar_gini)


st.header('Cluster diversity')
plot_diversity(df_stats_all_months)


@st.fragment
def plot_song_latent_distance(df_stats_all_months):
    fig_bar_timebin = px.bar(
        df_stats_all_months,
        x='date',
        y='latent_dist_mean',
        barmode='group',
        labels={
            'date': 'Date',
            'latent_dist_mean': 'Latent-space distance',
        },
    )
    st.plotly_chart(fig_bar_timebin)


# LATENT DISTANCE DISABLED FOR NOW
# st.header(
#     'Embedding distance',
#     help='Average latent-space distance between consecutive songs in a session',
# )
# plot_song_latent_distance(df_stats_all_months)
