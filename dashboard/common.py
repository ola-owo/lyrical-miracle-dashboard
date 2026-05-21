from string import ascii_uppercase as ALPHABET
from pathlib import Path

import plotly
import polars as pl
import streamlit.components.v1 as components


###
### Globals
###

# data
DATA_DIR = Path('data/')

# plotting
PALETTE = plotly.colors.qualitative.D3

# date/time management
TIME_ZONE = 'US/Eastern'  # for converting scrobbles timestamps from UTC

# clustering
N_CLUSTERS = 7
RANDOM_SEED = 1738
KMEANS_FILE = DATA_DIR / 'kmeans.pkl.gz'

# embeddings
EMBEDDING_DIM = 768

# session binning
SES_MAX_GAP = pl.duration(days=1)  # max gap between consecutive songs in a session

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
def make_df_cluster_labels(n_clusters):
    return pl.DataFrame(
        {
            'cluster': range(n_clusters),
            'cluster_label': [ALPHABET[i] for i in range(n_clusters)],
        }
    )


def timeout_popup(timeout_ms: int = 1_800_000):
    """
    When inactive, show an "are you still there?" popup which blocks the thread,
    allowing the app to spin down to zero.
    Without this the app stays alive indefinitely due to periodic health checks

    source: discuss.streamlit.io/t/streamlit-app-deployment-expensive-on-cloud-run/65337/10
    """
    components.html(
        """
        <script>
        (function () {
        const ctx = window.top || window;           // Prefer the top window so the alert blocks the main tab
        if (ctx.__sessionTimeoutStarted) return;    // Guard against multiple initializations
        ctx.__sessionTimeoutStarted = true;

        function scheduleNext() {
            ctx.__sessionTimeoutTimer = ctx.setTimeout(() => {
            try { ctx.alert("Are you still there?"); }
            finally {
                // Schedule the next run only after the alert is dismissed (ensures at most one pending timer)
                scheduleNext();
            }
            }, %d);
        }

        // Kick off the first run
        scheduleNext();
        })();
        </script>
        """
        % timeout_ms,
        height=0,
    )
