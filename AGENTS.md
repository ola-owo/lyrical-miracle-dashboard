# AGENTS.md

Streamlit dashboard for personal listening history with lyric-clustering analysis.

## Layout

- `Home.py`: entry page.
  `pages/`: Streamlit multipage (`1_Monthlies`, `2_Timelines`, `3_Search`).
- `dashboard/`: main Python package.
  - `database.py`: database I/O
  - `common.py`: common constants and helper functions
  - `vector_search.py`: vector-searching the lyrics database
  - `graphs.py`: networkx graphs
  - `albumart.py`: album art retrieval functions
  - `wrangling.py`: shared dataframe transformations
- `tests/`: pytest suite.
  `tests/app/`: page tests using `streamlit.testing.v1.AppTest`.
  Tests must never read `.streamlit/secrets.toml`:
  BigQuery/LastFM/Gemini calls are always patched.
- `notebooks/precompute.py`: marimo notebook
  that builds the derived data files in `data/`.
- `data/`: generated data files, not part of source code.
- `precompute.sh`: runs the notebook, then uploads `data/*` to GCS.
  Don't run it unless asked.
- `.streamlit/secrets.toml`: credentials.
  Do NOT read this file or run any commands that print it.
  Instead, let the user edit it directly if needed.

## Data sources

- Most dashboard data is in BigQuery.
  It's accessed by Streamlit through the `db_read_table` and `db_read_query` helpers.
- Additional derived data is computed by `precompute.py` and saved in `data/`.

## Development guidelines

- Each Streamlit page must include `timeout_popup()`.
  The timeout popup will block the main thread,
  preventing health checks and allowing Cloud Run to scale to zero after prolonged inactivity.
- Test page changes with `streamlit.testing.v1.AppTest`.
- Run the tests with `uv run pytest`.
  Tests marked `real_data` run the pages against the files in `data/`,
  and are skipped when those files are missing.
  The `dev` dependency group pulls in both the `test` and `precompute` groups.

## Conventions

- Format and lint with `uvx ruff format` and `uvx ruff check`.
  Only check the files you touched.
  Marimo has its own way of formatting notebooks, so `ruff` warnings there are expected.
- Put expensive reads and queries behind `@st.cache_data`.
- Put long-running global objects (like database connections) behind `@st.cache_resource`.
