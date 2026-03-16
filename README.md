# The Lyrical Miracle

A Streamlit dashboard for exploring and analyzing my historical music listening activity.

## Demo

Check out my live demo [here!](https://lyrical-miracle-49172755171.us-central1.run.app/)

## Features

- **Monthlies**: Detailed listening statistics and broken down by month.
- **Timelines**: Interactive timelines of my entire music listening history.
- **Semantic clustering**: Clustering of songs based on semantic meaning of lyrics.
- **Lyric search**: Find songs using lyrical semantic similarity.

## Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) | [Plotly](https://plotly.com/python/)
- **Clustering**: [Scikit-learn](https://scikit-learn.org/) | [NetworkX](https://networkx.org/)
- **LLM**: [Google Gemini](https://ai.google.dev/)
- **Vector Search**: [FAISS](https://github.com/facebookresearch/faiss)
- **Data Processing**: [Polars](https://pola.rs/) | [NumPy](https://numpy.org/)
- **Database**: [neondb](https://neon.com/) | [DuckDB](https://duckdb.org/)

## Setup

Run `uv sync` to build this project with all its dependencies.

### With Docker

Alternatively, you can build a Docker container:

```bash
docker buildx build -t streamlit .
```

### Secrets

Create `.streamlit/secrets.toml` with your Last.fm, Gemini, and SQL database credentials.

```toml
[connections.neon]
url = [...]
type = 'sql'

[lastfm]
user = [...]
key = [...]
secret = [...]

[gemini]
api_key = [...]
```

### Data

Run `precompute.py` to compute and save all the data needed to run the dashboard.

## Run

Start the Streamlit server:

```
uv run streamlit run lyrical-miracle-dashboard/Home.py --server.headless=true --server.port=8501
```

Or use Docker:

```bash
docker run -e PORT=8501 -p 8501:8501 --volume "$(pwd)/data:/app/data" streamlit
```
