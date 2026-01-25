from airflow.sdk import dag
import pendulum as pnd

import get_scrobbles, genius, embeddings


@dag(
    dag_id='lyrics_analyzer',
    start_date=pnd.datetime(2020, 1, 1, tz='UTC'),
    schedule='@weekly',
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1, # limited for now bc of duckdb
)
def pipeline():
    got_new_scrobbles = get_scrobbles.get_scrobbles()
    got_search_results = genius.search_songs()
    got_song_matches = genius.filter_search_results()
    got_song_meta = genius.get_song_metadata()
    got_lyrics = genius.get_lyrics()
    got_embeddings = embeddings.embed_lyrics()

    (
        got_new_scrobbles
        >> got_search_results
        >> got_song_matches
        >> got_song_meta
        >> got_lyrics
        >> got_embeddings
    )
    # got_song_matches >> got_song_meta


pipeline()
