from airflow.sdk import ObjectStoragePath, dag, task

import get_scrobbles, genius, embeddings


@dag(
    dag_id='lyrics_analyzer',
    schedule=None,
    catchup=False,
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
