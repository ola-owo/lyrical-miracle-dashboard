from time import sleep

from airflow.sdk import task

import polars as pl
from google import genai

from lyric_analyzer_base.database import *
from lyric_analyzer_base.keys import get_keys


@task
def embed_lyrics() -> bool:
    '''
    Get song lyrics embeddings using Gemini API

    TODO: Split by token count instead, use Gemini token-counter API
    '''

    # Number of songs to process per batch
    EMBEDDING_JOB_SIZE = 100
    # recommended embedding sizes are 768, 1536, and 3072 (full size)
    EMBEDDING_DIM = 768
    # wait time in between job status checks
    JOB_POLL_TIME = 60

    # get lyrics that don't have embeddings yet
    df_lyrics_new = duckdb_query(
        'select * from "lyrics" '
        'anti join "lyrics_embed" using (g_id)',
        read_only=True
    )
    if df_lyrics_new.is_empty():
        return False

    lyrics_to_obj = lambda text: {"parts": [{"text": text}]}
    struct_type = pl.Struct({
        'parts': pl.List(pl.Struct({
            'text': pl.String
        }))
    })

    # build a table of (key, request) pairs to be fed into gemini api
    # we use the genius song id as the key, and the lyrics as the request body
    lyrics_requests = (
        df_lyrics_new
        .select([
            pl.col('g_id').cast(pl.String).alias('key'),
            pl.col('lyrics')
                .alias('request')
                .map_elements(lyrics_to_obj, return_dtype=struct_type)
        ])
    )
    print(f'{lyrics_requests.height} songs need embeddings')

    '''
    - split df_lyrics into blocks
    - convert each block into a list of json requests (`lyrics_request_parts`)
    - package json list into a batch request
    - if request succeeded, download results and append to `embeddings_list`
    '''
    embeddings_list = [] # list of retrieved embeddings
    lyrics_requests_parts = (
        lyrics_requests
        .with_columns(pl.row_index().floordiv(EMBEDDING_JOB_SIZE).alias('index'))
        .partition_by('index')
    )
    gemini_client = genai.Client(api_key=get_keys()['gemini_api_key'])
    for df in lyrics_requests_parts:
        job = gemini_client.batches.create_embeddings(
            model='gemini-embedding-001',
            src={'inlined_requests': {
                'config': {'output_dimensionality': EMBEDDING_DIM},
                'contents': df['request'].to_list()}},
            config={'display_name': 'lyrics-batch-embeddings'})

        while True:
            job = gemini_client.batches.get(name=job.name)
            if job.state.name in ('JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED'):
                break
            print(f'Job state: {job.state.name}. Waiting {JOB_POLL_TIME} seconds...')
            sleep(JOB_POLL_TIME)

        print(f"Job finished with state: {job.state.name}")
        if job.state.name == 'JOB_STATE_FAILED':
            print(f"Error: {job.error}")
            break

        if job.state.name == 'JOB_STATE_SUCCEEDED':
            resp = job.dest.inlined_embed_content_responses
            embeddings_list.extend([r.response.embedding.values for r in resp])

    # merge embeddings with lyrics (and genius ids),
    df_lyrics_embed = lyrics_requests.select(
        pl.col('key').cast(int).alias('g_id'),
        pl.Series(embeddings_list).cast(pl.Array(pl.Float64, EMBEDDING_DIM)).alias('embedding')
    )

    duckdb_write_table(df_lyrics_embed, 'lyrics_embed', if_exists='append')
    return not df_lyrics_embed.is_empty()
