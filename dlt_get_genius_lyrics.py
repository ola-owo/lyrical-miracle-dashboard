'''
Get song metadata from Genius API
'''
import logging
from pathlib import Path

import duckdb
import polars as pl
import polars.selectors as cs
import dlt
from dlt import source, resource, transformer
from dlt.sources.helpers.requests.retry import Client

from lyric_analyzer_base.genius import make_client, _excluded_terms
from lyric_analyzer_base.database import *


log = logging.getLogger(__name__)
SPOTIFY_DB = Path('data/spotify.duckdb')

# Build the lyricsgenius client
excluded_terms = _excluded_terms.copy()
excluded_terms.remove('instrumental')
genius = make_client(skip_non_songs=False, excluded_terms=excluded_terms)
# EXPERIMENTAL: replace lyricsgenius requests session with dlt's session
# https://dlthub.com/docs/api_reference/dlt/sources/helpers/requests/session
genius_headers = genius._session.headers
genius._session = Client(request_timeout=10, respect_retry_after_header=True,
                         session_attrs=dict(headers=genius_headers))

@transformer()
def lyrics(genius_record):
    '''
    Scrape song lyrics from Genius
    '''

    lyrics = genius.lyrics(
        song_url=genius_record.get('path'),
        song_id=genius_record.get('id'),
        remove_section_headers=True
    )
    if lyrics:
        yield {'id': genius_record['id'], 'lyrics': lyrics}


DEST_TABLE = 'lyrics'
duckdb_dest = dlt.destinations.duckdb(str(SPOTIFY_DB))
pipeline = dlt.pipeline('genius_lyrics', destination=duckdb_dest, dataset_name='genius')
if DEST_TABLE in pipeline.dataset().tables:
    song_ids_old = pl.DataFrame(pipeline.dataset()[DEST_TABLE][['id']].arrow())
    log.info(f'filtering out {song_ids_old.height} old songs')
else:
    song_ids_old = pl.DataFrame(schema={'id': pl.Int64})

with duckdb.connect(SPOTIFY_DB, read_only=True) as cxn:
    song_ids = cxn.sql('''select distinct id, path from genius.song_matches
                       anti join "song_ids_old" using (id)''').pl()

n_search = song_ids.height
log.info(f'{n_search} songs to search')

for i, song_ids_part in enumerate(song_ids.iter_slices(50)):
    print(f'\rloading chunk {i+1}:', end='')
    song_ids_part = song_ids_part.to_arrow()
    load_info = pipeline.run(
        song_ids_part | lyrics(),
        table_name=DEST_TABLE,
        write_disposition='append',
        primary_key='id'
    )
