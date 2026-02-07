import json
from time import sleep
from random import random
import logging

from airflow.sdk import task
import polars as pl
import polars.selectors as cs
import numpy as np
import duckdb
import pendulum as pn
import fsspec

from lyric_analyzer_base.genius import *
from lyric_analyzer_base.database import *


log = logging.getLogger(__name__)


def normalize_song_titles(strings):
    '''
    normalize a polars string column/series.
    this is meant for song titles with extra tags like "feat. (...)" or "(remix)"
    '''
    return (
        strings
        .str.normalize()
        .str.replace(r'\(.*\)', '')
        .str.replace(r'\[.*\]', '')
        .str.replace(r'(?i)feat\.? .+$', '')
        .str.replace(r' - .*$', '')
        #.str.replace_all(r'[^\w\s]', '') # remove punctuation
        .str.replace_all(r'\s+', ' ')
    )


@task#.short_circuit
def search_songs() -> bool:
    '''
    Search Genius for any newly scrobbled songs.
    Songs that have already been searched in Genius are ignored.

    This task reads from table `scrobbles` and writes to tables
    `genius_search_queries` and `genius_search_results`.

    :return: Whether any new searches were made
    '''
    def parse_genius_search_res(hits) -> list[tuple]:
        '''
        Extract song info from genius search results.
        Return a dict, or None if there are no results

        Remember to add other genius_search_results fields (song,artist,searchtext)
        after calling this function
        '''
        if not hits:
            return None

        df_data = {}
        for h in hits:
            song_res = h['result']
            df_data.setdefault('g_id', []).append(song_res.get('id'))
            df_data.setdefault('g_title', []).append(song_res.get('full_title'))
            df_data.setdefault('g_song', []).append(song_res.get('title'))
            df_data.setdefault('g_artist', []).append(song_res.get('primary_artist_names'))
            df_data.setdefault('g_lyrics_complete', []).append(song_res.get('lyrics_state') == 'complete')
            df_data.setdefault('g_path', []).append(song_res.get('path'))

            release_date_parts = song_res.get('release_date_components') # key always exists sometimes is None
            if not release_date_parts:
                # release_date_parts is None
                release_date = None
            elif not all(release_date_parts.get(key) for key in ('year', 'month', 'day')):
                # one or more date components is None
                release_date = None
            else:
                release_date = pn.date(
                    release_date_parts['year'],
                    release_date_parts['month'],
                    release_date_parts['day']
                )
            df_data.setdefault('g_release_date', []).append(release_date)

        return df_data


    # Build a separate "genius search results" table with the same artist,id,etc fields
    # The difference is that we keep all search results and filter them later
    DF_GENIUS_SEARCH_COLS = [
        ('song', pl.String),
        ('artist', pl.String),
        ('searchtext', pl.String),
        ('g_id', pl.Int32),
        ('g_title', pl.String),
        ('g_artist', pl.String),
        ('g_song', pl.String),
        ('g_lyrics_complete', pl.Boolean),
        ('g_path', pl.String),
        ('g_release_date', pl.Date),
    ]
    DF_GENIUS_SEARCH_SCHEMA = dict(DF_GENIUS_SEARCH_COLS)

    # filter out already-searched songs
    # duckdb.create_function('normalize_song_titles', normalize_song_titles)
    df_genius_search_queries = duckdb_query(
        'select distinct s.song, s.artist '
        'from "scrobbles" s anti join "genius_search_queries" '
        'using (song, artist)',
        read_only=True
    )

    if df_genius_search_queries.is_empty():
        log.info('No new songs to search.')
        return False
    log.info(f'{df_genius_search_queries.height} new songs to search')

    # build search queries
    df_genius_search_queries = df_genius_search_queries.with_columns(
        pl.concat_str([
            ( # strip song title metadata (feat, remix, etc) from search
                pl.col('song')
                .map_batches(normalize_song_titles, return_dtype=pl.String)
            ),
            pl.col('artist'),
        ], separator=' ').alias('searchtext')
    )

    # build search results table
    df_genius_search_results = pl.DataFrame(schema=DF_GENIUS_SEARCH_SCHEMA)
    genius_searched = pl.Series('searched', [False] * df_genius_search_queries.height)

    # For each unsearched song (no_id_rows), search and get the top results
    # If the API call fails (connection lost, timeout, rate limit), just rerun this cell
    gen = make_client()
    no_id_rows = (
        df_genius_search_queries.with_row_index()
        .filter(~genius_searched)
        .get_column('index')
    )
    for i in no_id_rows:
        song = df_genius_search_queries[i, 'song']
        artist = df_genius_search_queries[i, 'artist']
        searchtext = df_genius_search_queries[i, 'searchtext']

        # API call!
        # note: search_songs() uses the public API with 10k max requests/day
        # note 2: search_songs() `per_page` max is 50
        res = gen.search_songs(searchtext, per_page=10)

        search_res_parsed = parse_genius_search_res(res['hits'])
        if not search_res_parsed:
            continue

        search_res_df = pl.DataFrame(
            search_res_parsed | {'song': song, 'artist': artist, 'searchtext': searchtext},
            schema=DF_GENIUS_SEARCH_SCHEMA
        )
        df_genius_search_results = pl.concat((df_genius_search_results, search_res_df))
        genius_searched[i] = True

    duckdb_write_table(df_genius_search_results, 'genius_search_results',
                       if_exists='append')
    duckdb_write_table(df_genius_search_queries, 'genius_search_queries',
                       if_exists='append')
    return True


@task#.short_circuit
def filter_search_results() -> bool:
    '''
    Choose the closest match for each searched song.

    This is based on the textual similarity of the Genius song/artist name to
    the LastFM song/artist name.
    
    Matches are reflected in column `is_match` of table `genius_search_results`.
    '''
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    df_unmatched_songs = (
        duckdb_query(
            'with temp as (select *, bool_or(is_match) over (partition by song,artist) as already_matched from "genius_search_results") '
            'select * exclude (is_match, match_score, already_matched) from temp where not already_matched',
            read_only=True
        )
        .select([
            pl.col(['artist', 'song', 'searchtext', 'g_id']),
            pl.col(['g_title', 'g_song', 'g_artist']).replace(None, ''),
            pl.concat_str(['song', 'artist'], separator=' by ').alias('title')
        ])
    )
    if df_unmatched_songs.is_empty():
        return False

    # vectorize song titles
    # using tfidf instead of word count bc it prioritizes unique words (like names)
    #   and depriortizes common words (like "feat", "by", "remix")
    titles_concat = normalize_song_titles(
        pl.concat((df_unmatched_songs['searchtext'], df_unmatched_songs['g_title']))
        .replace(None,'')
    )
    artists_concat = normalize_song_titles(
        pl.concat((df_unmatched_songs['artist'], df_unmatched_songs['g_artist']))
        .replace(None,'')
    )
    songs_concat = normalize_song_titles(
        pl.concat((df_unmatched_songs['song'], df_unmatched_songs['g_song']))
        .replace(None,'')
    )
    titles_nostrip_concat = (
        pl.concat((df_unmatched_songs['title'], df_unmatched_songs['g_title']))
        .replace(None,'')
    )
    counts_title = TfidfVectorizer(strip_accents='unicode').fit(titles_concat)
    counts_artist = TfidfVectorizer(strip_accents='unicode').fit(artists_concat)
    counts_song = TfidfVectorizer(strip_accents='unicode').fit(songs_concat)
    counts_title_nostrip = TfidfVectorizer(strip_accents='unicode').fit(titles_nostrip_concat)

    # get cosine similarity between original and genius titles
    cos_title = np.diag(cosine_similarity(
        counts_title.transform(df_unmatched_songs['searchtext']),
        counts_title.transform(df_unmatched_songs['g_title'])
    ))
    cos_artist = np.diag(cosine_similarity(
        counts_artist.transform(normalize_song_titles(df_unmatched_songs['artist'])),
        counts_artist.transform(normalize_song_titles(df_unmatched_songs['g_artist']))
    ))
    cos_song = np.diag(cosine_similarity(
        counts_song.transform(normalize_song_titles(df_unmatched_songs['song'])),
        counts_song.transform(normalize_song_titles(df_unmatched_songs['g_song']))
    ))
    cos_title_nostrip = np.diag(cosine_similarity(
        counts_title_nostrip.transform(df_unmatched_songs['title']),
        counts_title_nostrip.transform(df_unmatched_songs['g_title'])
    ))

    # take the closest match, or none if no good matches.
    # cos=0.3 is a decent cutoff from visual inspection,
    # but there might be a few false positives
    COSINE_CUTOFF = 0.3
    df_genius_song_matches = (
        df_unmatched_songs
        .with_columns(
            pl.Series(cos_title_nostrip).alias('cos_title_nostrip'),
            pl.Series(cos_title).alias('cos_title'),
            pl.Series(cos_artist).alias('cos_artist'),
            pl.Series(cos_song).alias('cos_song'),
            )
        .filter(pl.col('g_id').is_not_null())
        .with_columns((
                0.5*(pl.col('cos_artist') + pl.col('cos_song')) * pl.col('cos_title_nostrip')
            ).alias('cos'))
        .filter(
            pl.col('cos') > COSINE_CUTOFF,
            pl.col('cos') == pl.col('cos').max().over('searchtext')
            )
        .unique(['song', 'artist', 'searchtext']) # if there are multiple top matches, use the 1st one
        .sort('cos')
        .select([
            'song', 'artist', 'searchtext',
            cs.starts_with('g_'),
            pl.col('cos').alias('match_score')
        ])
    )

    log.info(f'{df_genius_song_matches.height}/{df_unmatched_songs.height} songs matched.')
    with duckdb.connect(DUCKDB_FILE) as cxn:
        cxn.register('df_genius_song_matches', df_genius_song_matches)
        cxn.execute(
            'UPDATE "genius_search_results" gsr SET is_match = true, match_score = gsm.match_score '
            'FROM (select * from df_genius_song_matches) gsm '
            'WHERE (gsr.g_id = gsm.g_id and gsr.song = gsm.song and gsr.artist = gsm.artist)'
        )
        cxn.unregister('df_genius_song_matches')
    # duckdb_query(
    #     'UPDATE "genius_search_results" gsr SET is_match = true, match_score = gsm.match_score '
    #     'FROM (select * from df_genius_song_matches) gsm '
    #     'WHERE (gsr.g_id = gsm.g_id and gsr.song = gsm.song and gsr.artist = gsm.artist)'
    # )
    # duckdb_write_table(df_genius_song_matches, 'genius_song_matches', if_exists='append')
    return not df_genius_song_matches.is_empty()


@task
def get_song_metadata() -> bool:
    '''
    Get full metadata from Genius for the input songs.
    '''

    df_genius_song_matches = duckdb_query(
        'select * from "genius_song_matches" gsm anti join "genius_full" gf on gsm.g_id = gf.id',
        read_only=True
    ).unique('g_id')

    n_search = df_genius_song_matches.height
    log.info(f'{n_search} songs to search')
    if n_search == 0:
        return False

    # method 1 (not working): make a list of json responses converted to dataframes,
    # and periodically concatenate and append to db
    # this doesn't work bc pl.read_json() doesn't always get the schema right if some fields are missing
    # song_jsons = []

    # method 2 (works but not ideal): make a temporary jsonl file and append each json response to it,
    # then periodically write the file to the db using duckdb read_json()
    # jsonfile = tempfile.NamedTemporaryFile('wt', delete_on_close=False)
    # jsonfile.close()
    # jsonfile_len = 0

    # method 3 (TODO): same as method 2 but use a virtual in-memory file
    # like this: https://stackoverflow.com/a/76495452
    json_buffer = []
    JSON_FILE = 'records.json'
    fs: fsspec.AbstractFileSystem = fsspec.filesystem('memory')

    def dump_buffer(fs):
        '''
        Write records retrieved so far to duckdb.
        '''
        nonlocal json_buffer
        with fs.open(JSON_FILE, 'wt') as f:
            json.dump(json_buffer, f)
        json_buffer = []
        
        with duckdb.connect(DUCKDB_FILE) as cxn:
            cxn.register_filesystem(fs)
            cxn.execute("INSERT OR IGNORE INTO genius_full SELECT * FROM read_json(?)",
                        [f'memory://{JSON_FILE}'])


    gen = make_client()
    gen.sleep_time = 0.5 # seconds
    for i in range(n_search):
        sleep(random())
        song_id = df_genius_song_matches[i, 'g_id']
        g_title = df_genius_song_matches[i, 'g_title']

        # API call!
        # This uses the public API which has a 10k/day request limit
        try:
            res = gen.song(song_id)
        except AssertionError as e:
            if "Unexpected response status code: 404" in str(e):
                log.error(f'song {song_id} returned 404 ({g_title})')
                continue
            raise e

        if not res:
            continue

        # (method 1)
        # with StringIO(json.dumps(res['song'])) as f:
        #     song_jsons.append(pl.read_json(f))

        # (method 2)
        # with open(jsonfile.name, 'at') as f:
        #     json.dump(res['song'], f)
        #     f.write('\n')
        #     jsonfile_len += 1

        # (method 3)
        json_buffer.append(res['song'])

        # (method 1)
        # if len(song_jsons) >= 100:
        #     duckdb_write_table(pl.concat(song_jsons, how='vertical_relaxed'),
        #                        'genius_full', if_exists='append')
        #     song_jsons = []

        # (method 2)
        # if jsonfile_len >= 100:
        #     with duckdb.connect(DUCKDB_FILE) as cxn:
        #         cxn.execute('insert or ignore into "genius_full" select * from read_json(?)', [jsonfile.name])
        #     jsonfile_len = 0

        # (method 3)
        if len(json_buffer) >= 100:
            dump_buffer(fs)

    else:
        # write to db one last time if needed
        # (method 1)
        # if song_jsons:
        #     duckdb_write_table(pl.concat(song_jsons, how='vertical_relaxed'),
        #                        'genius_full', if_exists='append')

        # (method 2)
        # if jsonfile_len > 0:
        #     with duckdb.connect(DUCKDB_FILE) as cxn:
        #         cxn.execute('insert or ignore into "genius_full" select * from read_json(?)', [jsonfile.name])

        # (method 3)
        if len(json_buffer) > 0:
            dump_buffer(fs)

    # (method 2)
    # Path(jsonfile.name).unlink()
    return True


@task#.short_circuit
def get_lyrics() -> bool:
    '''
    Get song lyrics from Genius for the input songs.

    TODO: flag instrumentals and avoid searching them.
    Instrumentals usually return None,
    so they are searched every time this task is run.
    Maybe there's an "instrumental" flag in the full genius song data?
    '''

    def write_lyrics(df_lyrics, mask=None):
        '''Append lyrics gotten so far to db table'''
        if mask:
            df_lyrics = df_lyrics.filter(mask)
        duckdb_write_table(
            df_lyrics.select(['g_id', 'lyrics']),
            'lyrics',
            if_exists='append')

    # get all song matches that we don't already have lyrics for
    df_lyrics = (
        duckdb_query(
            (
                'SELECT g_id, NULL as lyrics, g_path, g_title, g_release_date '
                'FROM "genius_song_matches" '
                'ANTI JOIN "lyrics" using (g_id) '
            ),
            read_only=True
        )
        .unique('g_id')
    )

    # filter out old songs (TODO: move this into sql query)
    release_date_cutoff = pn.Date.today().subtract(days=180)
    df_lyrics = df_lyrics.filter((pl.col('g_release_date') > release_date_cutoff)
                                 | pl.col('g_release_date').is_null())

    log.info(f'{df_lyrics.height} song lyrics to scrape')
    if df_lyrics.is_empty():
        return False

    df_lyrics_done = pl.Series([False] * df_lyrics.height)
    gen = make_client()
    gen.sleep_time = 2
    for i in range(df_lyrics.height):
        g_id = df_lyrics[i, 'g_id']
        g_path = df_lyrics[i, 'g_path']
        g_title = df_lyrics[i, 'g_title']
        g_release_date = df_lyrics[i, 'g_release_date']

        # API call!
        # This is actually a web scrape, not an API call.
        # Not sure what the rate limits are, a proxy might help
        try:
            # prefer song_url bc it saves an API call
            if g_path:
                res = gen.lyrics(song_url=g_path)
            else:
                res = gen.lyrics(song_id=g_id)
        except AssertionError as e:
            if "Unexpected response status code: 404" in str(e):
                log.error(f'song {g_id} returned 404 ({g_title})')
                continue
            # at this point, it's probably Too Many Requests or request Timeout
            write_lyrics(df_lyrics, df_lyrics_done)
            raise e
        except Exception as e:
            write_lyrics(df_lyrics, df_lyrics_done)
            raise e

        if not res:
            log.warning(f'song {g_id} returned no lyrics ({g_title})')
            if not g_release_date:
                log.info('unknown song age - will try again later')
                continue
            elif (pn.Date.today() - g_release_date).days < 180:
                log.info('song is still new - will try again later')
                continue
            else:
                log.warning("song is old, so we won't query it anymore")
                res = None

        df_lyrics[i, 'lyrics'] = res
        df_lyrics_done[i] = True
        sleep(random())

    write_lyrics(df_lyrics, df_lyrics_done)
    return not df_lyrics.is_empty()
