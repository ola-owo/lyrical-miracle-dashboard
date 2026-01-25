import logging

from airflow.sdk import task
import pendulum as pn
import polars as pl
import polars.selectors as cs

from lyric_analyzer_base.database import *
from lyric_analyzer_base.lastfm import get_scrobbles_between

TABLE_SCROBBLES = 'scrobbles'

log = logging.getLogger(__name__)

class GetScrobblesException(Exception):
    '''Error encountered when running `get_scrobbles()`'''
    pass


@task#.short_circuit
def get_scrobbles(start_time: pn.DateTime = None, end_time: pn.DateTime = None) -> bool:
    '''
    Get all the lastest scrobbles, and write them to table `scrobbles`.
    
    :return: Indicates whether there were any new scrobbles
    :rtype: Boolean
    '''

    with duckdb.connect(DUCKDB_FILE, read_only=True) as cxn:
        last_scrobble_ts = cxn.sql(
            f'select max(ts) from {TABLE_SCROBBLES}'
            ).fetchone()[0]
        first_scrobble_ts = cxn.sql(
            f'select min(ts) from {TABLE_SCROBBLES}'
            ).fetchone()[0]
        scrobbles = cxn.sql(f'select * from {TABLE_SCROBBLES} limit 0').pl()

    last_scrobble_time = pn.from_timestamp(last_scrobble_ts, 'UTC')
    first_scrobble_time = pn.from_timestamp(first_scrobble_ts, 'UTC')

    if start_time and end_time:
        '''
        TODO (maybe) if `start_time` and `end_time` both specified, just backfill.
        This is bc backfilling requires both `data_interval_start` and `data_interval_end`
        '''
        if start_time <= first_scrobble_time and end_time >= last_scrobble_time:
            raise GetScrobblesException("Can't have `start_time` before oldest scrobble and `end_time` after newest scrobble")
        elif start_time >= first_scrobble_time and end_time <= last_scrobble_time:
            log.warning('Data interval is inside (first_scrobble, last_scrobble) range - nothing to load')
            return False
    elif start_time:
        log.info('`start_time` specified: backfilling...')
        if start_time >= first_scrobble_time:
            log.warning('`start_time` is after the oldest scrobble - nothing to load')
            return False
        end_time = first_scrobble_time
    elif end_time:
        log.info('`end_time` specified: updating with new scrobbles...')
        if end_time <= last_scrobble_time:
            log.warning('`end_time` is before the newest scrobble - nothing to load')
            return False
        start_time = last_scrobble_time + pn.duration(seconds=1)
    else:
        log.info('neither `start_time` nor `end_time` specified: updating with new scrobbles...')
        start_time = last_scrobble_time + pn.duration(seconds=1)
        end_time = pn.now()

    start_ts = int(start_time.timestamp())
    end_ts = int(end_time.timestamp())
    
    while True:
        print(f'requesting scrobbles from {str(start_time)} to {str(end_time)}')

        # lastfm API call!
        resp = get_scrobbles_between(start_ts, end_ts)

        # convert json response to flat dataframe
        # note: result is sorted from new to old
        mbid_cols = cs.by_dtype(pl.Struct({'mbid': pl.String, '#text': pl.String}))
        df = pl.DataFrame(resp.json()['recenttracks']['track'])
        if df.is_empty():
            break
        # `date` col is sometimes missing from very recent scrobbles,
        # in this case we need to try again later
        if 'date' not in df.columns:
            break
        df = df.select(
            pl.col('date').struct.field('uts').cast(pl.Int32).alias('ts'),
            pl.col('name').alias('song'),
            mbid_cols.struct.field('#text').name.keep(),
            mbid_cols.struct.field('mbid').name.keep().name.suffix('_id'),
            pl.col('mbid').alias('song_id')
        ).select(scrobbles.columns)
        # scrobbles may be missing between oldest recieved scrobble time and requested start time,
        # so we have to re-request with end_ts=last_scrobble_time, df['ts'].min()
        scrobbles = pl.concat((scrobbles, df))
        end_ts = df['ts'][-1]
        end_time = pn.from_timestamp(end_ts)

    scrobbles = scrobbles.sort('ts')
    print(scrobbles.select(pl.from_epoch('ts'), 'artist', 'album'))

    duckdb_write_table(scrobbles, 'scrobbles', if_exists='append')

    return not scrobbles.is_empty()
