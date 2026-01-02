from airflow.sdk import task
import pendulum
import polars as pl
import polars.selectors as cs

from lyric_analyzer_base.database import *
from lyric_analyzer_base.lastfm import get_scrobbles_between

TABLE_SCROBBLES = 'scrobbles'


@task#.short_circuit
def get_scrobbles() -> pl.DataFrame:
    '''
    Get all the lastest scrobbles, and write them to table `scrobbles`.
    
    :return: Indicates whether there were any new scrobbles
    :rtype: Boolean
    '''
    with duckdb.connect(DUCKDB_FILE, read_only=True) as cxn:
        last_scrobble_ts = cxn.sql(
            f'select ts from {TABLE_SCROBBLES} order by ts desc limit 1'
            ).fetchone()[0] + 1
        first_scrobble_ts = cxn.sql(
            f'select ts from {TABLE_SCROBBLES} order by ts asc limit 1'
            ).fetchone()[0]
        scrobbles = cxn.sql(f'select * from {TABLE_SCROBBLES} limit 0').pl()

    last_scrobble_time = pendulum.from_timestamp(last_scrobble_ts, 'UTC')
    first_scrobble_time = pendulum.from_timestamp(first_scrobble_ts, 'UTC')
    endtime = int(pendulum.now().timestamp())
    
    while True:
        print(f'requesting scrobbles from {str(last_scrobble_time)}',
              f' to {str(pendulum.from_timestamp(endtime, 'UTC'))}')

        # lastfm API call!
        resp = get_scrobbles_between(last_scrobble_ts, endtime)

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
        # so we have to re-request with endtime=last_scrobble_time, df['ts'].min()
        scrobbles = pl.concat((scrobbles, df))
        endtime = df['ts'][-1]

    scrobbles = scrobbles.sort('ts')
    print(scrobbles.select(pl.from_epoch('ts'), 'artist', 'album'))

    duckdb_write_table(scrobbles, 'scrobbles', if_exists='append')

    return not scrobbles.is_empty()
