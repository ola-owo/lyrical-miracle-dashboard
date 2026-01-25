import requests

from .keys import get_keys_lastfm

API_BASE = 'https://ws.audioscrobbler.com/2.0'
LASTFM_KEYS = get_keys_lastfm()
BASE_PARAMS = {'format': 'json', 'api_key': LASTFM_KEYS['key']}


def lastfm_request(params: dict):
    """
    Make a LastFM API request.

    :param params: Request parameters
    :type params: dict
    """
    params = BASE_PARAMS | params
    resp = requests.get(API_BASE, params)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise e  # TODO: handle this?
    return resp


def get_old_scrobbles(until: int, limit: int = 1000):
    """
    Get the 1000 latest scrobbles before timestamp `until`

    :param until: Unix timestamp (seconds, UTC)
    """
    return lastfm_request(
        {
            'method': 'user.getRecentTracks',
            'user': LASTFM_KEYS['user'],
            'to': until,
            'limit': limit,
        }
    )


def get_new_scrobbles(since: int, limit: int = 1000):
    """
    Get the 1000 latest scrobbles after timestamp `since`.

    :param since: Unix timestamp (seconds, UTC)
    """
    return lastfm_request(
        {
            'method': 'user.getRecentTracks',
            'user': LASTFM_KEYS['user'],
            'from': since,
            'limit': limit,
        }
    )


def get_scrobbles_between(start: int, end: int, limit: int = 1000):
    """
    Get the 1000 latest scrobbles between timestamps `start` and `end`.

    :param start: Start timestamp (seconds, UTC)
    :param end: End timestamp (seconds, UTC).
    """
    return lastfm_request(
        {
            'method': 'user.getRecentTracks',
            'user': LASTFM_KEYS['user'],
            'from': start,
            'to': end,
            'limit': limit,
        }
    )


def get_image_url(track, artist, mbid=None):
    """
    Get an album image url
    
    :param track: Track name
    :param artist: Artist name
    :param mbid: MusicBrainz ID (optional)
    """
    return lastfm_request({
        'method': 'track.getInfo',
        'track': track,
        'artist': artist,
        'mbid': mbid
    })
