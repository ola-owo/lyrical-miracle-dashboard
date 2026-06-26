"""Helper functions for getting album covers"""

import requests
import streamlit as st

from dashboard.database import db_read_query


def lastfm_request(params: dict):
    """
    Make a LastFM API request.

    :param params: Request parameters
    :type params: dict
    """
    API_BASE = 'https://ws.audioscrobbler.com/2.0'
    params = params | {'format': 'json', 'api_key': st.secrets['lastfm']['key']}
    resp = requests.get(API_BASE, params)
    resp.raise_for_status()
    return resp


@st.cache_data
def get_lastfm_img(artist, song, mbid=None) -> str | None:
    """Search lastfm for a song and return its album art url"""

    def get_track_info(track, artist, mbid=None):
        """
        Get an album image url

        :param track: Track name
        :param artist: Artist name
        :param mbid: MusicBrainz ID (optional)
        """
        return lastfm_request(
            {'method': 'track.getInfo', 'track': track, 'artist': artist, 'mbid': mbid}
        )

    resp = get_track_info(artist, song, mbid).json()
    try:
        # imgs are ordered from smallest to largest
        img_url = resp['track']['album']['image'][-1]['#text']
    except KeyError:
        img_url = None
    return img_url


@st.cache_data
def get_genius_img(g_id: int) -> str | None:
    """get a song's album cover url from the genius songs table"""
    g = db_read_query(
        """SELECT song_art_image_thumbnail_url FROM "genius"."songs"
        WHERE id = $1 LIMIT 1""",
        (g_id,),
    )

    if g.is_empty():
        return None

    # use song_art_image or header_image if they exist
    for imgcol in ['song_art_image_thumbnail_url', 'header_image_thumbnail_url']:
        # if image is missing, genius uses this default:
        # https://assets.genius.com/images/default_cover_image.png?[TIMESTAMP]
        if 'default_cover_image' not in g[0, imgcol]:
            return g[0, imgcol]

    return None
