import requests
import streamlit as st

API_BASE = 'https://ws.audioscrobbler.com/2.0'


def lastfm_request(params: dict):
    """
    Make a LastFM API request.

    :param params: Request parameters
    :type params: dict
    """
    params = params | {'format': 'json', 'api_key': st.secrets['lastfm']['key']}
    resp = requests.get(API_BASE, params)
    resp.raise_for_status()
    return resp


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
