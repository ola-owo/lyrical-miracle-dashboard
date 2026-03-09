"Get album art"

import streamlit as st
from lyric_analyzer_base.database import duckdb_query
from lyric_analyzer_base.lastfm import get_track_info


@st.cache_data
def get_lastfm_img(artist, song, mbid=None) -> str | None:
    resp = get_track_info(artist, song, mbid).json()
    try:
        # imgs are ordered from smallest to largest
        img_url = resp['track']['album']['image'][-1]['#text']
    except KeyError:
        img_url = None
    return img_url


@st.cache_data
def get_genius_img(g_id) -> str | None:
    q = 'SELECT song_art_image_thumbnail_url FROM "genius"."songs" WHERE id = ? LIMIT 1'
    params = [g_id]

    g = duckdb_query(q, params, read_only=True)
    if g.is_empty():
        return None

    # use song_art_image or header_image if they exist
    for imgcol in ['song_art_image_thumbnail_url', 'header_image_thumbnail_url']:
        # if image is missing, genius uses this default:
        # https://assets.genius.com/images/default_cover_image.png?[TIMESTAMP]
        if 'default_cover_image' not in g[0, imgcol]:
            return g[0, imgcol]

    return None
