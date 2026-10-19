"""Tests for dashboard.albumart"""

from unittest.mock import Mock

import polars as pl
import pytest

from dashboard import albumart

DEFAULT_IMG = 'https://assets.genius.com/images/default_cover_image.png?123'


@pytest.fixture
def lastfm_resp():
    resp = Mock()
    resp.json.return_value = {
        'track': {
            'album': {
                'image': [
                    {'#text': 'small.jpg'},
                    {'#text': 'large.jpg'},
                ]
            }
        }
    }
    return resp


def test_lastfm_request_adds_api_key_and_format(fake_secrets, monkeypatch, lastfm_resp):
    get = Mock(return_value=lastfm_resp)
    monkeypatch.setattr(albumart.requests, 'get', get)

    resp = albumart.lastfm_request({'method': 'track.getInfo'})

    assert resp is lastfm_resp
    assert get.call_args.args == (
        'https://ws.audioscrobbler.com/2.0',
        {
            'method': 'track.getInfo',
            'format': 'json',
            'api_key': fake_secrets['lastfm']['key'],
        },
    )
    lastfm_resp.raise_for_status.assert_called_once_with()


def test_get_lastfm_img_returns_largest_image(monkeypatch, lastfm_resp):
    monkeypatch.setattr(albumart, 'lastfm_request', Mock(return_value=lastfm_resp))

    assert albumart.get_lastfm_img('Song', 'Artist') == 'large.jpg'


def test_get_lastfm_img_missing_album_returns_none(monkeypatch):
    resp = Mock()
    resp.json.return_value = {'track': {}}
    monkeypatch.setattr(albumart, 'lastfm_request', Mock(return_value=resp))

    assert albumart.get_lastfm_img('Song', 'Artist') is None


def _genius_frame(song_art: str, header: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            'song_art_image_thumbnail_url': [song_art],
            'header_image_thumbnail_url': [header],
        }
    )


def test_get_genius_img_returns_song_art(monkeypatch):
    query = Mock(return_value=_genius_frame('art.jpg', 'header.jpg'))
    monkeypatch.setattr(albumart, 'db_read_query', query)

    assert albumart.get_genius_img(101) == 'art.jpg'


def test_get_genius_img_falls_back_to_header_image(monkeypatch):
    query = Mock(return_value=_genius_frame(DEFAULT_IMG, 'header.jpg'))
    monkeypatch.setattr(albumart, 'db_read_query', query)

    assert albumart.get_genius_img(101) == 'header.jpg'


def test_get_genius_img_all_defaults_returns_none(monkeypatch):
    query = Mock(return_value=_genius_frame(DEFAULT_IMG, DEFAULT_IMG))
    monkeypatch.setattr(albumart, 'db_read_query', query)

    assert albumart.get_genius_img(101) is None


def test_get_genius_img_empty_result_returns_none(monkeypatch):
    empty = pl.DataFrame(
        schema={
            'song_art_image_thumbnail_url': pl.String,
            'header_image_thumbnail_url': pl.String,
        }
    )
    monkeypatch.setattr(albumart, 'db_read_query', Mock(return_value=empty))

    assert albumart.get_genius_img(101) is None
