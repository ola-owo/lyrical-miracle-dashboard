"""
SPOTIFY API IDEAS:

1. recently-played: could replace scrobbles API
- https://developer.spotify.com/documentation/web-api/reference/get-recently-played
- more info than lastfm (popularity, explicitness, ...)

2. /me/top/tracks and /me/top/artists
- gets top tracks/artists based on "affinity"
"""

import requests
from requests.utils import quote

from .keys import get_keys_spotify

TOKEN_URL = 'https://accounts.spotify.com/api/token'
API_BASE = 'https://api.spotify.com/v1'


class SpotifyClient:
    def __init__(self):
        keypair = get_keys_spotify()
        self.key = keypair['key']
        self.secret = keypair['secret']
        self.token = None
        self.header = None
        self.refresh_token()  # update token and header

    def refresh_token(self):
        """
        Get a new auth token from spotify
        """
        token = self.get_token(self.key, self.secret)
        self.token = token
        self.header = {
            'Authorization': f'{token["token_type"]} {token["access_token"]}'
        }

    @staticmethod
    def get_token(key, secret) -> dict:
        resp = requests.post(
            TOKEN_URL,
            {
                'grant_type': 'client_credentials',
                'client_id': key,
                'client_secret': secret,
            },
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            raise e  # TODO: handle this?
        return resp.json()

    def request(self, endpoint, params=None):
        resp = requests.get(API_BASE + endpoint, params=params, headers=self.header)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            if resp.status_code == 401:
                print('401 unauthorized -- refreshing token and retrying...')
                self.refresh_token()
                self.request(endpoint, params)
            raise e  # TODO: handle this?
        return resp.json()

    def search(self, q, type, tags=None, **kwargs):
        if tags:
            for k, v in tags.items():
                q += f' {k}:{v}'
        q = quote(q, safe=':')  # extra quoting layer bc of tags
        params = {'q': q, 'type': type} | kwargs
        return self.request('/search', params)

    def get_track(self, track_id):
        return self.request(f'/tracks/{track_id}')
