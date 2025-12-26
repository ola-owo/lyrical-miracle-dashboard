"""
musicbrainz api docs:
https://musicbrainz.org/doc/MusicBrainz_API

remember to change user agent eventually
Application name/<version> ( contact-email )
"""

import requests
from enum import Enum

# from .keys import get_keys_musicbrainz

API_BASE = 'https://musicbrainz.org/ws/2'
USER_AGENT = 'python-requests/2.32.4'

EntityType = Enum(
    'MBEntity',
    [
        'area',
        'artist',
        'event',
        'genre',
        'instrument',
        'label',
        'place',
        'recording',
        'release',
        'release-group',
        'series',
        'work',
        'url',
        'track',
    ],
)


class Entity:
    def __init__(self, entity_type: EntityType, mbid: str):
        self.type = entity_type
        self.id = mbid

    @property
    def name(self) -> str:
        return self.type.name


class MBClient:
    def __init__(self):
        self.session = requests.session()
        self.session.headers.update(
            {'User-Agent': USER_AGENT, 'Accept': 'application/json'}
        )

    def request(self, endpoint, params=None):
        resp = self.session.get(API_BASE + endpoint, params=params)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            raise e  # TODO: handle this?
        return resp.json()

    def get(self, entity: EntityType, mbid: str, inc: list[EntityType] | None = None):
        if not inc:
            inc = []
        inc_str = '+'.join(inc)
        endpoint = f'/{entity.name}/{mbid}'
        return self.request(endpoint, {'inc': inc_str})

    def browse(
        self,
        result_type: EntityType,
        entity: Entity,
        inc: list[EntityType] | None = None,
    ):
        if not inc:
            inc = []
        inc_str = '+'.join(inc)
        endpoint = f'/{result_type.name}'
        return self.request(endpoint, {entity.name: entity.id, 'inc': inc_str})
