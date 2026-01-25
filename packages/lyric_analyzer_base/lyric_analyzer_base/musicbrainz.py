"""
musicbrainz api docs:
https://musicbrainz.org/doc/MusicBrainz_API

remember to change user agent eventually
Application name/<version> ( contact-email )
"""

import requests
#from enum import Enum
from typing import Literal

# from .keys import get_keys_musicbrainz

API_BASE = 'https://musicbrainz.org/ws/2'
USER_AGENT = 'python-requests/2.32.4'

# EntityType = Enum(
#     'MBEntity',
#     [
#         'area',
#         'artist',
#         'event',
#         'genre',
#         'instrument',
#         'label',
#         'place',
#         'recording',
#         'release',
#         'release-group',
#         'series',
#         'work',
#         'url',
#         'track',
#     ],
# )
EntityType = Literal[
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
]


class Entity:
    def __init__(self, entity_type: EntityType, mbid: str):
        self.type = entity_type
        self.id = mbid

    @property
    def name(self) -> str:
        return self.type.name


class MBClient:
    '''MusicBrainz client (un-authenticated)'''
    def __init__(self, user_agent=USER_AGENT):
        self.session = requests.session()
        self.session.headers.update(
            {'User-Agent': user_agent, 'Accept': 'application/json'}
        )

    
    def request(self, endpoint, params=None):
        '''Make a GET reqeust to a given MB endpoint'''
        resp = self.session.get(API_BASE + endpoint, params=params)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            raise e  # TODO: handle this?
        return resp.json()

    
    def get(self, entity: EntityType, mbid: str, inc: list[EntityType] | None = None):
        '''Get an entity'''
        if not inc:
            inc = []
        inc_str = '+'.join(inc)
        endpoint = f'/{entity}/{mbid}'
        return self.request(endpoint, {'inc': inc_str})

    
    def browse(
        self,
        result_type: EntityType,
        entity: Entity,
        inc: list[EntityType] | None = None,
    ):
        '''Browse entities of type `result_type` that are related to `entity`'''
        if not inc:
            inc = []
        inc_str = '+'.join(inc)
        endpoint = f'/{result_type}'
        return self.request(endpoint, {entity.name: entity.id, 'inc': inc_str})

    
    def search(
        self,
        entity_type: EntityType,
        query: str,
        offset: int = None,
        limit: int = None
    ):
        '''Search for entities of a given type'''
        endpoint = f'/{entity_type}'
        params = {'query': query}
        if offset:
            params['offset'] = offset
        if limit:
            params['limit'] = limit
        return self.request(endpoint, params)