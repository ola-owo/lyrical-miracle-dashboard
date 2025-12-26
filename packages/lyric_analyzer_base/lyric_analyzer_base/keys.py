import yaml
import os

KEYS_FILE = os.environ.get('LYRICS_ANALYZER_KEYS', 'keys.yml')


# get api keys
def get_keys():
    with open(KEYS_FILE, 'r') as f:
        return yaml.load(f, yaml.CLoader)


def get_keys_lastfm():
    keys = get_keys()
    return {
        'user': keys['lastfm_user'],
        'key': keys['lastfm_key'],
        'secret': keys['lastfm_secret'],
    }


def get_keys_spotify():
    keys = get_keys()
    return {'key': keys['spotify_key'], 'secret': keys['spotify_secret']}


def get_key_genius():
    keys = get_keys()
    return keys['genius_token']


def get_keys_musicbrainz():
    raise Exception('TODO')
