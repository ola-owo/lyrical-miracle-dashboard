"""
LyricsGenius Genius.com api client
use search_song(title, artist) to search for song lyrics
  set get_full_info to speed up api call (not sure what extra info there is)

docs: https://lyricsgenius.readthedocs.io/en/master/reference/genius.html
github: https://github.com/johnwmillr/LyricsGenius
"""

from lyricsgenius import Genius

from .keys import get_key_genius

# exclude songs with these terms (case insensitive)
_excluded_terms = [
    '(live)',
    '(remix)',
    'instrumental',
    # exclude translations:
    'tradução',
    'traduccion',
    'traducción',
    'çeviri',
    'traduzione',
    'перевод',
    'переклад',
    'traduction',
    'Übersetzung',
    'Ubersetzung',
]


def make_client():
    return Genius(
        get_key_genius(),
        verbose=False,
        remove_section_headers=True,
        skip_non_songs=True,
        excluded_terms=_excluded_terms,
        sleep_time=1,
        timeout=15,
        retries=1,
    )
