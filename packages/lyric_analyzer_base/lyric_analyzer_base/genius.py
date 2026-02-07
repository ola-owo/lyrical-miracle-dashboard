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
    'traduções',
    'traduccion',
    'traducción',
    'traducciones',
    'traducciónes',
    'traduction',
    'traductions',
    'traduzione',
    'traduzioni',
    'vertaling',
    'vertalingen',
    'Übersetzung',
    'Ubersetzung',
    'Übersetzungen',
    'Ubersetzungen',
    'перевод',
    'переводы',
    'переклад',
    'переклади',
    'çeviri',
    'çeviriler'
    'अनुवाद',
    '翻译',
    '翻訳',
]

def make_client(
    verbose=False,
    remove_section_headers=True,
    skip_non_songs=True,
    excluded_terms=_excluded_terms,
    sleep_time=1,
    timeout=15,
    retries=1,
    **kwargs
):
    return Genius(
        get_key_genius(),
        verbose=verbose,
        remove_section_headers=remove_section_headers,
        skip_non_songs=skip_non_songs,
        excluded_terms=excluded_terms,
        sleep_time=sleep_time,
        timeout=timeout,
        retries=retries,
        **kwargs
    )
