def normalize_song_titles(strings):
    '''
    normalize a polars string column/series.
    this is meant for song titles with extra tags like "feat. (...)" or "(remix)"
    '''
    return (
        strings
        .str.normalize()
        .str.replace(r'\(.*\)', '')
        .str.replace(r'\[.*\]', '')
        .str.replace(r'(?i)feat\.? .+$', '')
        .str.replace(r' - .*$', '')
        #.str.replace_all(r'[^\w\s]', '') # remove punctuation
        .str.replace_all(r'\s+', ' ')
    )
