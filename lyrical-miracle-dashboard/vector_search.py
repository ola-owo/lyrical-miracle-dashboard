import streamlit as st
import numpy as np
import polars as pl
from google import genai
from google.genai import types
import faiss

from database import db_read_query
from common import DATA_DIR, EMBEDDING_DIM


###
### Get lyrics embeddings and build index
###
df_lyrics_embed = pl.scan_parquet(DATA_DIR / 'lyrics_embed.parquet')


@st.cache_resource
def make_index():
    """
    Make a faiss vector search index

    (for now) Use the flat inner-product index with genius ID mappings.
    This is best for smallish ( < 1M) datasets with few searches
    """
    emb_mat = (
        df_lyrics_embed.select('embedding')
        .cast(pl.Array(pl.Float32, EMBEDDING_DIM))
        .collect()
        .to_series()
        .to_numpy()
    )

    emb_index = faiss.IndexIDMap(faiss.IndexFlatIP(EMBEDDING_DIM))
    emb_index.add_with_ids(emb_mat, df_lyrics_embed.select('id').collect().to_series())
    return emb_index


@st.cache_resource
def make_client() -> genai.Client:
    """Returns a Gemini API client"""
    return genai.Client(api_key=st.secrets['gemini']['api_key'])


###
### Run search
###
@st.cache_data
def _embed_text(
    text: str | list[str], model: str = 'gemini-embedding-001', dim: int = 768
) -> np.typing.NDArray:
    """
    Embed one or more texts using gemini

    :param text: text(s) to embed
    :param client: Gemini API client
    :param model: embedding model name
    :param dim: embedding vector dimensionality
    :return: Text embeddings
    :rtype: Numpy 2D array
    """
    client = make_client()  # cached
    res = client.models.embed_content(
        model=model,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=dim),
    )
    emb = np.array([e.values for e in res.embeddings])
    emb = emb / np.linalg.norm(emb, axis=0)
    return emb


@st.cache_data
def _vector_search(
    search_vecs: np.typing.NDArray, n: int, min_dist=None, max_dist=None
) -> pl.DataFrame:
    """
    Search the index for one or more vectors

    :param search_vecs: Vector(s) to search for
    :param n: Number of search results
    :return: DataFrame with ranked results and similarity scores
    :rtype: DataFrame
    """
    index = make_index()  # cached
    search_res_score, search_res = index.search(search_vecs, n)
    search_res_df = pl.concat(
        pl.DataFrame(
            {'search_num': i, 'id': search_res[i, :], 'dist': search_res_score[i, :]}
        ).with_row_index('rank')
        for i in range(search_vecs.shape[0])
    )
    if min_dist is not None:
        search_res_df = search_res_df.filter(pl.col('dist') >= min_dist)
    if max_dist is not None:
        search_res_df = search_res_df.filter(pl.col('dist') <= max_dist)
    return search_res_df


@st.cache_data
def text_search(search_texts: str | list[str], n: int):
    """
    Search the index for one or more texts

    :param search_texts: Text(s) to search for
    :param n: Number of search results
    :return: DataFrame with ranked results and similarity scores
    :rtype: DataFrame
    """
    search_vecs = _embed_text(search_texts, dim=EMBEDDING_DIM)
    search_res = _vector_search(search_vecs, 5)
    search_df = pl.LazyFrame({'search': search_texts}).with_row_index('search_num')
    return (
        search_res.lazy()
        .join(search_df, on='search_num')
        .join(df_lyrics_embed, on='id', how='left')
        .select('search', 'rank', 'id', 'dist')
        .collect()
    )


@st.cache_data
def transform_search_res(search_res: pl.DataFrame):
    output_schema = pl.DataFrame(
        schema={
            'song': str,
            'artist': str,
            'album': str,
            'release_date': str,
            'url': str,
        }
    )
    if search_res.is_empty():
        return output_schema
    search_res = search_res.unique('id')
    g_ids_str = '(' + ','.join(search_res['id'].cast(str)) + ')'

    # get spotify song info
    # query = '\n'.join((
    #     f'WITH ids as (SELECT id FROM "genius"."song_matches" WHERE g_id in {g_ids_str})',
    #     'SELECT t.name song, t.artist, a.name album, a.release_date, t.external_urls__spotify url',
    #     'FROM ids INNER JOIN "spotify"."tracks" t USING (id) LEFT JOIN "spotify"."albums" a ON (t.album_id = a.id)',
    # ))

    # get genius song info
    query = f"""
    WITH ids as (SELECT g_id AS id FROM "genius"."song_matches"
        WHERE g_id in {g_ids_str})
    SELECT g.title song
        , g.primary_artist_names artist
        , g.album__name album
        , g.album__release_date_for_display release_date
        , g.url
    FROM ids INNER JOIN "genius"."songs" g USING (id)
    """

    return db_read_query(query)
