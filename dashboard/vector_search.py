"""Vector search tools"""

import streamlit as st
import numpy as np
import polars as pl
from google import genai
from google.genai import types
import faiss

from database import db_read_query
from common import EMBEDDING_DIM, SEARCH_VECTORS_PATH


###
### Get lyrics embeddings and build index
###
def get_lyrics_embed():
    return pl.scan_parquet(SEARCH_VECTORS_PATH).cast(
        pl.Array(pl.Float32, EMBEDDING_DIM)
    )


@st.cache_resource
def make_index():
    """
    Make a faiss vector search index

    (for now) Use the flat inner-product index with genius ID mappings.
    This is best for smallish ( < 1M) datasets with few searches
    """
    df_lyrics_embed = get_lyrics_embed()
    emb_mat = df_lyrics_embed.select('embedding').collect().to_series().to_numpy()

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
def _embed_query(
    text: str | list[str], task: str = 'search result', dim: int = EMBEDDING_DIM
) -> np.typing.NDArray:
    """
    Embed one or more texts using `gemini-embedding-2`
    By default, use the "search result" task type.

    :param text: text(s) to embed
    :param task: embedding task type,
        see [here](https://ai.google.dev/gemini-api/docs/embeddings#task-types-embeddings-2)
    :param dim: embedding vector dimensionality
    :return: Text embeddings
    :rtype: Numpy 2D array
    """
    GEMINI_MODEL = 'gemini-embedding-2'

    def build_prompt(text: str):
        text = text.replace('|', '')
        return f'task: {task} | query: {text}'

    if isinstance(text, str):
        text = [text]

    client = make_client()  # cached
    res = client.models.embed_content(
        model=GEMINI_MODEL,
        contents=[build_prompt(t) for t in text],
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
def text_search(search_texts: str | list[str], n_results: int = 5):
    """
    Search the index for one or more texts

    :param search_texts: Text(s) to search for
    :param n_results: Number of search results
    :return: DataFrame with ranked results and similarity scores
    :rtype: DataFrame
    """
    df_lyrics_embed = get_lyrics_embed()
    search_vecs = _embed_query(search_texts)
    search_res = _vector_search(search_vecs, n_results)
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
    """
    Transform search results by pulling extra genius song info

    Args:
        search_res: search results from `text_search()`
    """
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
    ids_str = '(' + ','.join(search_res['id'].cast(str)) + ')'
    query = f"""
    SELECT id
        , title song
        , primary_artist_names artist
        , album__name album
        , album__release_date_for_display release_date
        , url
    FROM "genius"."songs"
    WHERE id in {ids_str}
    """

    return db_read_query(query)
