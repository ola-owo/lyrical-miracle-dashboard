"""Tests for dashboard.vector_search"""

from types import SimpleNamespace
from unittest.mock import Mock

import faiss
import numpy as np
import polars as pl
import pytest

from dashboard import vector_search
from dashboard.common import EMBEDDING_DIM


def _basis(index: int) -> np.ndarray:
    vec = np.zeros(EMBEDDING_DIM, dtype='float32')
    vec[index] = 1.0
    return vec


@pytest.fixture
def search_vecs_path(tmp_path, synth):
    path = tmp_path / 'lyrics_search_vecs.parquet'
    synth.search_vecs.write_parquet(path)
    return path


@pytest.fixture
def search_vecs_patched(monkeypatch, search_vecs_path):
    monkeypatch.setattr(vector_search, 'SEARCH_VECTORS_PATH', search_vecs_path)
    return search_vecs_path


@pytest.fixture
def tiny_index() -> faiss.Index:
    """Index over the three basis vectors in the synthetic search data"""
    index = faiss.IndexIDMap(faiss.IndexFlatIP(EMBEDDING_DIM))
    index.add_with_ids(
        np.stack([_basis(0), _basis(1), _basis(2)]),
        np.array([101, 102, 103], dtype='int64'),
    )
    return index


def _fake_genai_client(vectors: list[list[float]]) -> Mock:
    client = Mock()
    client.models.embed_content.return_value = SimpleNamespace(
        embeddings=[SimpleNamespace(values=vec) for vec in vectors]
    )
    return client


###
### Index building
###
def test_get_lyrics_embed_schema(search_vecs_patched):
    df = vector_search.get_lyrics_embed().collect()
    assert df['id'].to_list() == [101, 102, 103]
    assert df.schema['embedding'] == pl.Array(pl.Float32, EMBEDDING_DIM)


def test_make_index_searches_correct_song(search_vecs_patched):
    index = vector_search.make_index()
    assert index.ntotal == 3

    scores, ids = index.search(_basis(2).reshape(1, -1), 1)
    assert ids[0, 0] == 103
    assert scores[0, 0] == pytest.approx(1.0)


###
### Vector search
###
def test_vector_search_ranks_by_similarity(monkeypatch, tiny_index):
    monkeypatch.setattr(vector_search, 'make_index', lambda: tiny_index)

    results = vector_search._vector_search(_basis(1).reshape(1, -1), n=3)

    assert set(results.columns) >= {'search_num', 'id', 'dist', 'rank'}
    ranked = results.sort('rank')
    assert ranked['id'][0] == 102
    assert set(results['id'].to_list()) == {101, 102, 103}
    assert ranked['dist'][0] == pytest.approx(1.0)


def test_vector_search_dist_filters(monkeypatch, tiny_index):
    monkeypatch.setattr(vector_search, 'make_index', lambda: tiny_index)

    near = vector_search._vector_search(_basis(1).reshape(1, -1), n=3, min_dist=0.5)
    assert near['id'].to_list() == [102]

    far = vector_search._vector_search(_basis(1).reshape(1, -1), n=3, max_dist=0.5)
    assert set(far['id'].to_list()) == {101, 103}


def test_vector_search_multiple_queries(monkeypatch, tiny_index):
    monkeypatch.setattr(vector_search, 'make_index', lambda: tiny_index)

    results = vector_search._vector_search(np.stack([_basis(0), _basis(2)]), n=1).sort(
        'search_num'
    )
    assert results.height == 2
    assert results['id'].to_list() == [101, 103]


###
### Embedding queries
###
def test_embed_query_single_text_is_unit_norm(monkeypatch):
    client = _fake_genai_client([[3.0, 4.0]])
    monkeypatch.setattr(vector_search, 'make_client', lambda: client)

    emb = vector_search._embed_query('hello')

    assert emb.shape == (1, 2)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0)
    assert np.allclose(emb[0], [0.6, 0.8])


def test_embed_query_multiple_texts_each_unit_norm(monkeypatch):
    client = _fake_genai_client([[3.0, 4.0], [1.0, 0.0]])
    monkeypatch.setattr(vector_search, 'make_client', lambda: client)

    emb = vector_search._embed_query(['a', 'b'])

    assert emb.shape == (2, 2)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0)


def test_embed_query_builds_prompt(monkeypatch):
    client = _fake_genai_client([[1.0, 0.0]])
    monkeypatch.setattr(vector_search, 'make_client', lambda: client)

    vector_search._embed_query('some | text', task='query')

    _, kwargs = client.models.embed_content.call_args
    assert kwargs['model'] == 'gemini-embedding-2'
    assert kwargs['contents'] == ['task: query | query: some  text']  # '|' is stripped
    assert kwargs['config'].output_dimensionality == EMBEDDING_DIM


###
### Text search
###
def test_text_search_joins_song_info(monkeypatch, search_vecs_patched):
    monkeypatch.setattr(
        vector_search,
        '_embed_query',
        lambda text: np.zeros((1, EMBEDDING_DIM), dtype='float32'),
    )
    search_res = pl.DataFrame(
        {'search_num': [0], 'id': [102], 'dist': [0.9]}
    ).with_row_index('rank')
    monkeypatch.setattr(vector_search, '_vector_search', Mock(return_value=search_res))

    results = vector_search.text_search('hello', 5)

    assert results.columns == ['search', 'rank', 'id', 'dist']
    assert results['search'].to_list() == ['hello']
    assert results['id'].to_list() == [102]
