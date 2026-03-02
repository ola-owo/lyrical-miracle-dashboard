"Cluster tracks according to lyrics embeddings"

import numpy as np
from scipy import sparse
import streamlit as st
from sklearn.cluster import KMeans

RANDOM_SEED = 1738


@st.cache_data
def run_kmeans(embeddings, n_clusters, random_seed=RANDOM_SEED):
    return KMeans(n_clusters, random_state=random_seed).fit(embeddings)


@st.cache_data
def run_spectral_clustering(embeddings, n_clusters, random_seed=RANDOM_SEED):
    """
    (EXPERIMENTAL) Run spectral clustering.
    For now, using symmetric normalized Laplacian bc its the only one that works well
    """
    emb = embeddings.to_numpy()
    assert n_clusters < emb.shape[0]

    def selfsim(mat):
        """
        Compute pairwise similarity between embeddings matrix.
        Input matrix should have one row per embedding.
        """
        x = mat / np.linalg.norm(mat, axis=1).reshape((-1, 1))
        return x @ x.T

    def threshold(mat, thresh_pct):
        """
        Keep the top `thresh_pct` most similar connections in `mat` (zero the rest),
        and convert it to a sparse matrix.
        """
        # print(np.quantile(mat.flatten(), np.linspace(0, 1, 10, endpoint=False)))
        out = mat.copy()
        thresh = np.quantile(mat, 1 - thresh_pct)
        out[out < thresh] = 0
        return sparse.bsr_array(out)

    # build sparse graph
    SIM_THRESH = 0.8  # proportion of connections to keep
    emb_sim = selfsim(emb)
    np.fill_diagonal(emb_sim, 0.0)  # remove similarity of nodes to themselves
    adj = threshold(emb_sim, SIM_THRESH)

    # compute symmetric normalized laplacian
    # NOTE: when normed, degree matrix 0s are set to 1 and nonzeros are square-rooted
    lap, deg = sparse.csgraph.laplacian(adj, normed=True, return_diag=True, copy=True)

    # get eigenvectors of laplacian
    deg = sparse.diags_array(np.square(deg))
    egvals, egvecs = sparse.linalg.eigsh(lap, k=n_clusters, which='SA')
    egvecs_normalizer = np.linalg.norm(egvecs, axis=1).reshape((-1, 1))
    egvecs_normalizer[egvecs_normalizer == 0] = 1
    egvecs = egvecs / egvecs_normalizer
    print('Laplacian eigenvalues:', egvals)

    # cluster the rows of the eigenvector matrix
    return KMeans(n_clusters, random_state=random_seed).fit(egvecs), egvecs
