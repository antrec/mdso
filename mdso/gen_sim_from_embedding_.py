#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S_new #future similarity matrix
For each point:
    find neighborhood
    fit line
    update elements of S_new
"""
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.neighbors import BallTree
from scipy.linalg import svd


def dis_local_lin_fit(X):
    """
    fit points by a line and use their projection on the line fit to get a
    local distance matrix in the neighborhood (used to add to the new
    similarity in gen_sim_from_embedding).
    """
    (n_, d_) = np.shape(X)

    # Make a linear fit
    b = np.mean(X, axis=0)
    X = X - b
    _, _, V = svd(X, full_matrices=False)

    # Use projection on the line fit
    Xproj = np.dot(X, V[0].T)
    # assert(Xproj.shape[0] == n_)

    # Return local dissimilarity matrix in coo-values format
    diffs = np.abs(np.tile(Xproj, n_) - np.repeat(Xproj, n_))
    iis = np.repeat(np.arange(n_), n_)
    jjs = np.tile(np.arange(n_), n_)

    return(iis, jjs, diffs)


def gen_sim_from_embedding(embedding, k_nbrs=10, norm_local_diss=True,
                           norm_sim=False, type_simil=None,
                           return_dense=False, eps_graph=False, eps_val=None):
    """
    """
    (n, dim) = np.shape(embedding)
    i_idx = []
    j_idx = []
    v_diss = []
    v_cpt = []

    tree = BallTree(embedding)

    # Get all neighbors at once ?
    _, all_nbrs = tree.query(embedding, k=k_nbrs)

    for idx in range(n):

        # _, nrst_nbrs = tree.query([embedding[idx]], k=k_nbrs)
        # nrst_nbrs = nrst_nbrs[0]
        nrst_nbrs = all_nbrs[idx]
        sub_embedding = embedding[nrst_nbrs, :]

        # fit the best line for these points
        i_sub, j_sub, v_sub = dis_local_lin_fit(sub_embedding)

        # normalize dissimilarity
        if norm_local_diss:
            v_sub /= v_sub.max()

        # update Diss_new or S_new
        i_idx.extend(nrst_nbrs[i_sub])
        j_idx.extend(nrst_nbrs[j_sub])
        v_diss.extend(v_sub)
        # Update count
        v_cpt.extend(np.ones(len(v_sub)))

    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    v_diss = np.array(v_diss)
    v_cpt = np.array(v_cpt)
    if norm_sim or not(type_simil):
        # create count matrix and invert it, and normalize Diss_new
        count_mat = coo_matrix((v_cpt, (i_idx, j_idx)),
                               shape=(n, n), dtype=int)
        # (i_count, j_count, v_count) = find(count_mat)
        # inv_count_mat = coo_matrix((1./v_count, (i_count, j_count)),
        #                            shape=(n, n), dtype='float')
    if not(type_simil):
        # Dissimilarity matrix
        S_new = coo_matrix((v_diss, (i_idx, j_idx)))
        S_new.data /= count_mat.data
        # S_new.multiply(inv_count_mat)
        # Switch from distance matrix to similarity matrix by taking max - self
        S_new.data *= -1
        S_new.data -= S_new.data.min()
        # S_new *= -1
        # S_new -= S_new.min()

    elif type_simil == 'exp':
        v_sim = np.exp(-v_diss)
    elif type_simil == 'inv':
        v_sim = 1./v_diss
    else:
        raise ValueError('algo_local_ordering.simil must be exp or inv.')

    if type_simil:
        S_new = coo_matrix((v_sim, (i_idx, j_idx)))
        if norm_sim:
            S_new.data /= count_mat.data
            # S_new.multiply(inv_count_mat)

    if return_dense:
        S_new = S_new.toarray()

    return(S_new)


if __name__ == '__main__':

# import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mdso import SimilarityMatrix, spectral_embedding
from time import time

# Set parameters for data generation
n = 500
type_noise = 'gaussian'
ampl_noise = 0.5
type_similarity = 'LinearStrongDecrease'
apply_perm = False
# Build data matrix
data_gen = SimilarityMatrix()
data_gen.gen_matrix(n, type_matrix=type_similarity, apply_perm=apply_perm,
                    noise_ampl=ampl_noise, law=type_noise)
mat = data_gen.sim_matrix
embedding = spectral_embedding(mat, norm_laplacian='random_walk',
                               scale_embedding='heuristic')
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
           c=np.arange(n))
plt.show()


t0 = time()
S_new = get_sim_from_embedding(embedding, k_nbrs=40, norm_local_diss=True,
                       norm_sim=False, type_simil=None,
                       return_dense=False, eps_graph=False, eps_val=None)
t1 = time()
print('former - elapsed : {}'.format(t1-t0))

t0 = time()
S_2 = get_sim_from_embedding2(embedding, k_nbrs=40, norm_local_diss=True,
                       norm_sim=False, type_simil=None,
                       return_dense=False, eps_graph=False, eps_val=None)
t1 = time()
print('new - elapsed : {}'.format(t1-t0))

    fig, axes = plt.subplots(1, 2)
    axes[0].matshow(mat)
    axes[1].matshow(S_new.toarray())
    plt.show()

    k_nbrs = 30
