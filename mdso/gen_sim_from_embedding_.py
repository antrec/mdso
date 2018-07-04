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


def fit_line_get_proj_dist(X):
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

    # Restrict to nonzero values (exclude the main diagonal)
    idx_pos = np.where(diffs > 0)[0]

    return(iis[idx_pos], jjs[idx_pos], diffs[idx_pos])


def gen_sim_from_embedding(embedding, k_nbrs=10, norm_local_diss=False,
                           norm_sim=True, type_simil=None):
    """
    """
    (n, dim) = np.shape(embedding)
    i_idx = []
    j_idx = []
    v_diss = []
    v_cpt = []

    tree = BallTree(embedding)

    # Get the neighbors for all points at once ?
    _, all_nbrs = tree.query(embedding, k=k_nbrs)

    for idx in range(n):

        # _, nrst_nbrs = tree.query([embedding[idx]], k=k_nbrs)
        # nrst_nbrs = nrst_nbrs[0]
        nrst_nbrs = all_nbrs[idx]
        sub_embedding = embedding[nrst_nbrs, :]

        # fit the nearest neighbors by a line, project them on it, and compute
        # the inter-points distance on that line to define a new disssimilarity
        # (given in (data, row, col) format)
        i_sub, j_sub, v_sub = fit_line_get_proj_dist(sub_embedding)

        # normalize dissimilarity
        if norm_local_diss:
            v_sub /= v_sub.max()

        # update Diss_new
        i_idx.extend(nrst_nbrs[i_sub])
        j_idx.extend(nrst_nbrs[j_sub])
        v_diss.extend(v_sub)
        # Update count
        v_cpt.extend(np.ones(len(v_sub)))

    S_new = coo_matrix((v_diss, (i_idx, j_idx)), shape=(n, n), dtype='float64')

    # if not type_simil:
    #     norm_sim = True

    if norm_sim:
        # create count matrix and use inverse to normalize Diss_new
        count_mat = coo_matrix((v_cpt, (i_idx, j_idx)),
                               shape=(n, n), dtype=int)
        S_new.data /= count_mat.data

    # Dissimilarity matrix to similarity matrix
    if not(type_simil):
        S_new.data *= -1
        S_new.data -= S_new.data.min()
    elif type_simil == 'exp':
        S_new.data = np.exp(-S_new.data)
    elif type_simil == 'inv':
        S_new.data = 1./S_new.data
    else:
        raise ValueError('algo_local_ordering.simil must be exp or inv.')

    return(S_new)


if __name__ == '__main__':

    import os
    from time import time
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mdso import SimilarityMatrix, spectral_embedding
    from mdso.utils import get_conn_comps

    # Set parameters for data generation
    n = 500
    type_noise = 'gaussian'
    ampl_noise = .5
    type_similarity = 'LinearStrongDecrease'
    apply_perm = False
    # Build data matrix
    data_gen = SimilarityMatrix()
    data_gen.gen_matrix(n, type_matrix=type_similarity, apply_perm=apply_perm,
                        noise_ampl=ampl_noise, law=type_noise)
    mat = data_gen.sim_matrix
    embedding = spectral_embedding(mat, norm_laplacian='random_walk',
                                   scale_embedding='heuristic',
                                   n_components=20)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
               c=np.arange(n))

    norm_local_diss_opts = [True, False]
    norm_sim_opts = [False, True]
    type_sim_opts = ['inv', 'exp', None]

    for norm_local_diss in norm_local_diss_opts:
        for type_simil in type_sim_opts:
            for norm_sim in norm_sim_opts:
                t0 = time()
                S_new = gen_sim_from_embedding(embedding, k_nbrs=20,
                                               norm_local_diss=norm_local_diss,
                                               norm_sim=norm_sim,
                                               type_simil=type_simil)
                t1 = time()
                print('computed new similarity from embedding \
                      - {}s'.format(t1-t0))
                fig, axes = plt.subplots(1, 2)
                axes[0].matshow(mat)
                axes[1].matshow(S_new.toarray())
    plt.show()


# ############ Real example : DNA data from E. coli ONT reads ################
    t0 = time()
    # Get similarity matrix
    mdso_dir = os.path.dirname(os.path.abspath(__file__))
    # mdso_dir = '/Users/antlaplante/THESE/RobustSeriationEmbedding/mdso/mdso'
    ecoli_data_dir = '/'.join(mdso_dir.split('/')[:-1])
    ecoli_data_dir += '/examples/e_coli/ecoli_data'
    sim_mat_fn = ecoli_data_dir + '/sim_mat.npz'
    if os.path.exists(sim_mat_fn):
        loader = np.load(sim_mat_fn)
        iis = loader['row']
        jjs = loader['col']
        vvs = loader['data']
        n_reads = loader['shape'][0]
        positions = loader['pos']
        # Remove lowest overlap score values to clean things up
        ovlp_thr = np.percentile(vvs, 50)
        over_thr = np.where(vvs > ovlp_thr)[0]
        sim_mat = coo_matrix((vvs[over_thr],
                             (iis[over_thr], jjs[over_thr])),
                             shape=loader['shape'],
                             dtype='float64').tocsr()
        t1 = time()
        print("Built similarity matrix - {}s".format(t1-t0))
        # Restrict to main connected component if disconnected similarity
        ccs = get_conn_comps(sim_mat, min_cc_len=10)
        sub_idxs = ccs[0]
        new_mat = sim_mat.tolil()[sub_idxs, :]
        new_mat = new_mat.T[sub_idxs, :].T
        sub_pos = positions[sub_idxs]
        true_perm = np.argsort(sub_pos)
        true_inv_perm = np.argsort(true_perm)
        t2 = time()
        print("Restricted to main conn. comp. - {}s".format(t2-t1))
        # Compute the embedding
        # Remark : amg is a lot faster than arpack on this large sparse
        # similarity matrix. It requires the pyamg package (install with
        # ```conda install pyamg``` or ```pip install pyamg``` for instance).
        # If not available, use arpack.
        try:
            embedding = spectral_embedding(new_mat, norm_laplacian=False,
                                           scale_embedding=False,
                                           eigen_solver='amg',
                                           norm_adjacency='coifman')
        except Exception:
            embedding = spectral_embedding(new_mat, norm_laplacian=False,
                                           scale_embedding=False,
                                           eigen_solver='arpack',
                                           norm_adjacency='coifman')
        t3 = time()
        print("Computed Laplacian embedding - {}s".format(t3-t2))
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
        #            c=true_inv_perm)
        # plt.title("3d embedding of DNA overlap based similarity matrix")

        t0 = time()
        S_new = gen_sim_from_embedding(embedding, k_nbrs=20,
                                       norm_local_diss=False,
                                       norm_sim=True,
                                       type_simil=None)
        t1 = time()
        print('computed new similarity from embedding \
              - {}s'.format(t1-t0))
        if not isinstance(new_mat, coo_matrix):
            new_mat = new_mat.tocoo()
        if not isinstance(S_new, coo_matrix):
            S_new = S_new.tocoo()
        # multiply pointwise by new_mat to avoid having too many nonzeros
        # on the scatter plot
        S_new.multiply(new_mat)
        fig, axes = plt.subplots(1, 2)
        axes[0].scatter(true_inv_perm[new_mat.row], true_inv_perm[new_mat.col],
                        edgecolors='b', facecolors='none')
        axes[1].scatter(true_inv_perm[S_new.row], true_inv_perm[S_new.col],
                        edgecolors='r', facecolors='none')
        plt.show()

    else:
        get_sim_script = '/'.join(mdso_dir.split('/')[:-1])
        get_sim_script += '/examples/e_coli/build_ecoli_sim_mat.py'
        print("File {} not found. Please run the script {} to get the \
              similarity matrix".format(sim_mat_fn, get_sim_script))
