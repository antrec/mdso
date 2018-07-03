#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S_new #futur similarity matrix
For each point:
    find neighborhood
    fit line
    update elements of S_new
"""
import numpy as np
from scipy.sparse import find, coo_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree
from scipy.linalg import svd


def compute_distances(embedding, sim_mat):
    # (n, dim) = np.shape(embedding)
    # if issparse(sim_mat):
    (iis, jjs, vvs) = find(sim_mat)
    xis = np.sum(np.power(embedding[iis, :], 2), axis=1)
    xjs = np.sum(np.power(embedding[jjs, :], 2), axis=1)
    ijprod = np.sum(np.multiply(embedding[iis, :], embedding[jjs, :]), axis=1)
    dds = xis + xjs - 2 * ijprod
    jjs = np.append(np.array([-1]), jjs)
    switch_j = jjs[1:] - jjs[:-1]
    # switch_j = np.append(np.array([-1]), jjs[1:] - jjs[:-1])
    (_, switch_idx, _) = find(switch_j)
    switch_idx = np.append(switch_idx, np.array([len(jjs)]))

    return(iis, switch_idx, dds)


def find_nearest_neighbours(embedding, idx_point, k, points_to_consider):
    '''
    find the k nearest neighbours to a point idx_point among those
    that are in points_to_consider.
    send back their current index in the initial embedding
    '''
    (n, dim) = np.shape(embedding)
    # compute distance matrice to idx_point
    reduced_emb = embedding[points_to_consider, :]
    reduced_emb_size = np.shape(reduced_emb)[0]
    point_idx_point = np.reshape(embedding[idx_point, :], (1, dim))
    embedding_translated = reduced_emb - \
        np.repeat(point_idx_point, reduced_emb_size, axis=0)

    # compute distances
    dist_point = np.sum(embedding_translated**2, axis=1)
    sort_dist = np.argsort(dist_point)
    k_tilde = min(len(sort_dist), k)

    nbrs_idxs = [points_to_consider[idx] for idx in list(sort_dist[:k_tilde])]

    ''' AR: on n'utilise plus la methode suivante, non ?
    # attribute for the method find_extremal_point_rope
    cum_weight_neighbors = np.sum(sort_dist[:k_tilde])
    '''
    return(np.array(nbrs_idxs))


def linear_fit(X):
    """
    fit points by a line
    """
    (n_samples, n_features) = np.shape(X)
    b = np.mean(X, axis=0)
    X = X - b
    U, S, V = svd(X, full_matrices=False)
    w = V[0]

    return(w, b)

def fit_local_line(sub_embedding):
    '''
    find the line that minimize MSE for points in the Sub_Embedding
    '''
    (_, dim) = np.shape(sub_embedding)
    b = np.mean(sub_embedding, axis=0)
    if len(b) != dim:
        raise ValueError('there is a problem in the choice of b.')
    # compute the first principal component!
    pca = PCA(n_components=1)
    pca.fit_transform(sub_embedding)
    w = np.reshape(pca.components_, (1, dim))
    # self.w = w  # tangent vector to the line
    # self.b = b  # a point on the line
    return(w, b)


def project_line(w, b, point):
    '''
    Give coefficient lamdba such that
    point = lambda*w+b ,
    line =(w: directed vector; b: point belonging to the line)
    '''
    if abs(np.sum(w**2) - 1) > 1e-2:
        raise ValueError('the vector w is not normalized')
    lamb = np.sum(w*(point-b))
    return(lamb)


def give_local_ordering(w, b, sub_embedding, k):
    '''
    Given a line and a set of point to order, gives the ordering
    induced by the projection of these points onto the line.
    '''
    l_lamb = []
    K = np.shape(sub_embedding)[0]
    # if K > k:
    #     raise ValueError('the size of the points_current should never '
    #                      'be greater than k.')
    # compute the lambd for each point
    for i in range(K):
        lamb = project_line(w, b, sub_embedding[i, :])
        l_lamb.append(lamb)
    return(l_lamb)


def create_local_dissimilarity(l_lamb):
    '''
    For a local neighborhood V={x_1,...,x_k} we have
    l_lamb = {lamb_1,...,lamb_k}
    We want S=(abs(lamb_i-lamb_j))_{i,j=1,...,k}
    '''
    array_lamb = np.array(l_lamb)
    K = len(l_lamb)
    Diss_sub = np.repeat(np.reshape(array_lamb, (K, 1)), K, axis=1) - \
        np.repeat(np.reshape(array_lamb, (1, K)), K, axis=0)
    Diss_sub = np.abs(Diss_sub)
    return(Diss_sub)


def get_sim_from_embedding(embedding, k_nbrs=10, norm_local_diss=True,
                           norm_sim=False, type_simil=None,
                           return_dense=False, eps_graph=False, eps_val=None):
    '''

    '''
    (n, dim) = np.shape(embedding)
    i_idx = []
    j_idx = []
    v_diss = []
    v_cpt = []

    tree = BallTree(embedding)

    # Get all neighbors at once ?
    # _, these_nbrs = tree.query(embedding, k=k_nbrs)

    for idx in range(n):

        _, nrst_nbrs = tree.query([embedding[idx]], k=k_nbrs)
        nrst_nbrs = nrst_nbrs[0]
        # nrst_nbrs = these_nbrs[idx]
        sub_embedding = embedding[nrst_nbrs, :]

        # fit the best line for these points
        (w, b) = fit_local_line(sub_embedding)

        # create a local similarity matrix
        l_lamb = give_local_ordering(w, b, sub_embedding, k_nbrs)
        Diss_sub_new = create_local_dissimilarity(l_lamb)

        # normalize dissimilarity
        if norm_local_diss:
            Diss_sub_new *= 1./Diss_sub_new.max()

        # update Diss_new or S_new
        (i_sub, j_sub, v_sub) = find(Diss_sub_new)
        i_sub = nrst_nbrs[i_sub]
        j_sub = nrst_nbrs[j_sub]
        i_idx.extend(list(i_sub))
        j_idx.extend(list(j_sub))
        v_diss.extend(list(v_sub))
        # Update count
        v_cpt.extend(list(np.ones(len(v_sub))))

    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    v_diss = np.array(v_diss)
    v_cpt = np.array(v_cpt)
    if norm_sim or not(type_simil):
        # create count matrix and invert it, and normalize Diss_new
        count_mat = coo_matrix((v_cpt, (i_idx, j_idx)),
                               shape=(n, n), dtype=int)
        (i_count, j_count, v_count) = find(count_mat)
        inv_count_mat = coo_matrix((1./v_count, (i_count, j_count)),
                                   shape=(n, n), dtype='float')
    if not(type_simil):
        # Dissimilarity matrix
        S_new = coo_matrix((v_diss, (i_idx, j_idx)))
        S_new.multiply(inv_count_mat)
        (i_new, j_new, v_new) = find(S_new)
        v_new *= -1
        v_new -= v_new.min()
        S_new = coo_matrix((v_new, (i_new, j_new)),
                           shape=(n, n))
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
            S_new.multiply(inv_count_mat)

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
ampl_noise = 2
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
S_new = get_sim_from_embedding(embedding, k_nbrs=10, norm_local_diss=True,
                       norm_sim=False, type_simil=None,
                       return_dense=False, eps_graph=False, eps_val=None)
t1 = time()
print('elapsed : {}'.format(t1-t0))
    fig, axes = plt.subplots(1, 2)
    axes[0].matshow(mat)
    axes[1].matshow(S_new.toarray())
    plt.show()
