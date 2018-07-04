#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Given a similarity matrix on sparse or numpy format, it creates a
Laplacian Embedding, for various type of graph Laplacian as well as
normalization.
So far the similarity is assumed to represent a fully connected graph.
'''
import warnings
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, lobpcg
from mdso.utils import check_similarity, compute_laplacian,\
                       _graph_is_connected, _set_diag
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import _deterministic_vector_sign_flip


def spectral_embedding(adjacency, n_components=8, eigen_solver=None,
                       random_state=None, eigen_tol=1e-15,
                       norm_laplacian=False, drop_first=True,
                       norm_adjacency=False, scale_embedding=True,
                       verb=0):
    """

    REMARK :
    This is an adaptation from the same function in scikit-learn
    [http://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html]
    but slightly modify to account for optional scalings of the embedding,
    ability to normalize the Laplacian with random_walk option, and ability to
    normalize the adjacency matrix with Lafon and Coifman normalization
    [https://doi.org/10.1016/j.acha.2006.04.006] (see check_similarity)


    Project the sample on the first eigenvectors of the graph Laplacian.
    The adjacency matrix is used to compute a normalized graph Laplacian
    whose spectrum (especially the eigenvectors associated to the
    smallest eigenvalues) has an interpretation in terms of minimal
    number of cuts necessary to split the graph into comparably sized
    components.
    This embedding can also 'work' even if the ``adjacency`` variable is
    not strictly the adjacency matrix of a graph but more generally
    an affinity or similarity matrix between samples (for instance the
    heat kernel of a euclidean distance matrix or a k-NN matrix).
    However care must taken to always make the affinity matrix symmetric
    so that the eigenvector decomposition works as expected.
    Note : Laplacian Eigenmaps is the actual algorithm implemented here.
    Read more in the :ref:`User Guide <spectral_embedding>`.
    Parameters
    ----------
    adjacency : array-like or sparse matrix, shape: (n_samples, n_samples)
        The adjacency matrix of the graph to embed.
    n_components : integer, optional, default 8
        The dimension of the projection subspace.
    eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}, default None
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems,
        but may also lead to instabilities.
    random_state : int, RandomState instance or None, optional, default: None
        A pseudo random number generator used for the initialization of the
        lobpcg eigenvectors decomposition.  If int, random_state is the seed
        used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`. Used when
        ``solver`` == 'amg'.
    eigen_tol : float, optional, default=0.0
        Stopping criterion for eigendecomposition of the Laplacian matrix
        when using arpack eigen_solver.
    norm_laplacian : bool or string, optional, default=True
        If True, then compute normalized Laplacian.
        If 'random_walk', compute the random_walk normalization
        [see e.g. https://arxiv.org/abs/0711.0189]
    norm_adjacency : bool or string, optional, default=False
        Whether to normalize the adjacency with the method from diffusion maps
    scale_embedding : bool or string, optional, default=False
        Whether to scale the embedding.
        If True or 'LE', default scaling from the Laplacian Eigenmaps method.
        If 'CTD', Commute Time Distance based scaling (1/sqrt(lambda_k)) used.
        If 'heuristic', use 1/sqrt(k) for each dimension k=1..n_components.
    drop_first : bool, optional, default=True
        Whether to drop the first eigenvector. For spectral embedding, this
        should be True as the first eigenvector should be constant vector for
        connected graph, but for spectral clustering, this should be kept as
        False to retain the first eigenvector.
    Returns
    -------
    embedding : array, shape=(n_samples, n_components)
        The reduced samples.
    Notes
    -----
    Spectral Embedding (Laplacian Eigenmaps) is most useful when the graph
    has one connected component. If there graph has many components, the first
    few eigenvectors will simply uncover the connected components of the graph.
    References
    ----------
    * https://en.wikipedia.org/wiki/LOBPCG
    * Toward the Optimal Preconditioned Eigensolver: Locally Optimal
      Block Preconditioned Conjugate Gradient Method
      Andrew V. Knyazev
      http://dx.doi.org/10.1137%2FS1064827500366124
    """
    adjacency = check_similarity(adjacency, normalize=norm_adjacency)

    try:
        from pyamg import smoothed_aggregation_solver
    except ImportError:
        if eigen_solver == "amg":
            raise ValueError("The eigen_solver was set to 'amg', but pyamg is "
                             "not available.")

    if eigen_solver is None:
        eigen_solver = 'arpack'
    elif eigen_solver not in ('arpack', 'lobpcg', 'amg'):
        raise ValueError("Unknown value for eigen_solver: '%s'."
                         "Should be 'amg', 'arpack', or 'lobpcg'"
                         % eigen_solver)

    random_state = check_random_state(random_state)

    n_nodes = adjacency.shape[0]
    # Whether to drop the first eigenvector
    if drop_first:
        n_components = n_components + 1

    if not _graph_is_connected(adjacency):
        warnings.warn("Graph is not fully connected, spectral embedding"
                      " may not work as expected.")

    if (eigen_solver == 'arpack' or eigen_solver != 'lobpcg' and
       (not sparse.isspmatrix(adjacency) or n_nodes < 5 * n_components)):
        try:

            laplacian, dd = compute_laplacian(adjacency, normed=norm_laplacian,
                                              return_diag=True)
            # Compute embedding. We compute the largest eigenvalue and then use
            # the opposite of the laplacian since computing the largest
            # eigenvalues is more efficient.
            (evals_max, _) = eigsh(laplacian, n_components, which='LM',
                                   tol=eigen_tol)
            maxval = evals_max.max()
            laplacian *= -1
            if sparse.isspmatrix(laplacian):
                diag_idx = (laplacian.row == laplacian.col)
                laplacian.data[diag_idx] += maxval
            else:
                laplacian.flat[::n_nodes + 1] += maxval
            lambdas, diffusion_map = eigsh(laplacian, n_components, which='LM',
                                           tol=eigen_tol)
            lambdas -= maxval
            lambdas *= -1
            idx = np.array(lambdas).argsort()
            d = lambdas[idx]
            embedding = diffusion_map.T[idx]
            if scale_embedding:
                if scale_embedding == 'CTD':
                    embedding[1:] = (embedding[1:, :].T * np.sqrt(1./d[1:])).T
                    # embedding = embedding.T
                elif scale_embedding == 'heuristic':
                    embedding = embedding.T * np.sqrt(1./np.arange(
                        1, n_components+1))
                    embedding = embedding.T
                else:
                    embedding *= dd

        except RuntimeError:
            warnings.warn("arpack did not converge. trying lobpcg instead."
                          " scale_embedding set to default.")
            # When submatrices are exactly singular, an LU decomposition
            # in arpack fails. We fallback to lobpcg
            eigen_solver = "lobpcg"

    if eigen_solver == 'amg':
        # Use AMG to get a preconditioner and speed up the eigenvalue
        # problem.
        # norm_laplacian='random_walk' does not work for the following,
        # replace by True
        if norm_laplacian:
            if norm_laplacian == 'unnormalized':
                norm_laplacian = False
            else:
                norm_laplacian = True
        laplacian, dd = compute_laplacian(adjacency, normed=norm_laplacian,
                                          return_diag=True)
        if not sparse.issparse(laplacian):
            warnings.warn("AMG works better for sparse matrices")
        # lobpcg needs double precision floats
        laplacian = check_array(laplacian, dtype=np.float64,
                                accept_sparse=True)
        laplacian = _set_diag(laplacian, 1, norm_laplacian)
        ml = smoothed_aggregation_solver(check_array(laplacian, 'csr'))
        M = ml.aspreconditioner()
        X = random_state.rand(laplacian.shape[0], n_components + 1)
        X[:, 0] = dd.ravel()
        lambdas, diffusion_map = lobpcg(laplacian, X, M=M, tol=1.e-12,
                                        largest=False)
        if scale_embedding:
            embedding = diffusion_map.T * dd
        else:
            embedding = diffusion_map.T
        if embedding.shape[0] == 1:
            raise ValueError

    elif eigen_solver == "lobpcg":
        # norm_laplacian='random_walk' does not work for the following,
        # replace by True
        if norm_laplacian:
            if norm_laplacian == 'unnormalized':
                norm_laplacian = False
            else:
                norm_laplacian = True
        laplacian, dd = compute_laplacian(adjacency, normed=norm_laplacian,
                                          return_diag=True)
        # lobpcg needs double precision floats
        laplacian = check_array(laplacian, dtype=np.float64,
                                accept_sparse=True)
        if n_nodes < 5 * n_components + 1:
            # see note above under arpack why lobpcg has problems with small
            # number of nodes
            # lobpcg will fallback to eigh, so we short circuit it
            if sparse.isspmatrix(laplacian):
                laplacian = laplacian.toarray()
            lambdas, diffusion_map = eigh(laplacian)
            embedding = diffusion_map.T[:n_components] * dd
        else:
            laplacian = _set_diag(laplacian, 1, norm_laplacian)
            # We increase the number of eigenvectors requested, as lobpcg
            # doesn't behave well in low dimension
            X = random_state.rand(laplacian.shape[0], n_components + 1)
            X[:, 0] = dd.ravel()
            lambdas, diffusion_map = lobpcg(laplacian, X, tol=1e-15,
                                            largest=False, maxiter=2000)
            if scale_embedding:
                embedding = diffusion_map.T[:n_components] * dd
            else:
                embedding = diffusion_map.T[:n_components]
            if embedding.shape[0] == 1:
                raise ValueError

    embedding = _deterministic_vector_sign_flip(embedding)

    if drop_first:
        return embedding[1:n_components].T
    else:
        return embedding[:n_components].T


if __name__ == '__main__':

    from time import time
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mdso import SimilarityMatrix

# ############ Synthetic example ################
    # Set parameters for data generation
    t0 = time()
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
    t1 = time()
    print("Generated similarity matrix -- {}".format(t1-t0))
    # Check it runs fine with different options
    norm_lap_opts = ['unnormalized', 'symmetric', 'random_walk']
    scaling_opts = [True, 'CTD', 'heuristic']
    norm_adj_opts = ['coifman', None]
    for norm_lap in norm_lap_opts:
        for scale in scaling_opts:
            for norm_adj in norm_adj_opts:
                t_b = time()
                embedding = spectral_embedding(mat, norm_laplacian=norm_lap,
                                               scale_embedding=scale,
                                               norm_adjacency=norm_adj)
                print("Computed embedding with norm_lap : {}, \
                      scale_embedding : {}, in {}s.".format(norm_lap,
                                                            scale,
                                                            time()-t_b))
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
               c=np.arange(n))
    plt.show()

# ############ Synthetic example ################
    # t0 = time()
    embedding = spectral_embedding(new_mat, norm_laplacian=False,
                                   scale_embedding=False, eigen_solver='amg',
                                   norm_adjacency='coifman')
    # t1 = time()
    # print('computed embedding in {}'.format(t1 - t0))
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
    #            c=these_inv_perm)
    # plt.show()
    # fig = plt.figure()
    # plt.scatter(embedding[:, 0], embedding[:, 1],
    #             c=these_inv_perm)
    # plt.show()
