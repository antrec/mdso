#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Given a similarity matrix on sparse or numpy format, it creates a
Laplacian Embedding, for various type of graph Laplacian as well as
normalization.
So far the similarity is assumed to represent a fully connected graph.
'''
import warnings
import time
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, lobpcg
from scipy.sparse.csgraph import connected_components

# from ..utils.tools import check_similarity, compute_laplacian
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import _deterministic_vector_sign_flip
# import matplotlib.pyplot as plt
# from sinkhorn_knopp import sinkhorn_knopp as skp

""" Right now put everything here,
later put back to utils.tools."""

import numpy as np
from scipy.sparse import issparse, isspmatrix, coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import kendalltau


def is_symmetric(m):
    """Check if a sparse matrix is symmetric
    (from Saullo Giovani)

    Parameters
    ----------
    m : array or sparse matrix
        A square matrix.

    Returns
    -------
    check : bool
        The check result.

    """
    if m.shape[0] != m.shape[1]:
        raise ValueError('m must be a square matrix')

    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)

    r, c, v = m.row, m.col, m.data
    tril_no_diag = r > c
    triu_no_diag = c > r

    if triu_no_diag.sum() != tril_no_diag.sum():
        return False

    rl = r[tril_no_diag]
    cl = c[tril_no_diag]
    vl = v[tril_no_diag]
    ru = r[triu_no_diag]
    cu = c[triu_no_diag]
    vu = v[triu_no_diag]

    sortl = np.lexsort((cl, rl))
    sortu = np.lexsort((ru, cu))
    vl = vl[sortl]
    vu = vu[sortu]

    check = np.allclose(vl, vu)

    return check


def check_similarity(arr, normalize=False):
    '''
    check that the matrix is square and symmetric, and
    normalize with Coifman's normalization (TODO : Sinkhorn-Knopp norm.).
    '''
    # check for squareness
    (n, m) = arr.shape
    if n != m:
        raise ValueError('the similarity matrix is not square')
    # check for symmetry
    if issparse(arr):
        if arr.format in ('lil', 'dok'):
            mat = arr.tocoo()
            needs_copy = False
        else:
            mat = arr
            needs_copy = True
        if (mat.data < 0).any():
            raise ValueError('the similarity matrix has negative entries')
        if not is_symmetric(mat):
            raise ValueError('specified similarity matrix is not\
                symmetric')
        if normalize:
            if not normalize == 'coifman':
                print("Warning: normalize argument is present but not Coifman.\
                      So far only Coifman's norm. is implemented!")
            w = mat.sum(axis=0).getA1() - mat.diagonal()
            mat = mat.tocoo(copy=needs_copy)
            isolated_node_mask = (w == 0)
            w = np.where(isolated_node_mask, 1, w)
            mat.data /= w[mat.row]
            mat.data /= w[mat.col]
            return(csr_matrix(mat, dtype='float64'))
        else:
            return(csr_matrix(mat, copy=needs_copy, dtype='float64'))

    else:
        if np.any(arr < 0):
            raise ValueError('the similarity matrix has negative entries')
        if not np.allclose(arr, arr.T, atol=1e-6):
            raise ValueError('specified similarity matrix is not\
                symmetric.')
        if normalize:
            if not normalize == 'coifman':
                print("Warning: normalize argument is present but not Coifman.\
                      So far only Coifman's norm. is implemented!")
            mat = np.array(arr)
            w = mat.sum(axis=0) - mat.diagonal()
            isolated_node_mask = (w == 0)
            w = np.where(isolated_node_mask, 1, w)
            mat /= w
            mat /= w[:, np.newaxis]
            return(mat)
        else:
            return(np.array(arr))


def get_conn_comps(mat, min_cc_len=1):
    """
    Returns a list of connected components of the matrix mat by decreasing size
    of the connected components, for all cc of size larger or equal than
    min_cc_len
    """
    n_c, lbls = connected_components(mat)
    srt_lbls = np.sort(lbls)
    dif_lbls = np.append(np.array([1]), srt_lbls[1:] - srt_lbls[:-1])
    dif_lbls = np.append(dif_lbls, np.array([1]))
    switch_lbls = np.where(dif_lbls)[0]
    diff_switch = switch_lbls[1:] - switch_lbls[:-1]
    ord_ccs = np.argsort(-diff_switch)
    len_ccs = diff_switch[ord_ccs]
    ccs_l = []
    for (i, cc_idx) in enumerate(ord_ccs):
        if len_ccs[i] < min_cc_len:
            break
        ccs_l.append(np.where(lbls == cc_idx)[0])
    return ccs_l


def kendall_circular(true_perm, order_perm):
    '''
    TODO : make it faster for large n with a coarser grained slicing first,
    i.e., taking np.roll with a larger value than 1 and then zooming in.
    '''
    n = true_perm.shape[0]
    if (order_perm.shape[0] != n):
        print("wrong length of permutations in kendall_circular!")
    order_perm = true_perm[order_perm]
    id_perm = np.arange(n)
    scores = np.zeros(n)
    for i in range(n):
        scores[i] = abs(kendalltau(id_perm, order_perm)[0])
        order_perm = np.roll(order_perm, 1, axis=0)

    return(np.max(scores), np.argmax(scores))


def evaluate_ordering(perm, true_perm, criterion='kendall',
                      circular=False):
    '''
    evaluate the model.
    INPUT:
        - the ground truth permutation
        - the ordered_chain
    '''
    l1 = len(perm)
    l2 = len(true_perm)
    if not l1 == l2:
        print("Problem : perm of length {}, "
              "and true_perm of length {}".format(l1, l2))
        print("perm : {}".format(perm))
    if criterion == 'kendall':
        if circular:
            (score, _) = kendall_circular(true_perm, perm)
        else:
            score = abs(kendalltau(true_perm, np.argsort(perm))[0])
        return(score)


# Graph laplacian
# Copied from scipy.sparse and added norm='random_walk' option
def compute_laplacian(csgraph, normed=False, return_diag=False,
                      use_out_degree=False):
    """
    (Copied from scipy.sparse and added norm='random_walk' option)
    Return the Laplacian matrix of a directed graph.

    Parameters
    ----------
    csgraph : array_like or sparse matrix, 2 dimensions
        compressed-sparse graph, with shape (N, N).
    normed : bool or string, optional
        If True, then compute normalized Laplacian.
        If 'random_walk', compute the random-walk normalized Laplacian.
        If False, then unnormalized Laplacian.
    return_diag : bool, optional
        If True, then also return an array related to vertex degrees.
    use_out_degree : bool, optional
        If True, then use out-degree instead of in-degree.
        This distinction matters only if the graph is asymmetric.
        Default: False.

    Returns
    -------
    lap : ndarray or sparse matrix
        The N x N laplacian matrix of csgraph. It will be a numpy array (dense)
        if the input was dense, or a sparse matrix otherwise.
    diag : ndarray, optional
        The length-N diagonal of the Laplacian matrix.
        For the normalized Laplacian, this is the array of square roots
        of vertex degrees or 1 if the degree is zero.

    Notes
    -----
    The Laplacian matrix of a graph is sometimes referred to as the
    "Kirchoff matrix" or the "admittance matrix", and is useful in many
    parts of spectral graph theory.  In particular, the eigen-decomposition
    of the laplacian matrix can give insight into many properties of the graph.

    Examples
    --------
    >>> from scipy.sparse import csgraph
    >>> G = np.arange(5) * np.arange(5)[:, np.newaxis]
    >>> G
    array([[ 0,  0,  0,  0,  0],
           [ 0,  1,  2,  3,  4],
           [ 0,  2,  4,  6,  8],
           [ 0,  3,  6,  9, 12],
           [ 0,  4,  8, 12, 16]])
    >>> csgraph.laplacian(G, normed=False)
    array([[  0,   0,   0,   0,   0],
           [  0,   9,  -2,  -3,  -4],
           [  0,  -2,  16,  -6,  -8],
           [  0,  -3,  -6,  21, -12],
           [  0,  -4,  -8, -12,  24]])
    """
    if csgraph.ndim != 2 or csgraph.shape[0] != csgraph.shape[1]:
        raise ValueError('csgraph must be a square matrix or array')

    if normed and (np.issubdtype(csgraph.dtype, np.signedinteger)
                   or np.issubdtype(csgraph.dtype, np.uint)):
        csgraph = csgraph.astype(float)

    create_lap = _laplacian_sparse if isspmatrix(csgraph) else _laplacian_dense
    degree_axis = 1 if use_out_degree else 0
    lap, d = create_lap(csgraph, normed=normed, axis=degree_axis)
    if return_diag:
        return lap, d
    return lap


def _setdiag_dense(A, d):
    A.flat[::len(d)+1] = d


def _laplacian_sparse(graph, normed=False, axis=0):
    if graph.format in ('lil', 'dok'):
        m = graph.tocoo()
        needs_copy = False
    else:
        m = graph
        needs_copy = True
    w = m.sum(axis=axis).getA1() - m.diagonal()
    if normed:
        if normed == 'random_walk':
            m = m.tocoo(copy=needs_copy)
            isolated_node_mask = (w == 0)
            w = np.where(isolated_node_mask, 1, w)
            m.data /= w[m.row]
            m.data *= -1
            m.setdiag(1 - isolated_node_mask)
        else:
            m = m.tocoo(copy=needs_copy)
            isolated_node_mask = (w == 0)
            w = np.where(isolated_node_mask, 1, np.sqrt(w))
            m.data /= w[m.row]
            m.data /= w[m.col]
            m.data *= -1
            m.setdiag(1 - isolated_node_mask)
    else:
        if m.format == 'dia':
            m = m.copy()
        else:
            m = m.tocoo(copy=needs_copy)
        m.data *= -1
        m.setdiag(w)
    return m, w


def _laplacian_dense(graph, normed=False, axis=0):
    m = np.array(graph)
    np.fill_diagonal(m, 0)
    w = m.sum(axis=axis)
    if normed:
        if normed == 'random_walk':
            isolated_node_mask = (w == 0)
            w = np.where(isolated_node_mask, 1, w)
            m /= w
            m *= -1
            _setdiag_dense(m, 1 - isolated_node_mask)
        else:
            isolated_node_mask = (w == 0)
            w = np.where(isolated_node_mask, 1, np.sqrt(w))
            m /= w
            m /= w[:, np.newaxis]
            m *= -1
            _setdiag_dense(m, 1 - isolated_node_mask)
    else:
        m *= -1
        _setdiag_dense(m, w)
    return m, w




def _graph_connected_component(graph, node_id):
    """Find the largest graph connected components that contains one
    given node
    Parameters
    ----------
    graph : array-like, shape: (n_samples, n_samples)
        adjacency matrix of the graph, non-zero weight means an edge
        between the nodes
    node_id : int
        The index of the query node of the graph
    Returns
    -------
    connected_components_matrix : array-like, shape: (n_samples,)
        An array of bool value indicating the indexes of the nodes
        belonging to the largest connected components of the given query
        node
    """
    n_node = graph.shape[0]
    if sparse.issparse(graph):
        # speed up row-wise access to boolean connection mask
        graph = graph.tocsr()
    connected_nodes = np.zeros(n_node, dtype=np.bool)
    nodes_to_explore = np.zeros(n_node, dtype=np.bool)
    nodes_to_explore[node_id] = True
    for _ in range(n_node):
        last_num_component = connected_nodes.sum()
        np.logical_or(connected_nodes, nodes_to_explore, out=connected_nodes)
        if last_num_component >= connected_nodes.sum():
            break
        indices = np.where(nodes_to_explore)[0]
        nodes_to_explore.fill(False)
        for i in indices:
            if sparse.issparse(graph):
                neighbors = graph[i].toarray().ravel()
            else:
                neighbors = graph[i]
            np.logical_or(nodes_to_explore, neighbors, out=nodes_to_explore)
    return connected_nodes


def _graph_is_connected(graph):
    """ Return whether the graph is connected (True) or Not (False)
    Parameters
    ----------
    graph : array-like or sparse matrix, shape: (n_samples, n_samples)
        adjacency matrix of the graph, non-zero weight means an edge
        between the nodes
    Returns
    -------
    is_connected : bool
        True means the graph is fully connected and False means not
    """
    if sparse.isspmatrix(graph):
        # sparse graph, find all the connected components
        n_connected_components, _ = connected_components(graph)
        return n_connected_components == 1
    else:
        # dense graph, find all connected components start from node 0
        return _graph_connected_component(graph, 0).sum() == graph.shape[0]


def _set_diag(laplacian, value, norm_laplacian):
    """Set the diagonal of the laplacian matrix and convert it to a
    sparse format well suited for eigenvalue decomposition
    Parameters
    ----------
    laplacian : array or sparse matrix
        The graph laplacian
    value : float
        The value of the diagonal
    norm_laplacian : bool
        Whether the value of the diagonal should be changed or not
    Returns
    -------
    laplacian : array or sparse matrix
        An array of matrix in a form that is well suited to fast
        eigenvalue decomposition, depending on the band width of the
        matrix.
    """
    n_nodes = laplacian.shape[0]
    # We need all entries in the diagonal to values
    if not sparse.isspmatrix(laplacian):
        if norm_laplacian:
            laplacian.flat[::n_nodes + 1] = value
    else:
        laplacian = laplacian.tocoo()
        if norm_laplacian:
            diag_idx = (laplacian.row == laplacian.col)
            laplacian.data[diag_idx] = value
        # If the matrix has a small number of diagonals (as in the
        # case of structured matrices coming from images), the
        # dia format might be best suited for matvec products:
        n_diags = np.unique(laplacian.row - laplacian.col).size
        if n_diags <= 7:
            # 3 or less outer diagonals on each side
            laplacian = laplacian.todia()
        else:
            # csr has the fastest matvec and is thus best suited to
            # arpack
            laplacian = laplacian.tocsr()
    return laplacian


def spectral_embedding(adjacency, n_components=8, eigen_solver=None,
                       random_state=None, eigen_tol=1e-15,
                       norm_laplacian=False, drop_first=True,
                       norm_adjacency=False, scale_embedding=True,
                       verb=0):
    """Project the sample on the first eigenvectors of the graph Laplacian.
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
    norm_laplacian : bool, optional, default=True
        If True, then compute normalized Laplacian.
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
                if scale_embedding == 'LE':
                    embedding *= dd
                elif scale_embedding == 'CTD':
                    embedding[1:] = (embedding[1:, :].T * np.sqrt(1./d[1:])).T
                    # embedding = embedding.T
                else:
                    embedding = embedding.T * np.sqrt(1./np.arange(
                        1, n_components+1))
                    embedding = embedding.T

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


    from mdso import MatrixGenerator

    # Set parameters for data generation
    n = 1000  # size of matrix
    type_noise = 'gaussian'  # distribution of the values of the noise
    ampl_noise = 1.5  # amplitude of the noise
    type_similarity = 'LinearStrongDecrease'  # type of synthetic similarity matrix
    # ("Linear" [vs "Circular"], "Banded" [vs "StrongDecrease"])
    apply_perm = True  # randomly permute the matrix, so that the ground truth is
    # not the trivial permutation (1, ..., n).

    # Set parameters for the ordering algorithm
    k_nbrs = 10  # number of neighbors in the local linear fit in the embedding
    dim = 10  # number of dimensions of the embedding
    circular = False  # whether we are running Circular or Linear Seriation
    scaled = True  # whether or not to scale the coordinates of the embedding so
    # that the larger dimensions have fewer importance

    # Build data matrix
    data_gen = MatrixGenerator()
    data_gen.gen_matrix(n, type_matrix=type_similarity, apply_perm=apply_perm,
                        noise_ampl=ampl_noise, law=type_noise)
    mat = data_gen.sim_matrix
    embedding = spectral_embedding(mat, norm_laplacian=False,
                                   scale_embedding=True, eigen_solver='amg')
    t0 = time()
    embedding = spectral_embedding(new_mat, norm_laplacian=False,
                                   scale_embedding=False, eigen_solver='arpack',
                                   norm_adjacency='coifman')

    t1 = time()
    print('computed embedding in {}'.format(t1 - t0))
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
               c=these_inv_perm)
    plt.show()

    fig = plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1],
                c=these_inv_perm)
    plt.show()
    # Embedding_graph()
    D = np.abs(np.random.normal(0, 1, (5, 5)))
    D = np.transpose(D) + D
    S = sp.csr_matrix(D)

    embeddi = make_laplacian_emb(D,
                                 3,
                                 type_laplacian='unnormalized',
                                 type_normalization='coifman',
                                 scaled=False)

    print(embeddi.shape)
    vizualize_embedding(embeddi)

    embeddi = make_laplacian_emb(S,
                                 3,
                                 type_laplacian='unnormalized',
                                 type_normalization='coifman',
                                 scaled=False)
    print(embeddi.shape)
    vizualize_embedding(embeddi)
