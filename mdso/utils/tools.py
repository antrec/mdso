#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools for handling dense, sparse, connected and disconnected
similarity matrices
"""
import numpy as np
from scipy.sparse import issparse, isspmatrix, coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components


def is_symmetric(m):
    """Check if a sparse matrix is symmetric
    (Saullo Giovani)

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
    if issparse(graph):
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
            if issparse(graph):
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
    if isspmatrix(graph):
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
    if not isspmatrix(laplacian):
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
