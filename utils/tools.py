#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools for handling dense, sparse, connected and disconnected
similarity matrices
"""
import numpy as np
from scipy.sparse import issparse, isspmatrix, coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import kendalltau


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
