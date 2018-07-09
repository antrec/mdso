#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main functions for getting latent ordering through
the spectral embedding of a similarity matrix, as in
arXiv ...
"""
import warnings
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import issparse
from .spectral_embedding_ import spectral_embedding
from .gen_sim_from_embedding_ import gen_sim_from_embedding
from .utils import get_conn_comps
from .merge_conn_comp_ import merge_conn_comp


def get_linear_ordering(new_embedding):
    '''
    Baseline Spectral Linear Ordering Algorithm (Atkins)
    input : 1d spectral embedding (Fiedler vector)
    output : permutation that sorts the entries of the Fiedler vector.
    '''
    shape_ebd = np.shape(new_embedding)

    if len(shape_ebd) > 1:
        first_eigen = new_embedding[:, 0]
    else:
        first_eigen = new_embedding

    return(np.argsort(first_eigen))


def get_circular_ordering(new_embedding):
    '''
    Baseline Spectral Circular Ordering Algorithm (Coifman)
    input : 2d spectral embedding
    output : permutation that sorts the angles between the entries of the first
    and second eigenvectors.
    '''
    first_eigen = new_embedding[:, 0]
    second_eigen = new_embedding[:, 1]
    ratio_eigen = np.divide(second_eigen, first_eigen)
    eigen_angles = np.arctan(ratio_eigen)
    eigen_angles[np.where(first_eigen < 0)] += np.pi

    return(np.argsort(eigen_angles))


class SpectralBaseline():
    """
    Basic Spectral Ordering Algorithm.
    For Linear Seriation, uses Atkins' method [ref]
    For Circular Seriation, uses Coifman's method [ref]
    """
    def __init__(self, circular=False, norm_laplacian=None,
                 norm_adjacency=None, eigen_solver=None,
                 scale_embedding=False):
        self.circular = circular
        if circular:
            if not norm_laplacian:
                norm_laplacian = 'unnormalized'
                # norm_laplacian = 'random_walk'
        else:
            if not norm_laplacian:
                norm_laplacian = 'unnormalized'
        self.norm_laplacian = norm_laplacian
        self.norm_adjacency = norm_adjacency
        self.eigen_solver = eigen_solver
        self.scale_embedding = scale_embedding

    def fit(self, X):
        """
        /!\ X must be connected.

        """
        (n_cc, _) = connected_components(X)
        if n_cc > 1:
            raise ValueError("The input matrix is not connected")
        # Get 1d or 2d Spectral embedding to retrieve the latent ordering
        self.new_embedding_ = spectral_embedding(
            X, norm_adjacency=self.norm_adjacency,
            norm_laplacian=self.norm_laplacian,
            eigen_solver=self.eigen_solver,
            scale_embedding=self.scale_embedding)

        if self.circular:
            self.ordering_ = get_circular_ordering(self.new_embedding_)
        else:
            self.ordering_ = get_linear_ordering(self.new_embedding_)

        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.ordering_


class SpectralOrdering():
    """
    Main functions for getting latent ordering through
    the spectral embedding of a similarity matrix, as in
    arXiv ...

    Parameters
    ----------
    dim : int, default 10
        The number of dimensions of the spectral embedding.

    k_nbrs : int, default 15
        The number of nearest neighbors in the local alignment algorithm.

    type_laplacian : string, default "random_walk"
        type of normalization of the Laplacianm Can be "unnormalized",
        "random_walk", or "symmetric".

    norm_adjacency : str or bool, default 'coifman'
        If 'coifman', use the normalization of the similarity matrix,
        W = Dinv @ W @ Dinv, to account for non uniform sampling of points on
        a 1d manifold (from Lafon and Coifman's approximation of the Laplace
        Beltrami operator)
        TODO : also implement the 'sinkorn' normalization

    scaled : string or boolean, default True
        if scaled is False, the embedding is just the concatenation of the
        eigenvectors of the Laplacian, i.e., all dimensions have the same
        weight.
        if scaled is "CTD", the k-th dimension of the spectral embedding
        (k-th eigenvector) is re-scaled by 1/sqrt(lambda_k), in relation
        with the commute-time-distance.
        If scaled is True or set to another string than "CTD", then the
        heuristic scaling 1/sqrt(k) is used instead.

    min_cc_len : int, default 10
        if the new similarity matrix is disconnected, keep only connected
        components of size larger than min_cc_len

    merge_if_ccs : bool, default False
        if the new similarity matrix is disconnected


    Attributes
        ----------
        embedding : array-like, (n_pts, dim)
            spectral embedding of the input matrix.

        new_sim : array-like, (n_pts, n_pts)
            new similarity matrix

        dense : boolean
            whether the input matrix is dense or not.
            If it is, then new_sim is also returned dense (otherwise sparse).
    """
    def __init__(self, n_components=8, k_nbrs=15, norm_adjacency='coifman',
                 norm_laplacian='unnormalized', scale_embedding='heuristic',
                 new_sim_norm_by_count=False, new_sim_norm_by_max=True,
                 new_sim_type=None, preprocess_only=False, min_cc_len=1,
                 merge_if_ccs=False, eigen_solver=None, circular=False):

        self.n_components = n_components
        self.k_nbrs = k_nbrs
        self.norm_adjacency = norm_adjacency
        self.norm_laplacian = norm_laplacian
        self.scale_embedding = scale_embedding
        self.new_sim_norm_by_count = new_sim_norm_by_count
        self.new_sim_norm_by_max = new_sim_norm_by_max
        self.new_sim_type = new_sim_type
        self.preprocess_only = preprocess_only
        self.min_cc_len = min_cc_len
        self.merge_if_ccs = merge_if_ccs
        self.eigen_solver = eigen_solver
        self.circular = circular

    def merge_connected_components(self, X, mode='similarity'):
        """
        If the new similarity matrix (computed from the Laplacian embedding)
        is disconnected, then the algorithm will only find partial orderings
        (one in each connected component).
        This method merges the partial orderings into one by using the
        original (connected) similarity matrix, or the embedding itself.
        """
        if not type(self.partial_orderings) == list:
            raise TypeError("self.ordering should be a list (of lists)")
        if not type(self.partial_orderings[0]) == list:
            return self
        else:
            self.ordering = merge_conn_comp(self.partial_orderings, X,
                                            h=self.k_nbrs)
            self.ordering = np.array(self.ordering)
        return self

    def fit(self, X):
        """
        Creates a Laplacian embedding and a new similarity matrix
        """

        if self.n_components == 1:
            # If n_components == 1, just run the baseline spectral method
            ordering_algo = SpectralBaseline(
                circular=self.circular,
                norm_adjacency=self.norm_adjacency,
                eigen_solver=self.eigen_solver)
            ordering_algo.fit(X)
            self.ordering = ordering_algo.ordering_
            return(self)

        else:
            # Compute the Laplacian embedding
            self.embedding = spectral_embedding(
                X, n_components=self.n_components,
                norm_laplacian=self.norm_laplacian,
                norm_adjacency=self.norm_adjacency,
                scale_embedding=self.scale_embedding,
                eigen_solver=self.eigen_solver)

            # Get the cleaned similarity matrix from the embedding
            self.new_sim = gen_sim_from_embedding(
                self.embedding, k_nbrs=self.k_nbrs,
                norm_by_max=self.new_sim_norm_by_max,
                norm_by_count=self.new_sim_norm_by_count,
                type_simil=self.new_sim_type)

        # Get the latent ordering from the cleaned similarity matrix
        if not self.preprocess_only:
            # Make sure we have only one connected component in new_sim
            ccs, n_c = get_conn_comps(self.new_sim)
            if n_c == 1:
                # Create a baseline spectral seriation solver
                ordering_algo = SpectralBaseline(
                    circular=self.circular, norm_adjacency=self.norm_adjacency,
                    eigen_solver=self.eigen_solver)
                ordering_algo.fit(self.new_sim)
                self.ordering = ordering_algo.ordering_
                self.new_embedding = ordering_algo.new_embedding_
            else:
                warning_msg = "new similarity disconnected. Reordering"\
                              " connected components."
                warnings.warn(warning_msg)
                # Create a baseline spectral seriation solver
                # Set circular=False because if we have broken the circle
                # in several pieces, we only have local linear orderings.
                ordering_algo = SpectralBaseline(
                    circular=False, norm_adjacency=self.norm_adjacency,
                    eigen_solver=self.eigen_solver)

                self.partial_orderings = []
                # Convert sparse matrix to lil format for slicing
                if issparse(X):
                    self.new_sim = self.new_sim.tolil()
                else:
                    self.new_sim = self.new_sim.toarray()
                for cc_idx, in_cc in enumerate(ccs):
                    if len(in_cc) < self.min_cc_len:
                        break
                    # Change the eigen_solver depending on size and sparsity
                    if issparse(X) and len(in_cc) > 5000:
                        ordering_algo.eigen_solver = 'amg'
                    else:
                        ordering_algo.eigen_solver = 'arpack'
                    ordering_algo.fit(self.new_sim[in_cc, :][:, in_cc])
                    self.partial_orderings.append(
                        in_cc[ordering_algo.ordering_])

                self.partial_orderings = [list(
                    partial_order) for partial_order in self.partial_orderings]

                if self.merge_if_ccs:
                    self.merge_connected_components(X)
                else:
                    self.ordering = self.partial_orderings

        return self

    def fit_transform(self, X):
        """

        """
        self.fit(X)
        return self.ordering
