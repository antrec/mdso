#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run some experiments and visualize the results.
"""

from mdso import SpectralOrdering, SimilarityMatrix, evaluate_ordering
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def run_one_exp(n, k, dim, ampl, type_matrix, n_avrg,
                type_noise='gaussian', norm_laplacian='random-walk',
                norm_adjacency=False, scale_embedding='heuristic',
                embedding_method='spectral'):
    """
    Run n_avrg experiments for a given set of parameters, and return the mean
    kendall-tau score and the associated standard deviation (among the
    n_avrg instances).
    """

    if type_matrix[0] == 'L':
        circular = False
    elif type_matrix[0] == 'C':
        circular = True
    else:
        raise ValueError("type matrix must be in ['LinearBanded',"
                         "'CircularBanded', 'LinearStrongDecrease',"
                         "'CircularStrongDecrease']")
    # Create matrix generator
    data_gen = SimilarityMatrix()
    # Create spectral solver
    reord_method = SpectralOrdering(n_components=dim, k_nbrs=k,
                                    norm_adjacency=norm_adjacency,
                                    norm_laplacian=norm_laplacian,
                                    scale_embedding=scale_embedding,
                                    circular=circular,
                                    merge_if_ccs=True,
                                    embedding_method=embedding_method)

    # Initialize array of results
    scores = np.zeros(n_avrg)
    for i_exp in range(n_avrg):
        np.random.seed(i_exp)
        data_gen.gen_matrix(n, type_matrix=type_matrix, apply_perm=True,
                            noise_ampl=ampl, law=type_noise)
        this_perm = reord_method.fit_transform(data_gen.sim_matrix)
        scores[i_exp] = evaluate_ordering(this_perm, data_gen.true_perm,
                                          circular=circular)

    return(scores.mean(), scores.std(), scores)





if __name__ == '__main__'
# Set parameters for data generation
n = 500  # size of matrix
type_noise = 'gaussian'  # distribution of the values of the noise
ampl_noise = 3  # amplitude of the noise
type_similarity = 'LinearStrongDecrease'  # type of synthetic similarity matrix
# ("Linear" [vs "Circular"], "Banded" [vs "StrongDecrease"])
apply_perm = True  # randomly permute the matrix, so that the ground truth is
# not the trivial permutation (1, ..., n).

# Set parameters for the ordering algorithm
k_nbrs = 7  # number of neighbors in the local linear fit in the embedding
n_components = 8  # number of dimensions of the embedding
circular = True if type_similarity[0] == 'C' else False  # circular or linear
scaled = 'heuristic'  # whether or not to scale the coordinates of the
# embedding so that the larger dimensions have fewer importance

np.random.seed(1)

# Build data matrix
data_gen = SimilarityMatrix()
data_gen.gen_matrix(n, type_matrix=type_similarity, apply_perm=apply_perm,
                    noise_ampl=ampl_noise, law=type_noise)

true_perm = np.array(data_gen.true_perm)
inv_true_perm = np.argsort(true_perm)

# Call Spectral Ordering method with various embeddings
ebd_methods = ['spectral', 'cMDS', 'MDS', 'TSNE']
scores = {}
for embedding_method in ebd_methods:
    if embedding_method == 'TSNE':
        n_components = 2
    reord_method = SpectralOrdering(n_components=n_components, k_nbrs=k_nbrs,
                                    circular=circular, scale_embedding=scaled,
                                    norm_laplacian='random_walk',
                                    embedding_method=embedding_method,
                                    merge_if_ccs=True,
                                    norm_adjacency=False)
    my_perm = reord_method.fit_transform(data_gen.sim_matrix)
    # print(type(my_perm))

    embedding = reord_method.embedding
    fig = plt.figure()
    if n_components > 2:
        ax = Axes3D(fig)
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                c=true_perm)
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], c=true_perm)

    plt.title("%s" % (embedding_method))
    # plt.show()

    score = evaluate_ordering(my_perm, data_gen.true_perm,
                            circular=circular)
    scores[embedding_method] = score

msg = "Spectral - d=%d - KT=%1.3f \n\
       cMDS - d=%d - KT=%1.3f \n\
       MDS - d=%d - KT=%1.3f \n\
       TSNE - d=%d - KT=%1.3f \n" % (n_components, scores['spectral'],
                                     n_components, scores['cMDS'],
                                     n_components, scores['MDS'],
                                     n_components, scores['TSNE'])

print(msg)
plt.show()