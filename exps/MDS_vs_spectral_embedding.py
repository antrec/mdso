#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run some experiments and visualize the results.
"""

from mdso import SpectralOrdering, SimilarityMatrix, evaluate_ordering
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pathlib
from run_experiments import run_synthetic_exps, plot_from_res


n = 500
k = 7
dim = 8
ampl_l = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
n_avrg = 100
type_matrix_l = ['CircularStrongDecrease', 'LinearStrongDecrease',
                    'CircularBanded', 'LinearBanded']
scaled = 'heuristic'
embedding_method_l = ['spectral', 'cMDS', 'MDS', 'TSNE']

exps_dir = os.path.dirname(os.path.abspath(__file__))
save_res_dir = exps_dir + '/results'
if not os.path.exists(save_res_dir):
    os.mkdir(save_res_dir)

# Run experiments
run_synthetic_exps(n, k, dim, ampl_l, type_matrix_l, scaled,
                    n_avrg=n_avrg, save_res_dir=save_res_dir,
                    norm_laplacian_l='random_walk',
                    embedding_method_l=embedding_method_l)



# # Set parameters for data generation
# n = 500  # size of matrix
# type_noise = 'gaussian'  # distribution of the values of the noise
# ampl_noise = 1.0  # amplitude of the noise
# type_similarity = 'CircularStrongDecrease'  # type of synthetic similarity matrix
# # ("Linear" [vs "Circular"], "Banded" [vs "StrongDecrease"])
# apply_perm = True  # randomly permute the matrix, so that the ground truth is
# # not the trivial permutation (1, ..., n).

# # Set parameters for the ordering algorithm
# k_nbrs = 7  # number of neighbors in the local linear fit in the embedding
# n_components = 8  # number of dimensions of the embedding
# circular = True if type_similarity[0] == 'C' else False  # circular or linear
# scaled = 'heuristic'  # whether or not to scale the coordinates of the
# # embedding so that the larger dimensions have fewer importance

# embedding_method = 'TSNE'
# norm_laplacian = 'random-walk'
# k = k_nbrs
# dim = n_components
# ampl = ampl_noise
# type_matrix = type_similarity
# n_avrg = 100

# save_res_dir = '/Users/antlaplante/THESE/RobustSeriationEmbedding/mdso/exps/results/'
# # Create directory for results if it does not already exist
# if save_res_dir:
#     pathlib.Path(save_res_dir).mkdir(parents=True, exist_ok=True)

# print("n:{}, k:{}, dim:{}, ampl:{}, "
#     "type_matrix:{}, scaled:{}, "
#     "norm_laplacian:{}, embd method:{}, "
#     "n_avrg:{}".format(n, k, dim, ampl,
#                         type_matrix,
#                         scaled,
#                         norm_laplacian,
#                         embedding_method,
#                         n_avrg))

# if save_res_dir:
#     # Check if the file already exists and read results if so
#     fn = "n_{}-k_{}-dim_{}-ampl_{}" \
#         "-type_mat_{}-embedding_{}" \
#         "-scaled_{}-norm_laplacian_{}-n_avrg_{}." \
#         "res".format(n, k, dim, ampl,
#                     type_matrix,
#                     embedding_method,
#                     scaled,
#                     norm_laplacian, n_avrg)
#     fn = save_res_dir + "/" + fn
#     if os.path.isfile(fn):
#         (mn, stdv) = fetch_res(n, k, dim, ampl,
#                             type_matrix,
#                             scaled,
#                             norm_laplacian,
#                             n_avrg,
#                             save_res_dir,
#                             embedding_method)
#     else:
#         # Run the experiments if the result file does not already exist
#         (mn, stdv, scores) = run_one_exp(n, k, dim, ampl, type_matrix, n_avrg,
#                                         norm_laplacian=norm_laplacian,
#                                         scale_embedding=scaled,
#                                         embedding_method=embedding_method)
#     # Print results
#     print("MEAN_SCORE:{}, STD_SCORE:{}"
#         "".format(mn, stdv))
#     fh = open(fn, 'a')
#     print(mn, stdv, file=fh)
#     print(scores, file=fh)
#     fh.close()



# # np.random.seed(1)

# # # Build data matrix
# # data_gen = SimilarityMatrix()
# # data_gen.gen_matrix(n, type_matrix=type_similarity, apply_perm=apply_perm,
# #                     noise_ampl=ampl_noise, law=type_noise)

# # true_perm = np.array(data_gen.true_perm)
# # inv_true_perm = np.argsort(true_perm)

# # # Call Spectral Ordering method with various embeddings
# # ebd_methods = ['spectral', 'cMDS', 'MDS', 'TSNE']
# # scores = {}
# # for embedding_method in ebd_methods:
# #     if embedding_method == 'TSNE':
# #         n_components = 2
# #     reord_method = SpectralOrdering(n_components=n_components, k_nbrs=k_nbrs,
# #                                     circular=circular, scale_embedding=scaled,
# #                                     norm_laplacian='random_walk',
# #                                     embedding_method=embedding_method,
# #                                     merge_if_ccs=True,
# #                                     norm_adjacency=False)
# #     my_perm = reord_method.fit_transform(data_gen.sim_matrix)
# #     # print(type(my_perm))

# #     embedding = reord_method.embedding
# #     fig = plt.figure()
# #     if n_components > 2:
# #         ax = Axes3D(fig)
# #         ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
# #                 c=true_perm)
# #     else:
# #         plt.scatter(embedding[:, 0], embedding[:, 1], c=true_perm)

# #     plt.title("%s" % (embedding_method))
# #     # plt.show()

# #     score = evaluate_ordering(my_perm, data_gen.true_perm,
# #                             circular=circular)
# #     scores[embedding_method] = score

# # msg = "Spectral - d=%d - KT=%1.3f \n\
# #        cMDS - d=%d - KT=%1.3f \n\
# #        MDS - d=%d - KT=%1.3f \n\
# #        TSNE - d=%d - KT=%1.3f \n" % (n_components, scores['spectral'],
# #                                      n_components, scores['cMDS'],
# #                                      n_components, scores['MDS'],
# #                                      n_components, scores['TSNE'])

# # print(msg)
# # plt.show()