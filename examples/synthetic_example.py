#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run some experiments and visualize the results.
"""

from mdso import SpectralOrdering, SimilarityMatrix, evaluate_ordering
import numpy as np
import matplotlib.pyplot as plt

# Set parameters for data generation
n = 500  # size of matrix
type_noise = 'gaussian'  # distribution of the values of the noise
ampl_noise = 3  # amplitude of the noise
type_similarity = 'CircularStrongDecrease'  # type of synthetic similarity matrix
# ("Linear" [vs "Circular"], "Banded" [vs "StrongDecrease"])
apply_perm = True  # randomly permute the matrix, so that the ground truth is
# not the trivial permutation (1, ..., n).

# Set parameters for the ordering algorithm
k_nbrs = 15  # number of neighbors in the local linear fit in the embedding
n_components = 8  # number of dimensions of the embedding
circular = True if type_similarity[0] == 'C' else False  # circular or linear
scaled = 'heuristic'  # whether or not to scale the coordinates of the
# embedding so that the larger dimensions have fewer importance

# Build data matrix
data_gen = SimilarityMatrix()
data_gen.gen_matrix(n, type_matrix=type_similarity, apply_perm=apply_perm,
                    noise_ampl=ampl_noise, law=type_noise)

# Call Spectral Ordering method
reord_method = SpectralOrdering(n_components=n_components, k_nbrs=k_nbrs,
                                circular=circular, scale_embedding=scaled,
                                norm_laplacian='random_walk')
my_perm = reord_method.fit_transform(data_gen.sim_matrix)
reord_method.new_sim = reord_method.new_sim.toarray()
# reord_method.fit(data_gen.sim_matrix)

score = evaluate_ordering(my_perm, data_gen.true_perm,
                          circular=circular)
print("Kendall-Tau score = {}".format(score))

inv_perm = np.argsort(data_gen.true_perm)
# Display some results
fig, axes = plt.subplots(2, 2)
axes[0, 0].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)  # labels along the bottom edge are off
axes[0, 0].matshow(data_gen.sim_matrix[:, inv_perm][inv_perm, :])
axes[0, 0].set_title("raw matrix")
axes[0, 1].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)  # labels along the bottom edge are off
axes[0, 1].matshow(reord_method.new_sim[:, inv_perm][inv_perm, :])
axes[0, 1].set_title("new matrix")
axes[1, 0].scatter(reord_method.embedding[:, 0], reord_method.embedding[:, 1],
                   c=data_gen.true_perm)
axes[1, 0].set_title("2d embedding")
axes[1, 1].scatter(np.arange(data_gen.n),
                   data_gen.true_perm[reord_method.ordering])
axes[1, 1].set_title("perm vs ground truth")
plt.show()



fig = plt.figure(); plt.scatter(reord_method.embedding[:, 0], reord_method.embedding[:, 1],
c=data_gen.true_perm);
plt.xlabel('f_1', fontsize=16);
plt.ylabel('f_2', fontsize=16);
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, labelleft=False, left=False); 

figpath='/Users/antlaplante/THESE/manuscript/thesis/figs/reconstructing_by_spectral_figs/embedding_noisy_circular_ampl_3.pdf'
plt.savefig(figpath, bbox_inches='tight', transparent=True, dpi=150)

reord_method2 = SpectralOrdering(n_components=n_components, k_nbrs=k_nbrs,
                                circular=circular, scale_embedding=scaled,
                                norm_laplacian=None)
reord_method2.fit(reord_method.new_sim)
fig = plt.figure(); plt.scatter(reord_method2.embedding[:, 0], reord_method2.embedding[:, 1],
c=data_gen.true_perm);
plt.xlabel('f_1', fontsize=16);
plt.ylabel('f_2', fontsize=16);
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, labelleft=False, left=False); 
figpath='/Users/antlaplante/THESE/manuscript/thesis/figs/reconstructing_by_spectral_figs/embedding_cleaned_circular_ampl_3.pdf'
plt.savefig(figpath, bbox_inches='tight', transparent=True, dpi=150)