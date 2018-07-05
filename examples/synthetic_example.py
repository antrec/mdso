#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run some experiments and visualize the results.
"""

from mdso import SpectralOrdering, SimilarityMatrix, evaluate_ordering, \
SpectralBaseline
import numpy as np
import matplotlib.pyplot as plt

# Set parameters for data generation
n = 500  # size of matrix
type_noise = 'gaussian'  # distribution of the values of the noise
ampl_noise = 0.5  # amplitude of the noise
type_similarity = 'LinearStrongDecrease'  # type of synthetic similarity matrix
# ("Linear" [vs "Circular"], "Banded" [vs "StrongDecrease"])
apply_perm = True  # randomly permute the matrix, so that the ground truth is
# not the trivial permutation (1, ..., n).

# Set parameters for the ordering algorithm
k_nbrs = 20  # number of neighbors in the local linear fit in the embedding
n_components = 5  # number of dimensions of the embedding
circular = False  # whether we are running Circular or Linear Seriation
scaled = 'heuristic'  # whether or not to scale the coordinates of the
# embedding so that the larger dimensions have fewer importance

# Build data matrix
data_gen = SimilarityMatrix()
data_gen.gen_matrix(n, type_matrix=type_similarity, apply_perm=apply_perm,
                    noise_ampl=ampl_noise, law=type_noise)

# Call Spectral Ordering method
reord_method = SpectralOrdering(n_components=n_components, k_nbrs=k_nbrs,
                                circular=circular, scale_embedding=scaled,
                                norm_laplacian='unnormalized')
my_perm = reord_method.fit_transform(data_gen.sim_matrix)
reord_method.new_sim = reord_method.new_sim.todense()
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

base_spectral = SpectralBaseline(circular=False, scale_embedding=False)
other_order = base_spectral.fit_transform(reord_method.new_sim)
plt.figure()
plt.scatter(np.arange(data_gen.n),
            data_gen.true_perm[reord_method.ordering])
plt.show()

plt.figure()
plt.scatter(base_spectral.new_embedding_[:, 0],
            base_spectral.new_embedding_[:, 1],
            c=data_gen.true_perm)
plt.show()

plt.figure()
plt.matshow(reord_method.new_sim[:, inv_perm][inv_perm, :])
plt.show()

plt.figure()
plt.scatter(reord_method.embedding[:, 0], reord_method.embedding[:, 1],
                   c=data_gen.true_perm)
plt.show()
# fig = plt.figure()
# plt.matshow(data_gen.sim_matrix[:, inv_perm][inv_perm, :])
# myfigpath = '/Users/antlaplante/Dropbox/Seriation_DNA_Assembly/Talks/Symbiose18/Rmatrix.pdf'
# plt.savefig(myfigpath, transparent=True, bbox_inches='tight')
# plt.show()
#
# fig = plt.figure()
# plt.plot(reord_method.embedding[inv_perm[::-1], 0], 'o', mfc='none')
# myfigpath = '/Users/antlaplante/Dropbox/Seriation_DNA_Assembly/Talks/Symbiose18/fiedlerRmat.pdf'
# plt.savefig(myfigpath, transparent=True, bbox_inches='tight')
# plt.show()
