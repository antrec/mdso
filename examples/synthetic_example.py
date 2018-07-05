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
ampl_noise = 1  # amplitude of the noise
type_similarity = 'CircularStrongDecrease'  # type of synthetic similarity matrix
# ("Linear" [vs "Circular"], "Banded" [vs "StrongDecrease"])
apply_perm = True  # randomly permute the matrix, so that the ground truth is
# not the trivial permutation (1, ..., n).

# Set parameters for the ordering algorithm
k_nbrs = 10  # number of neighbors in the local linear fit in the embedding
n_components = 3  # number of dimensions of the embedding
circular = True if type_similarity[0] == 'C' else False  # circular or linear
scaled = 'heuristic'  # whether or not to scale the coordinates of the
# embedding so that the larger dimensions have fewer importance

# Build data matrix
data_gen = SimilarityMatrix()
data_gen.gen_matrix(n, type_matrix=type_similarity, apply_perm=apply_perm,
                    noise_ampl=ampl_noise, law=type_noise)

# Call Spectral Ordering method
scaled=False
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

print(reord_method.embedding.mean(axis=0))
print(reord_method.new_embedding.mean(axis=0))
# reord_method.new_embedding -= reord_method.new_embedding[:, :].mean(axis=0)
plt.figure()
plt.scatter(reord_method.new_embedding[:, 0],
            reord_method.new_embedding[:, 1],
            c=data_gen.true_perm)
plt.show()


from scipy.sparse import coo_matrix
my_perm = reord_method.fit_transform(coo_matrix(data_gen.sim_matrix))
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

print(reord_method.embedding.mean(axis=0))
print(reord_method.new_embedding.mean(axis=0))
# reord_method.new_embedding -= reord_method.new_embedding[:, :].mean(axis=0)
plt.figure()
plt.scatter(reord_method.new_embedding[:, 0],
            reord_method.new_embedding[:, 1],
            c=data_gen.true_perm)
plt.show()



from mdso import spectral_embedding
from scipy.sparse import coo_matrix
sim_matrix = data_gen.sim_matrix
plain_embedding = spectral_embedding(sim_matrix, drop_first=False)
rw_embedding = spectral_embedding(sim_matrix, drop_first=False,
                                  norm_laplacian='random_walk')
sparse_embedding = spectral_embedding(coo_matrix(sim_matrix), drop_first=False)
sparse_rw_embedding = spectral_embedding(coo_matrix(sim_matrix),
                                         norm_laplacian='random_walk',
                                         drop_first=False)

plain_embedding.mean(axis=0)
plain_embedding[:, 0]

rw_embedding.mean(axis=0)
rw_embedding[:, 0]

sparse_embedding.mean(axis=0)
sparse_embedding[:, 1] - plain_embedding[:, 1]

sparse_rw_embedding.mean(axis=0)
sparse_rw_embedding[:, 1] - rw_embedding[:, 1]

sim_matrix.dtype
sim_matrix.diagonal()


from mdso.utils import compute_laplacian
reg_lap, dd = compute_laplacian(sim_matrix, normed=False, return_diag=True)
rw_lap, rw_dd = compute_laplacian(sim_matrix, normed='random_walk', return_diag=True)

(dd == rw_dd).all()
dinv = 1./dd
Lrw = np.diag(dinv) @ reg_lap
(Lrw - rw_lap)

from scipy.sparse.linalg import eigsh
from scipy import sparse
laplacian = Lrw
n_nodes = laplacian.shape[0]
n_components = 20
eigen_tol = 0
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

embedding = embedding.T
embedding.mean(axis=0)


Lap = np.eye(n) - sim_matrix.dot(np.diag(1/sim_matrix.dot(np.ones(n))))
(Lap - rw_lap)
d1 = sim_matrix.dot(np.ones(n))
d2 = sim_matrix.sum(axis=0) - sim_matrix.diagonal()
(d2 - d1)
np.diag(rw_lap)
# compute embedding
%timeit [d, Vec] = np.linalg.eig(rw_lap)

(Vec[:, 1] - embedding[:, 1])



type(reord_method.new_sim)
sim_matrix = reord_method.new_sim.copy()
reg_lap, dd = compute_laplacian(sim_matrix, normed=False, return_diag=True)
sparse_sim = coo_matrix(sim_matrix).tolil()
rw_lap, rw_dd = compute_laplacian(sim_matrix, normed='unnormalized',
                                  return_diag=True)
rw_lap2, _ = compute_laplacian(sparse_sim, normed='random_walk',
                                  return_diag=True)
laplacian = rw_lap.copy()
n_nodes = laplacian.shape[0]
n_components = 8
eigen_tol = 1e-15
(evals_max, _) = eigsh(laplacian, n_components, which='LM',
                       tol=eigen_tol)
maxval = evals_max.max()
laplacian *= -1
if sparse.isspmatrix(laplacian):
    diag_idx = (laplacian.row == laplacian.col)
    laplacian.data[diag_idx] += maxval
else:
    laplacian.flat[::n_nodes + 1] += maxval
lambdas, diffusion_map = eigsh(laplacian, n_components, which='LM', tol=eigen_tol)
lambdas -= maxval
lambdas *= -1
idx = np.array(lambdas).argsort()
d = lambdas[idx]
embedding = diffusion_map.T[idx]
embedding = embedding.T
embedding.mean(axis=0)

[d, Vec] = np.linalg.eig(rw_lap)
idx = d.argsort()
d = d[idx]
Vec = Vec[:, idx]
Vec = Vec[:, :n_components]
Vec.mean(axis=0)


plt.figure()
plt.scatter(embedding[:, 1], embedding[:, 2], c=data_gen.true_perm)
plt.show()

plt.figure()
plt.scatter(Vec[:, 1], -Vec[:, 2], c=data_gen.true_perm)
plt.show()
#
# plt.figure()
# plt.scatter(reord_method.embedding[:, 0], reord_method.embedding[:, 1],
#                    c=data_gen.true_perm)
# plt.show()
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
