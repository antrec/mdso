#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run experiments and visualize the results.
"""
import os
import subprocess
from time import time
import numpy as np
from scipy.sparse import coo_matrix, find
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mdso import spectral_embedding
from mdso.utils import get_conn_comps
from mdso.gen_sim_from_embedding_ import gen_sim_from_embedding, find_isolated


# ############ Real example : DNA data from E. coli ONT reads ################
t0 = time()
# Get similarity matrix
# examples_dir = os.path.dirname(os.path.abspath(__file__))
examples_dir = os.getcwd() + '/examples/'
ecoli_data_dir = examples_dir + '/e_coli/ecoli_data'
sim_mat_fn = ecoli_data_dir + '/sim_mat.npz'

if not os.path.exists(sim_mat_fn):
    get_sim_script = examples_dir + '/e_coli/build_ecoli_sim_mat.py'
    print("File {} not found. Running the script {} to get the \
          similarity matrix".format(sim_mat_fn, get_sim_script))
    subprocess.call(['python', get_sim_script])

loader = np.load(sim_mat_fn)
iis = loader['row']
jjs = loader['col']
vvs = loader['data']
n_reads = loader['shape'][0]
positions = loader['pos']
# Remove lowest overlap score values to clean things up
ovlp_thr = np.percentile(vvs, 50)
over_thr = np.where(vvs > ovlp_thr)[0]
sim_mat = coo_matrix((vvs[over_thr],
                     (iis[over_thr], jjs[over_thr])),
                     shape=loader['shape'],
                     dtype='float64').tocsr()
t1 = time()
print("Built similarity matrix - {}s".format(t1-t0))
# Restrict to main connected component if disconnected similarity
ccs, n_c = get_conn_comps(sim_mat, min_cc_len=10)
sub_idxs = ccs[0]
new_mat = sim_mat.tolil()[sub_idxs, :]
new_mat = new_mat.T[sub_idxs, :].T
sub_pos = positions[sub_idxs]
true_perm = np.argsort(sub_pos)
true_inv_perm = np.argsort(true_perm)
t2 = time()
print("Restricted to main conn. comp. - {}s".format(t2-t1))
# Compute the embedding
# Remark : amg is a lot faster than arpack on this large sparse
# similarity matrix. It requires the pyamg package (install with
# ```conda install pyamg``` or ```pip install pyamg``` for instance).
# If not available, use arpack.
try:
    embedding = spectral_embedding(new_mat, norm_laplacian=False,
                                   scale_embedding=False,
                                   eigen_solver='amg',
                                   norm_adjacency='coifman')
except Exception:
    embedding = spectral_embedding(new_mat, norm_laplacian=False,
                                   scale_embedding=False,
                                   eigen_solver='arpack',
                                   norm_adjacency='coifman')
t3 = time()
print("Computed Laplacian embedding - {}s".format(t3-t2))
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
           c=true_inv_perm)
plt.title("3d embedding of DNA overlap based similarity matrix")

plt.show()

t4 = time()
ebd_sim = gen_sim_from_embedding(embedding, k_nbrs=20)
t5 = time()
print("Computed new similarity from embedding - {}s".format(t5-t4))
is_aligned_to_nbrs, _ = find_isolated(embedding, k_nbrs=20)
t6 = time()
print("Found isolated nodes in embedding (could be done for cheaper)"
      " - {}s".format(t6-t5))

plt.plot(is_aligned_to_nbrs, 'o', mfc='none')
plt.show()

n_out = int(new_mat.shape[0] // 100)
isolated_nodes = np.argsort(-is_aligned_to_nbrs)[:n_out]

sim_mat_new = new_mat.tolil(copy=True)
for idx in isolated_nodes:
    sub_mat = sim_mat_new[idx, :]
    (_, jj, vv) = find(sub_mat)
    srt_v = np.argsort(-vv)
    sim_mat_new[idx, srt_v[1:]] = 0
    sim_mat_new[srt_v[1:], idx] = 0

sim_mat_new = ebd_sim.copy()
ccs2, _ = get_conn_comps(sim_mat_new)
cc_idx = 0
for cc_idx in range(len(ccs2)):
    print(len(ccs2[cc_idx]))
    these_pos = sub_pos[ccs2[cc_idx]]
    these_perm = np.argsort(these_pos)
    these_inv_perm = np.argsort(these_perm)

    my_mat = sim_mat_new.tolil()[ccs2[cc_idx], :]
    my_mat = my_mat.T[ccs2[cc_idx], :].T

    try:
        sub_embedding = spectral_embedding(my_mat, norm_laplacian=False,
                                       scale_embedding=False,
                                       eigen_solver='amg',
                                       norm_adjacency='coifman')
    except Exception:
        sub_embedding = spectral_embedding(my_mat, norm_laplacian=False,
                                       scale_embedding=False,
                                       eigen_solver='arpack',
                                       norm_adjacency='coifman')
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(sub_embedding[:, 0], sub_embedding[:, 1], sub_embedding[:, 2],
    #            c=these_inv_perm)
    # plt.title("3d embedding of DNA overlap based similarity matrix")
    #
    # plt.show()

    fig = plt.figure()
    plt.scatter(sub_embedding[:, 0], sub_embedding[:, 1],
               c=these_inv_perm)
    plt.title("2d embedding of DNA overlap based similarity matrix")
    plt.show()


from mdso import SpectralOrdering
reord_method = SpectralOrdering(scale_embedding=False, k_nbrs=20,
                                circular=True, eigen_solver='amg',
                                norm_adjacency='coifman',
                                norm_laplacian=False,
                                merge_if_ccs=True,
                                n_components=8)
%time reord_method.fit(new_mat)

my_ords = reord_method.partial_orderings
type(my_ords)
my_ords = [list(el) for el in my_ords]
len(my_ords)
sum([len(ord) for ord in my_ords])
new_mat.shape
if type(my_ords) == list:
    fig = plt.figure()
    for sub_ord in my_ords[:]:
        # plt.plot(np.sort(these_inv_perm[sub_ord]), these_pos[sub_ord], 'o')
        plt.plot(np.sort(true_inv_perm[sub_ord]), sub_pos[sub_ord], 'o')
    plt.show()

from mdso import merge_conn_comp

raw_mat = coo_matrix((vvs,
                     (iis, jjs)),
                     shape=loader['shape'],
                     dtype='float64').tocsr()
# Restrict to main connected component if disconnected similarity
sub_raw_mat = sim_mat.tolil()[sub_idxs, :]
sub_raw_mat = sub_raw_mat.T[sub_idxs, :].T

newperm = reord_method.partial_orderings
h = 5
while len(newperm) < sub_raw_mat.shape[0]:
    h = int(h * 1.5)
    if h > sub_raw_mat.shape[0]:
        break
    newperm = merge_conn_comp(newperm, sub_raw_mat, [], h=h, seed_comp_idx=1)

    len(newperm)


newperm = merge_conn_comp(reord_method.partial_orderings, sub_raw_mat, embedding, h=20, mode='embedding')
plt.plot(np.sort(true_inv_perm[newperm]), sub_pos[newperm], 'o')
plt.show()



fig = plt.figure()
ax = Axes3D(fig)
for sub_ord in reord_method.partial_orderings:
    ax.scatter(embedding[sub_ord, 0], embedding[sub_ord, 1], embedding[sub_ord, 2])
plt.title("3d embedding of DNA overlap based similarity matrix")

plt.show()
