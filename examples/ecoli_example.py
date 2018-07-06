#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run experiments and visualize the results.
"""
import os
import subprocess
from time import time
import numpy as np
from scipy.sparse import coo_matrix
from matplotlib.pyplot import plt
from mpl_toolkits.mplot3d import Axes3D
from mdso import spectral_embedding
from mdso.utils import get_conn_comps


# ############ Real example : DNA data from E. coli ONT reads ################
t0 = time()
# Get similarity matrix
examples_dir = os.path.dirname(os.path.abspath(__file__))
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
