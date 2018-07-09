#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run experiments and visualize the results.
"""
import os
import subprocess
import numpy as np
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mdso import SpectralOrdering, merge_conn_comp
from mdso.utils import get_conn_comps


# ############ Real example : DNA data from E. coli ONT reads ################
# Get similarity matrix
examples_dir = os.path.dirname(os.path.abspath(__file__))
# /!\ THE PREVIOUS LINE DOES NOT WORK WITHIN A CONSOLE (IPYTHON, HYDROGEN...),
# BUT ONLY WHEN THE WHOLE FILE IS EXECUTED AS A SCRIPT. SET YOUR OWN PATH TO
# THE DNA FILES IF NEEDED OR GO TO THE MDSO DIRECTORY AND UNCOMMENT
# THE FOLLOWING LINE. /!\
# examples_dir = os.getcwd() + '/examples/'
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
# We set the threshold around the median
ovlp_thr = np.percentile(vvs, 60)
over_thr = np.where(vvs > ovlp_thr)[0]
sim_mat = coo_matrix((vvs[over_thr],
                     (iis[over_thr], jjs[over_thr])),
                     shape=loader['shape'],
                     dtype='float64').tocsr()
# Restrict to main connected component if disconnected similarity
ccs, n_c = get_conn_comps(sim_mat, min_cc_len=10)
sub_idxs = ccs[0]
new_mat = sim_mat.tolil()[sub_idxs, :]
new_mat = new_mat.T[sub_idxs, :].T
# Get reference ordering
sub_pos = positions[sub_idxs]
true_perm = np.argsort(sub_pos)
true_inv_perm = np.argsort(true_perm)

# Set parameters for Spectral Ordering method
scale_embedding = False
k_nbrs = 15
circular = True
eigen_solver = 'amg'  # faster than arpack on large sparse matrices.
# requires pyamg package (conda install pyamg or pip install pyamg)
norm_adjacency = 'coifman'  # yields better results in practice
norm_laplacian = False  # normalization of the laplacian seems to mess things
# up for large sparse matrices
merge_if_ccs = True  # the new similarity matrix may be disconnected
reord_method = SpectralOrdering(scale_embedding=scale_embedding, k_nbrs=k_nbrs,
                                circular=circular, eigen_solver=eigen_solver,
                                norm_adjacency=norm_adjacency,
                                norm_laplacian=norm_laplacian,
                                merge_if_ccs=merge_if_ccs,
                                n_components=8)
# Run the method
reord_method.fit(new_mat)

# Plot the laplacian embedding
embedding = reord_method.embedding
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
           c=true_inv_perm)
plt.title("3d embedding of DNA overlap based similarity matrix")
# plt.show()

# Plot the ordering found in each connected component of the new similarity
# matrix generated from the embedding
my_ords = reord_method.partial_orderings
fig = plt.figure()
for sub_ord in my_ords[:]:
    # plt.plot(np.sort(these_inv_perm[sub_ord]), these_pos[sub_ord], 'o')
    plt.plot(np.sort(true_inv_perm[sub_ord]), sub_pos[sub_ord], 'o')
    plt.title("ordering found in each conn. comp. of embedding-based"
              "similarity")
# plt.show()

# The method merge_conn_comp uses the input matrix new_mat to merge the
# connected components. In practice, though, the matrix new_mat was thresholded
# with a given percentile. Thus, the method may still yield several contigs.
# We can call the merging method with the non-thresholded matrix to make sure
# we end up with a single contig.
my_ords = list(reord_method.ordering)
if type(my_ords[0]) == list:  # then we have several contigs
    fig = plt.figure()
    for sub_ord in my_ords:
        plt.plot(np.sort(true_inv_perm[sub_ord]), sub_pos[sub_ord], 'o')
        plt.title("ordering found by merging conn. comp. of embedding-based"
                  "similarity with the thresholded overlap-based similarity")
    # plt.show()

# Run merge_conn_comp with the non-thresholded matrix
raw_mat = coo_matrix((vvs, (iis, jjs)),
                     shape=loader['shape'],
                     dtype='float64').tocsr()
# Restrict to main connected component if disconnected similarity
sub_raw_mat = raw_mat.tolil()[sub_idxs, :]
sub_raw_mat = sub_raw_mat.T[sub_idxs, :].T
newperm = merge_conn_comp(reord_method.partial_orderings, sub_raw_mat,
                          h=k_nbrs)
fig = plt.figure()
plt.plot(np.sort(true_inv_perm[newperm]), sub_pos[newperm], 'o')
plt.title("ordering found by merging conn. comp. of embedding-based"
          "similarity with the raw overlap-based similarity")

plt.show()
