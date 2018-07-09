#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
With an initial connected similarity matrix, our preprocessing might split
up the new similarity matrix is several components.
In the case of DNA reconstruction, the user might know that its data comes from
a single contigue. Hence it is of interest to merge the various contigue our
pre-processing might reveal.

Here twigs are supposed to be list of index forming a connected component of
the similarity_matrix.

The main function is merge
INPUT:
    - new_embedding
    - old_similarity
    - list connected components: list of index forming a connected component of
    the similarity_matrix.
    - mode= choice between computing similarities between contigue endings with
    similarity or from distance in the new embedding
OUTPUT:
    - a permutation of [1,...,n]
'''
import warnings
import numpy as np
import copy
from scipy.sparse import issparse


def merge_conn_comp(l_connected, sim, h=10):
    if type(l_connected[-1]) == int:
        return(l_connected)
    n_cc = len(l_connected)
    l_out = l_connected.copy()
    a = np.zeros((2*n_cc, 2*n_cc))
    if issparse(sim):
        sim = sim.tolil()
    for i_cc in range(n_cc-1):
        for j_cc in range(i_cc+1, n_cc):
            for i_tip in ['begin', 'end']:
                for j_tip in ['begin', 'end']:
                    if i_tip == 'begin':
                        # filling beginning of i values
                        i_idxs = l_out[i_cc][:h]
                        this_i = 2*i_cc
                    elif i_tip == 'end':
                        # filling ending of i values
                        i_idxs = l_out[i_cc][-h:]
                        this_i = 2*i_cc + 1
                    if j_tip == 'begin':
                        # filling beginning of j values
                        j_idxs = l_out[j_cc][:h]
                        this_j = 2*j_cc
                    elif j_tip == 'end':
                        # filling ending of j values
                        j_idxs = l_out[j_cc][-h:]
                        this_j = 2*j_cc + 1

                    inter_sim = sim[i_idxs, :]
                    inter_sim = inter_sim.T[j_idxs, :].T
                    a[this_i, this_j] = inter_sim.sum()

    for _ in range(n_cc):
        # Find argmax
        (i_max, j_max) = np.unravel_index(a.argmax(), a.shape)
        if a[i_max, j_max] == 0:
            break
        # assert(i_max < j_max)
        if not (i_max < j_max):  # transpose i and j if i > j
            i_max_c = i_max
            i_max = j_max
            j_max = i_max_c
        i_tip = 'begin' if (i_max % 2 == 0) else 'end'
        i_cc = i_max // 2
        j_tip = 'begin' if (j_max % 2 == 0) else 'end'
        j_cc = j_max // 2
        # Merge sequences i and j
        s1 = l_out[i_cc]
        s2 = l_out[j_cc]
        if i_tip == 'begin':
            s1 = s1[::-1]
        if j_tip == 'end':
            s2 = s2[::-1]
        l_out[i_cc] = s1 + s2
        l_out[j_cc] = []
        # Update matrix after merging i and j (i becomes the merged sequence)
        # There are four possible cases (i is first arrow, j is second arrow)
        # --> --> ; --> <-- ; <-- --> ; <-- <--
        a_old = a.copy()
        if i_tip == 'begin':  # begin of new i becomes end of old i
            this_i = 2*i_cc + 1
            this_i_new = 2*i_cc
            a[this_i_new, :] = a_old[this_i, :]
            a[:, this_i_new] = a_old[:, this_i]
        if j_tip == 'begin':  # end of new i becomes end of old j
            this_j = 2*j_cc + 1
            this_i_new = 2*i_cc + 1
            a[this_i_new, :] = a_old[this_j, :]
            a[:, this_i_new] = a_old[:, this_j]
        elif j_tip == 'end':  # end of new i becomes beginning of old j
            this_j = 2*j_cc
            this_i_new = 2*i_cc + 1
            a[this_i_new, :] = a_old[this_j, :]
            a[:, this_i_new] = a_old[:, this_j]
        # erase component j
        for this_j in [2*j_cc, 2*j_cc+1]:
            a[:, this_j] = 0
            a[this_j, :] = 0
        # a = np.triu(a, k=1)

    l_out = [el for el in l_out if len(el) > 1]
    return(l_out)


def distance_set_points(set1, set2, sim, emb, mode='similarity'):
    '''
    compute a measure of similarity between the set of index set1 and set2
    according to the initial similarity matrix or the embedding.
    '''
    if len(set1) != len(set2):
        raise ValueError('the length of set1 and set2 are not the same.')
    if mode == 'similarity':
        if issparse(sim):
            sub_matrix = sim.tolil()[set1, :]
            sub_matrix = sub_matrix.T[set2, :].T
            similar = np.sum(sub_matrix.toarray())
        else:
            similar = np.sum(sim[set1, set2])
        return(similar)
    elif mode == 'embedding':
        # compute all vs all distances
        h = len(set1)
        xs = np.tile(emb[set1, :].T, h).T
        ys = np.repeat(emb[set2, :], h, axis=0)
        all_dists = np.sqrt(np.sum(np.power(xs - ys, 2), axis=1))
        all_dists = np.reshape(all_dists, (h, h))
        all_sim = np.exp(-all_dists)
        sim_mean = all_sim.mean()
        # sum_max = np.sum(np.max(all_sim, axis=0))
        # sim_max = all_sim.max()
        dist = sim_mean

        return(dist)


def find_closest(set, l_c, emb, sim, h=5, mode='similarity', trim_end=False):
    '''
    METHOD:
    given a list of index 'set', it finds the list in l_c which is the closest
    to set.
    OUTPUT: (type_extremity, l, value)
        - type_extremity is either 'similarity' or 'embedding'.
        - l is the chosen element of l_c.
        - value is the similarity of set with l.
    '''
    h = min(2*h, min([len(l_comp) for l_comp in l_c]))  # TODO: might need *1/2
    h //= 2
    if trim_end:
        set = set[:h]
    else:
        set = set[-h:]
    dist_begin = [distance_set_points(set, l_comp[:h], sim, emb, mode=mode) for
                  l_comp in l_c]
    dist_end = [distance_set_points(set, l_comp[-h:], sim, emb, mode=mode) for
                l_comp in l_c]
    print("dists to beginning of sequences : \n {}".format(dist_begin))
    print("dists end of sequences : \n {}".format(dist_end))
    if max(dist_end) < max(dist_begin):
        return('begin', l_c[dist_begin.index(max(dist_begin))],
               max(dist_begin))
    else:
        return('end', l_c[dist_end.index(max(dist_end))], max(dist_end))


def find_contig(end_sigma, begin_sigma,
                l_c, emb, sim, h=5, mode='similarity'):
    '''
    METHOD: a and b represent respectively the type of extremity but which the
    next component should be aggregate to sigma. v_begin or v_end determine
    the extremity of sigma to aggregate.
    INPUT:
        - end_sigma and begin_sigma are the h last and initial elements of
        sigma
        - emb and sim respectively the modified embedding and initial
        similarity matrix.
        - l_c is a list of list of index.
    OUTPUT: (l, type_extremity)
        - type_extremity is either 'begin' or 'end'
        - l is a reordered list that should be append
        at type_extremity of the current list sigma.
    '''
    print("compute distances between beginning of main contig and others.")
    (a, l_c_b, v_begin) = find_closest(begin_sigma, l_c,
                                       emb, sim, h=h, mode=mode,
                                       trim_end=True)
    print("now compute distances between end of main contig and others.")
    (b, l_c_e, v_end) = find_closest(end_sigma, l_c,
                                     emb, sim, h=h, mode=mode,
                                     trim_end=False)

    # Case where no match found (can only happen in similarity mode)
    if (v_begin == 0) and (v_end == 0):
        return(None, None)
    # ADDING AT THE BEGINNING OF SIGMA
    elif v_begin > v_end:
        # remove the list from l_c
        l_c.remove(l_c_b)
        if a == 'end':
            return(l_c_b, 'begin')
        elif a == 'begin':
            return(l_c_b[::-1], 'begin')
    # ADDING AT THE END OF SIGMA
    else:
        # means that we will append at the end of permutation
        # remove the list from l_c
        l_c.remove(l_c_e)
        if b == 'begin':
            return(l_c_e, 'end')
        elif b == 'end':
            return(l_c_e[::-1], 'end')


def merge_conn_comp2(l_connected, sim, emb=None, h=5, mode='similarity',
                    seed_comp_idx=None):
    '''
    Implementation of Algorithm 4.
    INPUT:
        - l_connected is a list of list of index between 0 and n-1.
        - sim is a square symmetric and non-negative matrix.
        - emb is a matrix of size n_element*d_embedding.
        - h a window parameter.
        - mode is either 'embedding' or 'similarity'
    OUTPUT:
        - a list of index representing a permutation of [0,..,n-1]
    '''
    n = sim.shape[0]
    # check that l_connected have no repetition
    # A.R. : remark, if min_cc_len>1, len(set(l_merged)) can be lower than n
    l_merged = sum(l_connected, [])
    sigma_len = len(set(l_merged))
    if sigma_len > n:
        raise ValueError('l_connected has either too many index\
         or repeted index.')
    if len(l_connected) == 1:
        return(l_connected[0])
    # check parameters
    if mode not in ['similarity', 'embedding']:
        raise ValueError('mode should be either similarity or embedding.')
    if mode == 'embedding' and emb is None:
        raise ValueError('In embedding mode, \
                         keyword argument emb must be given')
    # initialize
    l_c = copy.copy(l_connected)
    if seed_comp_idx:
        if seed_comp_idx > len(l_c) - 1:
            seed_comp_idx = 0
        seed_cc = l_c[seed_comp_idx]
        l_c.remove(seed_cc)
        sigma = seed_cc
    else:
        # start with the largest component
        largest_item = l_c[np.argmax([len(l) for l in l_c])]
        l_c.remove(largest_item)
        sigma = largest_item

    while len(sigma) < sigma_len:
        print("aggregated contig of length {}".format(len(sigma)))
        # print('entered here')
        h = min(h, len(sigma)//2)
        begin_sigma = sigma[:h]
        end_sigma = sigma[-h:]
        (l_closest, end_or_begin) = find_contig(end_sigma, begin_sigma, l_c,
                                                emb, sim, h=h,
                                                mode=mode)
        if end_or_begin == 'end':
            sigma = sigma + l_closest
            # perm.extend(l_closest)
        elif end_or_begin == 'begin':
            sigma = l_closest + sigma
        else:  # Added a case where there is no match in beginning or end
            warnings.warn("The connected components could not be merged with\
                           the input similarity matrix ! Returning a sequence\
                           of reordered connected components.")
            l_c.append(sigma)
            return(l_c)

    return(sigma)
