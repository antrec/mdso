#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Evaluate the permutation given a ground truth
'''
import numpy as np
from scipy.stats import kendalltau


def kendall_circular(true_perm, order_perm):
    '''
    TODO : make it faster for large n with a coarser grained slicing first,
    i.e., taking np.roll with a larger value than 1 and then zooming in.
    '''
    n = true_perm.shape[0]
    if (order_perm.shape[0] != n):
        print("wrong length of permutations in kendall_circular!")
    order_perm = true_perm[order_perm]
    id_perm = np.arange(n)
    scores = np.zeros(n)
    for i in range(n):
        scores[i] = abs(kendalltau(id_perm, order_perm)[0])
        order_perm = np.roll(order_perm, 1, axis=0)

    return(np.max(scores), np.argmax(scores))


def evaluate_ordering(perm, true_perm, criterion='kendall',
                      circular=False):
    '''
    evaluate the model.
    INPUT:
        - the ground truth permutation
        - the ordered_chain
    '''
    l1 = len(perm)
    l2 = len(true_perm)
    if not l1 == l2:
        print("Problem : perm of length {}, "
              "and true_perm of length {}".format(l1, l2))
        print("perm : {}".format(perm))
    if criterion == 'kendall':
        if circular:
            (score, _) = kendall_circular(true_perm, perm)
        else:
            score = abs(kendalltau(true_perm, np.argsort(perm))[0])
        return(score)
