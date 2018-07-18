#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment. We will vary:
    - type of noise
    - type of matrix
    - support of noise
    - amplitude of noise
    - type Laplacian
    - size of embedding
    - parameter k of local neighborhood
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from mdso import SimilarityMatrix, SpectralOrdering, evaluate_ordering


def run_one_exp(n, k, dim, ampl, type_matrix, n_avrg,
                type_noise='gaussian', norm_laplacian='unnormalized',
                norm_adjacency=False, scale_embedding='heuristic'):
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
                                    merge_if_ccs=True)

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


def listify(*args):
    """
    Given a set of parameters where some of them are given as lists and the
    others as values, put brackets around the values to make lists of 1 element
    so that we can use a for loop on all parameters in  run_synthetic_exps.
    For instance, the tuple of parameters
    (n=500, k=[10, 15, 20], dim=10, ampl=[0, 1, 2])
    is changed into
    (n=[500], k=[10, 15, 20], dim=[10], ampl = [0, 1, 2]) by this function.
    """
    listed_args = ()
    for arg in args:
        if type(arg) == list:
            listed_args += (arg, )
        else:
            listed_args += ([arg], )
    return(listed_args)


def fetch_res(n, k, dim, ampl_noise, type_mat, scaled, norm_laplacian, n_avrg,
              save_res_dir):
    """
    Get the results from a given experiment when they were saved to a file,
    in order to use them for plotting in plot_from_res, or not to redo the same
    computation twice in run_synthetic_exps.
    """
    fn = "n_{}-k_{}-dim_{}-ampl_{}-type_mat_{}" \
         "-scaled_{}-norm_laplacian_{}-n_avrg_{}.res".format(
             n, k, dim, ampl_noise, type_mat, scaled, norm_laplacian, n_avrg)
    fn = save_res_dir + "/" + fn
    with open(fn, 'r') as f:
        first_line = f.readlines()[0]

    res_mean, res_std = [float(el) for el in first_line.split()]
    return(res_mean, res_std)


def fetch_res_full(n, k, dim, ampl_noise, type_mat,
                   scaled, norm_laplacian, n_avrg,
                   save_res_dir):
    """
    Get the results from a given experiment when they were saved to a file,
    in order to use them for plotting in plot_from_res, or not to redo the same
    computation twice in run_synthetic_exps.
    """
    fn = "n_{}-k_{}-dim_{}-ampl_{}-type_mat_{}" \
         "-scaled_{}-norm_laplacian_{}-n_avrg_{}.res".format(
             n, k, dim, ampl_noise, type_mat, scaled, norm_laplacian, n_avrg)
    fn = save_res_dir + "/" + fn
    with open(fn, 'r') as f:
        full = " ".join(line.rstrip('\n') for line in f)
    full = str(full)
    re_read = re.compile('\[.*\]')
    new_full = re_read.search(full).group()
    new_full = new_full.replace('[', '')
    new_full = new_full.replace(']', '')
    res = [float(el) for el in new_full.split()]
    return(res)


def run_synthetic_exps(n_l, k_l, dim_l, ampl_l, type_matrix_l, scaled_l,
                       norm_laplacian_l=None, n_avrg=20, save_res_dir=None):
    """
    Run synthetic experiments. Parameters  ending with '_l' can be given in the
    form of a list. Then, experiments will be ran for each values in this list,
    all other parameters fixed.

    Parameters
    ----------
    n_l : int or list of ints
        size of matrix

    k_l : int or list of ints
        (list of) values used for the number of nearest neighbours to fit by
        a local line in the spectral embedding

    dim_l :  int or list of ints
        (list of) number of dimensions of the spectral embedding

    ampl_l : float or list of floats
        (list of) level of noise

    type_matrix_l : string or list of strings
        type of synthetic matrix, among ['LinearBanded', 'CircularBanded',
        'LinearStrongDecrease', 'CircularStrongDecrease'].

    scaled_l : boolean or string or list of booleans or strings
        whether or not to scale the dimensions of the embedding.
        if scaled = False, do not use any scaling.
        if scaled = 'CTD', scale dimension k by 1/sqrt(lambda_k) (associated
        eigenvalue of the laplacian)
        otherwise, use the default heuristic scaling : 1/sqrt(k)

    norm_laplacian_l : None or list of strings
        if norm_laplacian_l = None, use the default laplacian type,
        'random_walk'. otherwise, use the specified norm_laplacianlacian, that
        takes values among ['unnormalized', 'random_walk', 'symmetric']

    n_avrg : int, default 20
        number of experiments to run with a given combination of parameters,
        for the purpose of averaging and displaying standard deviations.
    """

    # Create save_res_dir if it does not exist
    if save_res_dir:
        if not os.path.exists(save_res_dir):
            os.mkdir(save_res_dir)

    # Make arguments be lists even for singleton arguments
    (n_l, k_l, dim_l, ampl_l, type_matrix_l, scaled_l,
     norm_laplacian_l) = listify(n_l, k_l, dim_l, ampl_l, type_matrix_l,
                                 scaled_l, norm_laplacian_l)

    # Run experiments
    for n in n_l:
        for k in k_l:
            for dim in dim_l:
                for ampl in ampl_l:
                    for type_matrix in type_matrix_l:
                        for scaled in scaled_l:
                            for norm_laplacian in norm_laplacian_l:
                                print("n:{}, k:{}, dim:{}, ampl:{}, "
                                      "type_matrix:{}, scaled:{}, "
                                      "norm_laplacian:{}, "
                                      "n_avrg:{}".format(
                                          n, k, dim, ampl, type_matrix, scaled,
                                          norm_laplacian, n_avrg))
                                # Check if result file already exists
                                if save_res_dir:
                                    fn = "n_{}-k_{}-dim_{}-ampl_{}-"\
                                         "type_mat_{}-scaled_{}-norm_laplacia"\
                                         "n_{}-n_avrg_{}.res".format(
                                             n, k, dim, ampl, type_matrix,
                                             scaled, norm_laplacian, n_avrg)
                                    fn = save_res_dir + "/" + fn
                                    if os.path.isfile(fn):
                                        (mn, stdv) = fetch_res(
                                            n, k, dim, ampl, type_matrix,
                                            scaled, norm_laplacian, n_avrg,
                                            save_res_dir)
                                        # Print results
                                        print("MEAN_SCORE:{}, STD_SCORE:{}"
                                              "".format(mn, stdv))
                                        continue

                                (mn, stdv, scores) = run_one_exp(
                                    n, k, dim, ampl, type_matrix, n_avrg,
                                    scale_embedding=scaled,
                                    norm_laplacian=norm_laplacian)
                                # Print results
                                print("MEAN_SCORE:{}, STD_SCORE:{}"
                                      "".format(mn, stdv))
                                if save_res_dir:
                                    fh = open(fn, 'a')
                                    print(mn, stdv, file=fh)
                                    print(scores, file=fh)
                                    fh.close()
    return


def check_args_for_plot(*args):
    """
    In plot_from_res, we want to make a plot with the amplitude of noise in the
    x_axis, the score in y-axis, and allow another parameter to vary, in order
    to superpose the plots for the different values of this parameter.
    Therefore, we need ampl_l to be a list, and another parameter to be given
    as a list.
    This function checks that only two parameters among all are given as lists,
    and that ampl_l is one of them.

    args : (n_l, k_l, dim_l, ampl_l, type_matrix_l, scaled_l, norm_laplacian_l)
    """

    is_arg_list = np.zeros(len(args))
    for (ix, arg) in enumerate(args):
        if list == type(arg):
            is_arg_list[ix] = 1
    listed_args = list(np.argwhere(is_arg_list)[:, 0])

    if 3 not in listed_args:
        raise TypeError("ampl_l must be a list of noise amplitudes"
                        "to make a plot.")
    listed_args.remove(3)
    if len(listed_args) > 1:
        raise TypeError("Only one argument (apart from ampl_l)"
                        "can be a list to make one plot.")
    l_arg_ix = listed_args[0]
    listed_arg = args[l_arg_ix]

    list_of_args = []
    for i in range(len(listed_arg)):
        these_args = args[:l_arg_ix] + (listed_arg[i],) + args[l_arg_ix+1:]
        list_of_args.append(these_args)

    return(list_of_args, listed_arg, l_arg_ix)


def add_plot_exp(n, k, dim, ampl_l, type_matrix, scaled, norm_laplacian,
                 n_avrg, save_res_dir, col='k', marker='d'):
    """
    Function to add one plot for a given value of the varying parameter, in the
    figure created in plot_from_res.
    """

    n_ampl = len(ampl_l)
    x = np.array(ampl_l)
    means = np.zeros(n_ampl)
    stds = np.zeros(n_ampl)
    for (ix, ampl_noise) in enumerate(ampl_l):
        (mn, stdv) = fetch_res(n, k, dim, ampl_noise, type_matrix, scaled,
                               norm_laplacian, n_avrg, save_res_dir)
        means[ix] = mn
        stds[ix] = stdv

    plt.plot(x, means, marker=marker, lw=2, color=col, linestyle='dashed')
    plt.fill_between(x, means + 1./np.sqrt(n_avrg)*stds,
                     means-1./np.sqrt(n_avrg)*stds, facecolor=col, alpha=0.25)
    # plt.fill_between(x, means + stds,
    #                  means - stds, facecolor=col, alpha=0.25)

    return


def add_plot_exp_full(n, k, dim, ampl_l, type_matrix, scaled, norm_laplacian,
                      n_avrg, save_res_dir, col='k', marker='d'):
    """
    Function to add one plot for a given value of the varying parameter, in the
    figure created in plot_from_res.
    """

    n_ampl = len(ampl_l)
    x = np.array(ampl_l)
    gain_means = np.zeros(n_ampl)
    gain_lowers = np.zeros(n_ampl)
    gain_uppers = np.zeros(n_ampl)
    for (ix, ampl_noise) in enumerate(ampl_l):
        l_values = fetch_res_full(n, k, dim, ampl_noise, type_matrix, scaled,
                                  norm_laplacian, n_avrg, save_res_dir)
        l_values_baseline = fetch_res_full(n, k, 1, ampl_noise, type_matrix,
                                           scaled, norm_laplacian,
                                           n_avrg, save_res_dir)
        l_values = np.array(l_values)
        l_values_baseline = np.array(l_values_baseline)
        if len(l_values_baseline) != n_avrg or len(l_values) != n_avrg:
            raise ValueError('these two list should be of same size')
        gain = l_values - l_values_baseline
        sorted_gain = np.sort(gain)
        gain_means[ix] = np.mean(gain)
        gain_lowers[ix] = sorted_gain[int(0.3*len(sorted_gain))]
        gain_uppers[ix] = sorted_gain[int(1*len(sorted_gain))-1]
        '''
        Beware that because the gain is sometimes negative and hard to
        represent with negative values and logscale at the same time,
        the interval confidence may be considered a bit unbiaised.
        '''

    plt.semilogy(x, gain_means, marker=marker,
                 lw=2, color=col, linestyle='dashed')
    plt.fill_between(x, gain_uppers, gain_lowers,
                     facecolor=col, alpha=0.25)

    return


def plot_from_res(n_l, k_l, dim_l, ampl_l, type_matrix_l, scaled_l,
                  norm_laplacian_l=None, n_avrg=20, save_res_dir=None,
                  save_fig_path=None):
    """
    Plot the kendall-tau score vs the noise amplitude for several values of one
    varying parameter. The varying parameter can be chosen amongst
    n, k, dim, type_matrix, scaled, norm_laplacian,
    and it must be given as a list of values.
    ampl_l must also be given as a list of values, and all other parameters
    must be given as values (not lists, even of length 1).
    If the user wishes to have only one plot on the figure (no varying
    parameter), then one of the parameters other than ampl_l has to be given as
    a list of length 1.
    """
    (list_of_args,
     the_l_arg, l_arg_ix) = check_args_for_plot(n_l, k_l, dim_l, ampl_l,
                                                type_matrix_l, scaled_l,
                                                norm_laplacian_l)

    arg_names = ['n', 'k_nbrs', 'dim', 'noise_ampl', 'type_mat', 'scaled',
                 'norm_laplacian']
    listed_arg_name = arg_names[l_arg_ix]

    l_legend = []
    color_l = ['k', 'b', 'r', 'g', 'm', 'c', 'y', 'w']  # sorted in gray level
    # color_l = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    markers_l = ['d', 'o', 's', 'p', 'v', '^', 'D', '>']
    plt.subplots(1)
    for (ix, these_args) in enumerate(list_of_args):
        (n, k, dim, ampl_l, type_matrix, scaled, norm_laplacian) = these_args
        add_plot_exp(n, k, dim, ampl_l, type_matrix, scaled, norm_laplacian,
                     n_avrg, save_res_dir, col=color_l[ix],
                     marker=markers_l[ix])
        this_arg_val = the_l_arg[ix]
        if listed_arg_name == 'dim' and this_arg_val == 1:
            l_legend.append("baseline")
        else:
            l_legend.append("{} = {}".format(listed_arg_name, this_arg_val))
    plt.xlabel('noise level', fontsize=16)
    plt.ylabel('Kendall-tau', fontsize=16)
    plt.legend(l_legend)
    if not save_fig_path:
        plt.show()
    else:
        # plt.show()
        plt.savefig(save_fig_path, bbox_inches='tight')
    return


def plot_from_res_gain(n_l, k_l, dim_l, ampl_l, type_matrix_l, scaled_l,
                       norm_laplacian_l=None, n_avrg=20, save_res_dir=None,
                       save_fig_path=None):
    """
    Plot the gain (difference between Kendall-tau) vs the noise amplitude for
    several values of one varying parameter. The varying parameter can be
    chosen amongst n, k, dim, type_matrix, scaled, norm_laplacian, and it must
    be given as a list of values.
    ampl_l must also be given as a list of values, and all other
    parameters  must be given as values (not lists, even of length 1).
    If the user wishes to have only one plot on the figure (no varying
    parameter), then one of the parameters other than ampl_l has to be given as
    a list of length 1.
    """
    (list_of_args,
     the_l_arg, l_arg_ix) = check_args_for_plot(n_l, k_l, dim_l, ampl_l,
                                                type_matrix_l, scaled_l,
                                                norm_laplacian_l)

    arg_names = ['n', 'k_nbrs', 'dim', 'noise_ampl', 'type_mat', 'scaled',
                 'norm_laplacian']
    listed_arg_name = arg_names[l_arg_ix]

    l_legend = []
    color_l = ['k', 'b', 'r', 'g', 'm', 'c', 'y', 'w']
    # color_l = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    markers_l = ['d', 'o', 's', 'p', 'v', '^', 'D', '>']
    plt.subplots(1)
    '''
    dimension 1 might be a problem
    '''
    for (ix, these_args) in enumerate(list_of_args):
        (n, k, dim, ampl_l, type_matrix, scaled, norm_laplacian) = these_args
        add_plot_exp_full(n, k, dim, ampl_l, type_matrix, scaled,
                          norm_laplacian, n_avrg, save_res_dir,
                          col=color_l[ix], marker=markers_l[ix])
        this_arg_val = the_l_arg[ix]
        if listed_arg_name == 'dim' and this_arg_val == 1:
            l_legend.append("baseline")
        else:
            l_legend.append("{} = {}".format(listed_arg_name, this_arg_val))
    plt.xlabel('noise level', fontsize=16)
    plt.ylabel('gain in Kendall-tau', fontsize=16)
    plt.legend(l_legend)
    if not save_fig_path:
        plt.show()
    else:
        # plt.show()
        plt.savefig(save_fig_path, bbox_inches='tight')

    return


if __name__ == '__main__':

    """
    """
    # Plots from the main paper (Kendall-Tau scores for several noise amplitude
    # and dimensions).
    n = 500
    k = 15
    dim_l = [1, 3, 5, 7, 10, 15, 20]
    ampl_l = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    n_avrg = 100
    type_matrix_l = ['CircularStrongDecrease', 'LinearStrongDecrease',
                     'CircularBanded', 'LinearBanded']
    scaled = 'heuristic'

    exps_dir = os.path.dirname(os.path.abspath(__file__))
    save_res_dir = exps_dir + '/results'
    if not os.path.exists(save_res_dir):
        os.mkdir(save_res_dir)
    # Run experiments
    run_synthetic_exps(n, k, dim_l, ampl_l, type_matrix_l, scaled,
                       n_avrg=n_avrg, save_res_dir=save_res_dir,
                       norm_laplacian_l='random_walk')

    # Make the Figures from the paper
    for type_matrix in type_matrix_l:
        fig_name = "kendall-tau-vs-noise-for-several-dims-typematrix_{}.pdf"\
                   "".format(type_matrix)
        fig_name = save_res_dir + '/' + fig_name

        plot_from_res(n, k, dim_l, ampl_l, type_matrix, scaled,
                      norm_laplacian_l='random_walk', n_avrg=n_avrg,
                      save_res_dir=save_res_dir,
                      save_fig_path=fig_name)

    # Supplementary plots : sensitivity to k_nbrs parameter.
    k_l = [5, 10, 15, 20, 30, 50]
    dim = 10
    scaled = 'heuristic'
    # Run experiments
    run_synthetic_exps(n, k_l, dim, ampl_l, type_matrix_l, scaled,
                       n_avrg=n_avrg, save_res_dir=save_res_dir,
                       norm_laplacian_l='random_walk')
    # Make the Figures from the paper
    for type_matrix in type_matrix_l:
        fig_name = "kendall-tau-vs-noise-for-several-k_nns-typematrix_{}.pdf" \
                   "".format(type_matrix)
        fig_name = save_res_dir + '/' + fig_name

        plot_from_res(n, k_l, dim, ampl_l, type_matrix, scaled,
                      norm_laplacian_l='random_walk', n_avrg=n_avrg,
                      save_res_dir=save_res_dir,
                      save_fig_path=fig_name)

    # Supplementary plots : sensitivity to the scaling of the embedding.
    scaled_l = ['heuristic', 'CTD', False]
    k = 15
    dim = 20
    # Run experiments
    run_synthetic_exps(n, k, dim, ampl_l, type_matrix_l, scaled_l,
                       n_avrg=n_avrg, save_res_dir=save_res_dir,
                       norm_laplacian_l='random_walk')
    # Make the Figures from the paper
    for type_matrix in type_matrix_l:
        fig_name = "kendall-tau-vs-noise-for-several-scalings-typematrix"\
                   "_{}.pdf".format(type_matrix)
        fig_name = save_res_dir + '/' + fig_name

        plot_from_res(n, k, dim, ampl_l, type_matrix, scaled_l,
                      norm_laplacian_l='random_walk', n_avrg=n_avrg,
                      save_res_dir=save_res_dir,
                      save_fig_path=fig_name)
