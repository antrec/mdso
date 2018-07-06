#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to run a batch of n_avrg experiments with a fixed combination of
parameters (n, k, dim, ampl_noise, type_mat, scaled, norm_laplacian), useful if you
wish to run experiments on a cluster, with a loop on the values of (n, k, dim,
ampl_noise, type_mat, scaled, norm_laplacian) in a shell script (e.g.,
run_exps_cluster.sh) with my SGE cluster.
"""
import os
import argparse
import pathlib
from run_experiments import run_one_exp, fetch_res


# Define argument parser
parser = argparse.ArgumentParser(description="run some experiments"
                                 "with combination of parameters dim, "
                                 "amplitude, k, norm_laplacian, "
                                 "type_matrix")

parser.add_argument("-r", "--root_dir",
                    help="directory where to store result files.",
                    type=str,
                    default="./")
parser.add_argument("-i", "--type_laplacian_initial",
                    help="Laplacian for embedding. 'random_walk' or "
                    "'unnormalized'.",
                    type=str,
                    default='random_walk')
parser.add_argument("-m", "--type_matrix",
                    help="Type of similarity matrix. 'LinearBanded', "
                    "'LinearStrongDecrease', 'CircularBanded' or "
                    "'CircularStrongDecrease'.",
                    type=str,
                    default='LinearBanded')
parser.add_argument("-k", "--k_nbrs",
                    help="number of nearest-neighbors used in to approximate "
                    "a local line",
                    type=int,
                    default=15)
parser.add_argument("-d", "--dim", help="dimension of the Laplacian embedding",
                    type=int,
                    default=3)
parser.add_argument("-a", "--amplitude_noise",
                    help="amplitude of the noise on the matrix.",
                    type=float,
                    default=0.5)
parser.add_argument("-n", "--n",
                    help="number of elements (size of similarity matrix)",
                    type=int,
                    default=500)
parser.add_argument("-s", "--scale_embedding",
                    help="If scaled == 0, do not apply any scaling."
                    "If scaled == 1, apply CTD scaling to embedding, "
                    "(y_k /= sqrt(lambda_k))."
                    "If scaled == 2, apply default scaling (y_k /= sqrt(k)).",
                    type=int,
                    default=2)
parser.add_argument("-e", "--n_exps",
                    help="number of experiments performed to average results",
                    type=int,
                    default=100)
parser.add_argument("--type_noise",
                    help="type of noise ('uniform' or 'gaussian')",
                    type=str,
                    default='gaussian')

# Get arguments
args = parser.parse_args()

n = args.n
k = args.k_nbrs
dim = args.dim
type_matrix = args.type_matrix
norm_laplacian = args.type_laplacian_initial
scale_code = args.scale_embedding
if scale_code == 0:
    scaled = False
elif scale_code == 1:
    scaled = 'CTD'
else:
    scaled = 'heuristic'
type_noise = args.type_noise
ampl = args.amplitude_noise
n_avrg = args.n_exps
save_res_dir = args.root_dir

# Create directory for results if it does not already exist
if save_res_dir:
    pathlib.Path(save_res_dir).mkdir(parents=True, exist_ok=True)

print("n:{}, k:{}, dim:{}, ampl:{}, "
      "type_matrix:{}, scaled:{}, "
      "norm_laplacian:{}, "
      "n_avrg:{}".format(n, k, dim, ampl,
                         type_matrix,
                         scaled,
                         norm_laplacian, n_avrg))

if save_res_dir:
    # Check if the file already exists and read results if so
    fn = "n_{}-k_{}-dim_{}-ampl_{}" \
         "-type_mat_{}" \
         "-scaled_{}-norm_laplacian_{}-n_avrg_{}." \
         "res".format(n, k, dim, ampl,
                      type_matrix, scaled,
                      norm_laplacian, n_avrg)
    fn = save_res_dir + "/" + fn
    if os.path.isfile(fn):
        (mn, stdv) = fetch_res(n, k, dim, ampl,
                               type_matrix,
                               scaled,
                               norm_laplacian,
                               n_avrg,
                               save_res_dir)
    else:
        # Run the experiments if the result file does not already exist
        (mn, stdv, scores) = run_one_exp(n, k, dim, ampl, type_matrix, n_avrg,
                                         norm_laplacian=norm_laplacian,
                                         scale_embedding=scaled)
    # Print results
    print("MEAN_SCORE:{}, STD_SCORE:{}"
          "".format(mn, stdv))
    fh = open(fn, 'a')
    print(mn, stdv, file=fh)
    print(scores, file=fh)
    fh.close()
