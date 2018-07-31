# mdso (Multi-dimensional Spectral Ordering)
Created by Antoine Recanati at INRIA, Paris, with Thomas Kerdreux.
Code from our paper Reconstructing Latent Orderings by Spectral Clustering (https://arxiv.org/pdf/1807.07122.pdf).

## Introduction
This package implements the Spectral Ordering method which relies on a Laplacian embedding from a pairwise similarity matrix to reorder a set of points sequentially.
It also handles circular orderings.
Our method uses several dimensions of the Laplacian embedding to read the ordering, instead of only 1 for a linear ordering or 2 for a circular ordering in the usual spectral method (see details in [our paper](https://arxiv.org/pdf/1807.07122.pdf)).
Specifically, we compute a multi-dimensional Laplacian embedding, and use it to derive a new similarity matrix that leverages the global linear (or circular) structure of the data.
We then apply the usual spectral method on this new similarity matrix, resulting in improved robustness to noise in the raw similarity.

## Installation with pip
The code is written in python3.7.
The package can be installed from source with the following lines
```sh
git clone https://github.com/antrec/mdso.git
cd mdso
pip install .
```

## Methods
The package provides the following functions
- `spectral_embedding`, computes the laplacian embedding from a similarity matrix. Borrowed from scikit-learn, with a few minor changes.
- `SpectralBaseline` implements the basic Spectral Ordering algorithm from Atkins (for linear orderings) or Coifman (for circular orderings)
- `SpectralOrdering` is our improved version which computes the spectral embedding, use it to derive a new similarity, and then calls SpectralBaseline method on the new similarity.

## Examples
* Running `/examples/synthetic_example.py` shows a simple example, where we generate a synthetic similarity matrix with `SimilarityMatrix` and call the method `SpectralOrdering.fit` on this data matrix. The parameters (type of the matrix, number of dimensions, type of normalization of the Laplacian, etc. can be easily changed in the script.)

* Running `examples/ecoli_example.py` reproduces an experiment where we compute the layout of Oxford Nanopore reads of a bacterial genome (*E. coli*) released by the Loman lab (see their [page](http://lab.loman.net/2015/09/24/first-sqk-map-006-experiment/)). The script should download the dataset, together with the [minimap2](https://github.com/lh3/minimap2) software that enables computing the overlaps from the DNA reads (if it is not already on your path). The overlaps are then computed to construct the similarity matrix, on which our method is called to reorder the reads by increasing position on the genome. We provide visual plots where we use a reference genome (also downloaded by the script automatically) to get a proxy of the true position of the reads on the genome, obtained by alignment also with [minimap2](https://github.com/lh3/minimap2).
This experiment requires the [biopython][biopython] package, which should be installed automatically when running the script.
We recommend installing the `amg` package, with `conda install pyamg` or `pip install pyamg`, to allow for a substantial speed up in the laplacan embedding computation (from a few minutes to a few seconds for this experiment). However, the code will still run without it (the arpack solver will be used then).

## Reproducing the experiments from the paper
The script `exps/run_experiments.py` reproduces the experiments from our paper. However, it may take a long time on a regular machine (several hours), since we let many parameters vary and average the results over 100 experiments per combination of parameters.
The scripts `exps/run_exps_cluster.sh` and `exps/parse_args_run_one_exp.py` are a way to run experiments in parallel on a cluster, but you have to adapt it to your own cluster configuration.


[minimap2]: https://github.com/lh3/minimap2
[biopython]: http://biopython.org/wiki/Download#Easy_Install
