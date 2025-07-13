# Copyright (C) 2025 Piotr Jabłoński
# Extended copyright information can be found in the LICENSE file.

import argparse
import json

import numpy as np
from helper import benchmark_python
from sklearn.datasets import (
    fetch_20newsgroups,
    fetch_openml,
    load_breast_cancer,
    load_wine,
    make_blobs,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg", type=str, choices=["kmeans", "hdbscan", "n2d", "nddc"], required=True
    )
    parser.add_argument(
        "--hdbscan-min-cluster-size",
        type=int,
        default=10,
        help="min_cluster_size for HDBSCAN",
    )
    parser.add_argument(
        "--hdbscan-min-samples", type=int, default=5, help="min_samples for HDBSCAN"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of epochs for autoencoder training",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="500,500,2000",
        help="Comma-separated layer sizes for autoencoder architecture (e.g. '500,500,2000')",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output for training",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["synth", "bcw", "wine", "mnist", "20-newsgroups"],
        required=True,
        help="Dataset to use",
    )
    parser.add_argument(
        "--scaled",
        action="store_true",
        default=False,
        help="Standardize the dataset by removing mean and scaling to unit variance before clustering",
    )
    parser.add_argument(
        "--blobs-samples",
        type=int,
        default=300,
        help="Number of samples for synthetic blobs dataset",
    )
    parser.add_argument(
        "--blobs-centers",
        type=int,
        default=3,
        help="Number of centers for synthetic blobs dataset",
    )
    parser.add_argument(
        "--blobs-features",
        type=int,
        default=2,
        help="Number of features for synthetic blobs dataset",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=50,
        help="Number of runs to average results over",
    )
    args = parser.parse_args()

    if args.dataset == "synth":
        X, y = make_blobs(
            n_samples=args.blobs_samples,
            centers=args.blobs_centers,
            n_features=args.blobs_features,
        )
        n_clusters = len(np.unique(y))
    elif args.dataset == "bcw":
        X, y = load_breast_cancer(return_X_y=True)
        n_clusters = 2
    elif args.dataset == "wine":
        X, y = load_wine(return_X_y=True)
        print(np.unique(y))
        n_clusters = 3
    elif args.dataset == "mnist":
        X, y = fetch_openml("mnist_784", return_X_y=True, as_frame=False)
        X = X[:1000]
        y = y[:1000].astype(int)
        n_clusters = 10
    elif args.dataset == "20-newsgroups":
        from sklearn.feature_extraction.text import TfidfVectorizer

        X, y = fetch_20newsgroups(
            remove=("headers", "footers", "quotes"), return_X_y=True
        )
        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        X = vectorizer.fit_transform(X)
        n_clusters = 20

    if args.alg == "kmeans":
        params = {"n_clusters": n_clusters}
    elif args.alg == "hdbscan":
        params = {
            "min_cluster_size": args.hdbscan_min_cluster_size,
            "min_samples": args.hdbscan_min_samples,
        }
    elif args.alg == "n2d":
        params = {
            "n_clusters": n_clusters,  # UmapGMM
            "epochs": args.epochs,
            "arch": [int(x) for x in args.arch.split(",")],
            "verbose": args.verbose,
        }
    elif args.alg == "nddc":
        params = {
            "n_clusters": n_clusters,
            "epochs": args.epochs,
            "arch": [int(x) for x in args.arch.split(",")],
        }

    if args.scaled:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        params["dataset"] = args.dataset + "-scaled"
    else:
        params["dataset"] = args.dataset

    result = benchmark_python(X=X, y=y, algorithm=args.alg, runs=args.runs, **params)
    print(json.dumps(result, indent=4))
