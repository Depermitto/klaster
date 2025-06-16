import argparse
import json
from time import perf_counter

import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import (
    make_blobs,
    load_breast_cancer,
    load_wine,
    fetch_openml,
    fetch_20newsgroups,
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from hdbscan import HDBSCAN


def benchmark_python(X, y, algorithm, runs, **params):
    all_results = []
    for _ in range(runs):
        start_time = perf_counter()
        if algorithm == "kmeans":
            cluster_labels = KMeans(**params).fit_predict(X)
        elif algorithm == "hdbscan":
            cluster_labels = HDBSCAN(**params).fit_predict(X)
        end_time = perf_counter()

        all_results.append(
            [
                end_time - start_time,
                benefit_of_doubt_acc(y, cluster_labels),
                metrics.adjusted_rand_score(y, cluster_labels),
                metrics.adjusted_mutual_info_score(y, cluster_labels),
            ]
        )

    mean_results = np.mean(all_results, axis=0)
    return {
        "time": mean_results[0],
        "accuracy": mean_results[1],
        "ARI": mean_results[2],
        "AMI": mean_results[3],
    }


def benefit_of_doubt_acc(y_true, y_pred):
    from scipy.stats import mode

    most_common_label = mode(y_true)[0]
    label_mapping = {}
    for cluster in np.unique(y_pred):
        if cluster == -1:  # if treated as noise by density-based algorithms
            label_mapping[cluster] = most_common_label
        else:
            true_label = mode(y_true[y_pred == cluster])[0]
            label_mapping[cluster] = true_label

    aligned_labels = np.array([label_mapping.get(pred) for pred in y_pred])
    return metrics.accuracy_score(y_true, aligned_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, choices=["kmeans", "hdbscan"], required=True)
    parser.add_argument(
        "--min_cluster_size", type=int, default=10, help="min_cluster_size for HDBSCAN"
    )
    parser.add_argument(
        "--min_samples", type=int, default=5, help="min_samples for HDBSCAN"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["synth", "bcw", "wine", "mnist", "20-newsgroups"],
        required=True,
        help="Dataset to use",
    )
    parser.add_argument(
        "--no-scaling",
        action="store_true",
        default=False,
        help="Do not standardize the dataset by removing mean and scaling to unit variance before clustering",
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
        n_clusters = 3
    elif args.dataset == "mnist":
        X, y = fetch_openml("mnist_784", return_X_y=True)
        X = X[:1000]
        y = y[:1000].astype(int)
        n_clusters = 10
    elif args.dataset == "20-newsgroups":
        X, y = fetch_20newsgroups(
            remove=("headers", "footers", "quotes"), return_X_y=True
        )
        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        X = vectorizer.fit_transform(X)
        n_clusters = 20

    if not args.no_scaling:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    if args.alg == "kmeans":
        params = {"n_clusters": n_clusters}
    else:
        params = {
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
        }

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        result = benchmark_python(
            X=X, y=y, algorithm=args.alg, runs=args.runs, **params
        )
    print(json.dumps(result, indent=4))
