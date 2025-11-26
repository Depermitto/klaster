# Copyright (C) 2025 Piotr Jabłoński
# Extended copyright information can be found in the LICENSE file.

import json
import os
from datetime import datetime
from time import perf_counter

import numpy as np
from sklearn import metrics

from tqdm import tqdm


def benchmark_python(X, y, algorithm, runs, **params):
    timestamp = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
    all_results = []
    for _ in tqdm(range(runs)):
        start_time = perf_counter()
        if algorithm == "kmeans":
            from sklearn.cluster import KMeans

            cluster_labels = KMeans(n_clusters=params["n_clusters"]).fit_predict(X)
        elif algorithm == "hdbscan":
            from hdbscan import HDBSCAN

            cluster_labels = HDBSCAN(
                min_cluster_size=params["min_cluster_size"],
                min_samples=params["min_samples"],
            ).fit_predict(X)
        elif algorithm == "n2d":
            import n2d

            n_clusters = params["n_clusters"]
            ae = n2d.AutoEncoder(
                input_dim=X.shape[1],
                latent_dim=n_clusters,
                architecture=params["arch"],
            )
            manifoldGMM = n2d.UmapGMM(n_clusters)

            clusterer = n2d.n2d(ae, manifoldGMM)
            cluster_labels = clusterer.fit_predict(
                X,
                epochs=params["epochs"],
                verbose=params["verbose"],
                weight_id=f"weights/{timestamp}-ae.weights.h5",
            )
        elif algorithm == "nddc":
            from nddc import NDDC
            import torch

            X_tensor = torch.from_numpy(X).float()
            n_clusters = params["n_clusters"]

            clusterer = NDDC(
                input_dim=X_tensor.shape[1],
                latent_dim=n_clusters,
                architecture=params["arch"],
                n_clusters=n_clusters,
            )

            clusterer.fit(
                X_tensor,
                epochs=params["epochs"],
                model_id=f"weights/{timestamp}-nddc.pt",
                batch_size=32,
                reconstruction_weight=0.5,
                clustering_weight=0.5,
                constrast_weight=5,
            )
            cluster_labels = clusterer.predict(X_tensor)

        end_time = perf_counter()

        all_results.append(
            [
                end_time - start_time,
                benefit_of_doubt_acc(y, cluster_labels),
                metrics.adjusted_rand_score(y, cluster_labels),
                metrics.mutual_info_score(y, cluster_labels),
            ]
        )

    mean_results = np.mean(all_results, axis=0)
    dict_results = {
        "runs": runs,
        "time": mean_results[0],
        "accuracy": mean_results[1],
        "ARI": mean_results[2],
        "NMI": mean_results[3],
    }

    os.makedirs("output", exist_ok=True)
    filename = f"{timestamp}-{algorithm}-{params['dataset']}.json"
    filepath = os.path.join("output", filename)
    with open(filepath, "w") as f:
        json.dump(dict_results, f, indent=4)

    return dict_results


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

    aligned_labels = np.array(
        [label_mapping.get(pred, most_common_label) for pred in y_pred]
    )
    return metrics.accuracy_score(y_true, aligned_labels)
