from time import perf_counter

import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans

from hdbscan import HDBSCAN
import n2d


def benchmark_python(X, y, algorithm, runs, **params):
    all_results = []
    for _ in range(runs):
        start_time = perf_counter()
        if algorithm == "kmeans":
            cluster_labels = KMeans(**params).fit_predict(X)
        elif algorithm == "hdbscan":
            cluster_labels = HDBSCAN(**params).fit_predict(X)
        else:
            n_clusters = params["n_clusters"]
            ae = n2d.AutoEncoder(
                X.shape[1],
                latent_dim=n_clusters,
                architecture=[200, 200, 500],
            )
            manifoldGMM = n2d.UmapGMM(n_clusters)

            clusterer = n2d.n2d(ae, manifoldGMM)
            cluster_labels = clusterer.fit_predict(
                X, weight_id=f"weights/{params['dataset']}-ae_weights.h5"
            )

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
