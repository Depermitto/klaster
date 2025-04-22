import numpy as np
from time import time

from sklearn import metrics
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def accuracy_score(y_true, y_pred):
    from scipy.stats import mode

    most_common_label = mode(y_true)[0]
    contingency_matrix = metrics.confusion_matrix(y_true, y_pred)
    label_mapping = {}
    for cluster in np.unique(y_pred):
        if cluster == -1:  # if treated as noise by density-based algorithms
            label_mapping[cluster] = most_common_label
        else:
            true_label = mode(y_true[y_pred == cluster])[0]
            label_mapping[cluster] = true_label

    aligned_labels = np.array([label_mapping.get(pred) for pred in y_pred])
    return metrics.accuracy_score(y_true, aligned_labels)


def bench_estimator(estimator, name, data, labels, times=50):
    all_results = []
    for _ in range(times):
        t0 = time()
        est = make_pipeline(StandardScaler(), estimator).fit(data)
        results = [time() - t0, accuracy_score(labels, est[-1].labels_)]
        results += [
            m(labels, est[-1].labels_)
            for m in [
                metrics.homogeneity_score,
                metrics.v_measure_score,
                metrics.adjusted_rand_score,
                metrics.adjusted_mutual_info_score,
            ]
        ]
        results.append(
            metrics.silhouette_score(
                data,
                est[-1].labels_,
                metric="euclidean",
                sample_size=300,
            )
        )
        all_results.append(results)

    formatter_result = """\033[1m\033[92mname\033[0m:   \033[1m\033[95m{:s}\033[0m
        elapsed: {:.3f}s
        accuracy: {:.2f}
        homogeneity: {:.3f}
        v-measure: {:.3f}
        adjusted rand score: {:.3f}
        adjusted mutual information: {:.3f}
        silhouette score {:.3f}"""
    print(formatter_result.format(name, *np.mean(all_results, axis=0)))


if __name__ == "__main__":
    datasets = {
        "Breast Cancer": load_breast_cancer(return_X_y=True),
        "Digits": load_digits(return_X_y=True),
        "Iris": load_iris(return_X_y=True),
        "Pima Indians Diabetes": load_diabetes(return_X_y=True),
    }
    for name, dataset in datasets.items():
        print(f"\033[1m{name:-^42}")
        data, labels = dataset
        size = np.unique(labels).size

        times = 50
        kmeans = KMeans(init="k-means++", n_clusters=size, n_init=4, random_state=0)
        bench_estimator(kmeans, "k-means++", data, labels, times)

        hdbscan = HDBSCAN(min_cluster_size=10, min_samples=5)
        bench_estimator(hdbscan, "hdbscan", data, labels, times)
