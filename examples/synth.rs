use hdbscan::Hdbscan;
use linfa::{Dataset, traits::Fit};
use linfa_clustering::KMeans;
use linfa_datasets::generate;
use ndarray::{Axis, array};
use ndarray_rand::rand::thread_rng;

use klaster::research::benchmark_runtime;

const RUNS: usize = 1_000;

fn main() {
    let mut rng = thread_rng();
    let expected_centroids = array![[0., 1.], [-10., 20.], [-1., 10.]];
    let dataset = Dataset::from(generate::blobs(100, &expected_centroids, &mut rng));

    let kmeans_fit_predict_measure = || {
        let model = KMeans::params_with_rng(expected_centroids.len_of(Axis(0)), rng.clone())
            .fit(&dataset)
            .expect("KMeans bad fit");
        let centroids = model.centroids();
        centroids.iter().cloned().collect()
    };
    println!(
        "KMeans {}",
        benchmark_runtime(kmeans_fit_predict_measure, RUNS)
    );

    let data: Vec<Vec<f64>> = dataset
        .records()
        .outer_iter()
        .map(|row| row.to_vec())
        .collect();
    let hdbscan_fit_predict_measure = || {
        let model = Hdbscan::default_hyper_params(&data);
        let labels = model.cluster().unwrap();
        let centroids = model
            .calc_centers(hdbscan::Center::Centroid, &labels)
            .expect("HDBSCAN bad fit");
        centroids.into_iter().flatten().collect()
    };
    println!(
        "HDBSCAN {}",
        benchmark_runtime(hdbscan_fit_predict_measure, RUNS)
    );
}
