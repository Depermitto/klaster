// Copyright (C) 2025 Piotr Jabłoński
// Extended copyright information can be found in the LICENSE file.

use klaster::KMeans;
use linfa_datasets::generate;
use ndarray::array;
use ndarray_rand::rand::thread_rng;

fn main() {
    let expected_centroids = array![[-1., 1., 1.], [8., 2., 2.]];
    let k_clusters = expected_centroids.nrows();

    let mut rng = thread_rng();
    let data = generate::blobs(300, &expected_centroids, &mut rng);

    let model_fitted = KMeans::new_plusplus(k_clusters)
        .with_max_iter(100)
        .with_tolerance(1e-6)
        .fit(&data);
    println!("{:?}", model_fitted.centroids());
}
