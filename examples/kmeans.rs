// Copyright (C) 2025 Piotr Jabłoński
// Extended copyright information can be found in the LICENSE file.

fn main() {
    let expected_centroids = ndarray::array![[-1., 1., 1.], [8., 2., 2.]];
    let k_clusters = expected_centroids.nrows();

    let mut rng = ndarray_rand::rand::thread_rng();
    let data = linfa_datasets::generate::blobs(300, &expected_centroids, &mut rng);

    let model_fitted = klaster::KMeans::new_plusplus(k_clusters)
        .with_max_iter(100)
        .with_tolerance(1e-6)
        .fit(&data);
    println!("{:?}", model_fitted.centroids());
}
