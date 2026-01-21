// Copyright (C) 2025 Piotr Jabłoński
// Extended copyright information can be found in the LICENSE file.

fn main() {
    let n_clusters = 2;
    let expected_centroids = ndarray::array![[-1., 1., 1.], [8., 2., 2.]];

    let mut rng = ndarray_rand::rand::thread_rng();
    let data = linfa_datasets::generate::blobs(300, &expected_centroids, &mut rng);

    let model_fitted = klaster::KMeans::new_plusplus(n_clusters).fit(&data);
    println!("{:?}", model_fitted.centroids());
}
