use hdbscan::{Center, Hdbscan};
use linfa_datasets::generate;
use ndarray::array;
use ndarray_rand::rand::thread_rng;

fn main() {
    let mut rng = thread_rng();
    let expected_centroids = array![[0., 1.], [-10., 20.], [-1., 10.]];
    let data = generate::blobs(100, &expected_centroids, &mut rng)
        .outer_iter()
        .map(|row| row.to_vec())
        .collect::<Vec<Vec<f64>>>();
    let clusterer = Hdbscan::default_hyper_params(&data);
    let labels = clusterer.cluster().unwrap();
    let centroids = clusterer.calc_centers(Center::Centroid, &labels).unwrap();
    println!("{:?}", centroids);
}
