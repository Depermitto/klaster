use linfa::DatasetBase;
use linfa::traits::Fit;
use linfa_clustering::KMeans;
use linfa_datasets::generate;
use ndarray::{Axis, array};
use ndarray_rand::rand::thread_rng;

fn main() {
    let mut rng = thread_rng();
    let expected_centroids = array![[0., 1.], [-10., 20.], [-1., 10.]];
    let data = generate::blobs(100, &expected_centroids, &mut rng);
    let n_clusters = expected_centroids.len_of(Axis(0));
    let observations = DatasetBase::from(data.clone());
    let model = KMeans::params_with_rng(n_clusters, rng.clone())
        .tolerance(1e-2)
        .fit(&observations)
        .expect("KMeans fitted");
    let centroids = model.centroids();
    println!("{:?}", centroids);
}
