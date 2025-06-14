use hdbscan::Hdbscan;
use linfa::traits::{Fit, Predict, Transformer};
use linfa_clustering::KMeans;
use linfa_preprocessing::linear_scaling::LinearScaler;

use klaster::metrics::*;

const NCLUSTERS: usize = 10;

fn main() {
    // Load files into dataset
    let mnist::Mnist {
        trn_img, trn_lbl, ..
    } = mnist::MnistBuilder::new()
        .base_path("data/mnist")
        .finalize();

    let train_images = ndarray::Array2::from_shape_vec((60_000, 28 * 28), trn_img)
        .expect("MNIST bad image conversion")
        .mapv(|x| x as f32);
    let y_true =
        ndarray::Array1::from_shape_vec(60_000, trn_lbl).expect("MNIST bad label conversion");
    let train_labels = y_true.mapv(|x| x as f32);
    let dataset = linfa::Dataset::new(train_images, train_labels);

    // Standardize features by removing the mean and scaling to unit variance
    let scaler = LinearScaler::standard()
        .fit(&dataset)
        .expect("Cannot scale dataset");
    let dataset = scaler.transform(dataset);

    // KMeans: Fit and predict
    let model = KMeans::params(NCLUSTERS)
        .fit(&dataset)
        .expect("KMeans bad fit");
    let y_pred: Vec<u8> = model
        .predict(&dataset)
        .into_iter()
        .map(|x| x as u8)
        .collect();
    let y_true: Vec<u8> = y_true.to_vec();

    // KMeans: Judgement day
    let kmeans_acc = benefit_of_doubt_acc(&y_true, &y_pred);
    println!("KMeans: {}", kmeans_acc);

    // HDBSCAN: Fit and predict (takes 20 years to complete)
    let data: Vec<Vec<f32>> = dataset
        .records()
        .outer_iter()
        .map(|row| row.to_vec())
        .collect();
    let model = Hdbscan::default_hyper_params(&data);
    let y_pred: Vec<u8> = model
        .cluster()
        .expect("HDBSCAN bad fit")
        .into_iter()
        .map(|x| x as u8)
        .collect();

    // HDBSCAN: Judgement day
    let hdbscan_acc = benefit_of_doubt_acc(&y_true, &y_pred);
    println!("HDBSCAN: {}", hdbscan_acc);
}
