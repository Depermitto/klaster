use hdbscan::Hdbscan;
use linfa::{
    Dataset,
    traits::{Fit, Predict, Transformer},
};
use linfa_clustering::KMeans;
use linfa_preprocessing::linear_scaling::LinearScaler;
use ndarray::s;
use ndarray_rand::rand::thread_rng;

use klaster::{metrics::benefit_of_doubt_acc, research::benchmark_runtime};

const NCLUSTERS: usize = 2;
const RUNS: usize = 200;

fn main() {
    // Load file into dataset
    let file = std::fs::File::open("/mnt/barracuda/Datasets/bcw.csv")
        .expect("Breast cancer file not found");
    let dataset = linfa_datasets::array_from_csv(file, true, b',').expect("Bad csv file read");

    let targets = dataset.column(1).to_owned();
    let y_true: Vec<usize> = targets.iter().map(|x| *x as usize).collect();
    let records = dataset.slice(s![.., 2..]).to_owned();
    let dataset = Dataset::new(records, targets);

    // Standardize features by removing the mean and scaling to unit variance
    let scaler = LinearScaler::standard()
        .fit(&dataset)
        .expect("Cannot scale dataset");
    let dataset = scaler.transform(dataset);

    let kmeans_fit_predict_measure = || {
        let rng = thread_rng();
        let model = KMeans::params_with_rng(NCLUSTERS, rng)
            .fit(&dataset)
            .expect("KMeans bad fit");
        let y_pred = model.predict(&dataset).to_vec();

        vec![benefit_of_doubt_acc(&y_true, &y_pred)]
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
        let y_pred: Vec<i32> = model.cluster().expect("HDBSCAN bad fit").to_vec();

        vec![benefit_of_doubt_acc(&y_true, &y_pred)]
    };
    println!(
        "HDBSCAN {}",
        benchmark_runtime(hdbscan_fit_predict_measure, RUNS)
    );
}
