// Copyright (C) 2025 Piotr Jabłoński
// Extended copyright information can be found in the LICENSE file.

use std::{collections::HashSet, fs, path::Path};

use clap::{Arg, Command};
use hdbscan::{Hdbscan, HdbscanHyperParams};
use linfa::{
    Dataset,
    traits::{Fit, Predict, Transformer},
};
use linfa_datasets::generate;
use linfa_preprocessing::linear_scaling::LinearScaler;
use metrics::{benchmark_runtime, benefit_of_doubt_acc};
use ndarray::{Array2, s};
use ndarray_rand::rand::{Rng, thread_rng};

mod metrics;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("Clustering Benchmark")
        .arg(
            Arg::new("alg")
                .long("alg")
                .required(true)
                .value_parser(["kmeans-ref", "kmeans", "hdbscan-ref", "hdbscan", "n2d"])
                .help("Algorithm to use"),
        )
        .arg(
            Arg::new("hdbscan_min_cluster_size")
                .long("hdbscan-min-cluster-size")
                .default_value("10")
                .help("min_cluster_size for HDBSCAN"),
        )
        .arg(
            Arg::new("hdbscan_min_samples")
                .long("hdbscan-min-samples")
                .default_value("5")
                .help("min_samples for HDBSCAN"),
        )
        .arg(
            Arg::new("n2d_epochs")
                .long("n2d-epochs")
                .default_value("1000")
                .help("Number of epochs for n2d autoencoder training"),
        )
        .arg(
            Arg::new("n2d_arch")
                .long("n2d-arch")
                .default_value("500,500,2000")
                .help("Comma-separated layer sizes for n2d autoencoder architecture"),
        )
        .arg(
            Arg::new("n2d_verbose")
                .long("n2d-verbose")
                .action(clap::ArgAction::SetTrue)
                .help("Enable verbose output for n2d training"),
        )
        .arg(
            Arg::new("dataset")
                .long("dataset")
                .required(true)
                .value_parser(["synth", "bcw", "wine", "mnist", "20-newsgroups"])
                .help("Dataset to use"),
        )
        .arg(
            Arg::new("scaled")
                .long("scaled")
                .action(clap::ArgAction::SetTrue)
                .help("Standardize the dataset before clustering"),
        )
        .arg(
            Arg::new("blobs_samples")
                .long("blobs-samples")
                .default_value("300")
                .help("Number of samples for synthetic blobs dataset"),
        )
        .arg(
            Arg::new("blobs_centers")
                .long("blobs-centers")
                .default_value("3")
                .help("Number of centers for synthetic blobs dataset"),
        )
        .arg(
            Arg::new("blobs_features")
                .long("blobs-features")
                .default_value("2")
                .help("Number of features for synthetic blobs dataset"),
        )
        .arg(
            Arg::new("runs")
                .long("runs")
                .default_value("50")
                .help("Number of runs to average results over"),
        )
        .get_matches();

    let mut rng = thread_rng();
    let mut dataset_name = matches
        .get_one::<String>("dataset")
        .expect("dataset argument missing")
        .clone();
    let (n_clusters, mut dataset) = match dataset_name.as_str() {
        "synth" => {
            let blobs_samples = matches
                .get_one::<String>("blobs_samples")
                .ok_or("missing blobs-samples")?
                .parse::<usize>()?;
            let blobs_centers = matches
                .get_one::<String>("blobs_centers")
                .ok_or("missing blobs-centers")?
                .parse::<usize>()?;
            let blobs_features = matches
                .get_one::<String>("blobs_features")
                .ok_or("missing blobs-features")?
                .parse::<usize>()?;

            let mut expected_centroids = Array2::<f64>::zeros((blobs_centers, blobs_features));
            for i in 0..blobs_centers {
                for j in 0..blobs_features {
                    expected_centroids[[i, j]] = rng.gen_range(-10.0..10.0);
                }
            }
            println!("expected centroids:\n{}", expected_centroids);

            let dataset = Dataset::from(generate::blobs(
                blobs_samples,
                &expected_centroids,
                &mut rng,
            ));
            let dataset = dataset.with_targets(expected_centroids);
            todo!("do something with {:?}", dataset);
        }
        "bcw" => {
            let file = std::fs::File::open("data/bcw.csv").expect("BCW dataset not found");
            let dataset = linfa_datasets::array_from_csv(file, true, b',').expect("bad csv file");

            let targets = dataset.column(1).to_owned();
            let y_true: Vec<usize> = targets.iter().map(|x| *x as usize).collect();
            let records = dataset.slice(s![.., 2..]).to_owned();
            let dataset = Dataset::new(records, y_true.into());

            (2, dataset)
        }
        "wine" => {
            let file =
                std::fs::File::open("data/winequality-red.csv").expect("wine dataset not found");
            let dataset = linfa_datasets::array_from_csv(file, true, b',').expect("bad csv file");

            let targets = dataset.column(11).to_owned();
            println!(
                "{:?}",
                targets.iter().map(|x| *x as usize).collect::<HashSet<_>>()
            );
            let y_true: Vec<usize> = targets.iter().map(|x| *x as usize).collect();
            let records = dataset.slice(s![.., ..-2]).to_owned();
            println!("{:?}", records);
            let dataset = Dataset::new(records, y_true.into());

            (6, dataset)
        }
        "mnist" => {
            let mnist::Mnist {
                trn_img, trn_lbl, ..
            } = mnist::MnistBuilder::new()
                .base_path("data/mnist")
                .finalize();

            const SUBSET: usize = 1_000;
            let trn_img = trn_img[..SUBSET * 28 * 28].to_vec();
            let trn_lbl = trn_lbl[..SUBSET].to_vec();

            let train_images = ndarray::Array2::from_shape_vec((SUBSET, 28 * 28), trn_img)
                .expect("MNIST bad image conversion")
                .mapv(|x| x as f64);
            let y_true = ndarray::Array1::from_shape_vec(SUBSET, trn_lbl)
                .expect("MNIST bad label conversion");
            let train_labels = y_true.mapv(|x| x as usize);
            let dataset = linfa::Dataset::new(train_images, train_labels);

            (10, dataset)
        }
        "20-newsgroups" => unimplemented!(),
        _ => return Err("unknown dataset".into()),
    };

    if matches.get_flag("scaled") {
        let scaler = LinearScaler::standard()
            .fit(&dataset)
            .unwrap_or_else(|_| panic!("cannot scale {}", dataset_name));
        dataset_name = format!("{}-scaled", dataset_name);
        dataset = scaler.transform(dataset);
    }

    let runs: usize = matches
        .get_one::<String>("runs")
        .ok_or("missing runs")?
        .parse()?;
    let result = match matches
        .get_one::<String>("alg")
        .ok_or("missing alg")?
        .as_str()
    {
        "kmeans-ref" => {
            let y_true = dataset.targets().to_vec();

            benchmark_runtime(runs, || {
                let rng = thread_rng();
                let model = linfa_clustering::KMeans::params_with_rng(n_clusters, rng.clone())
                    .init_method(linfa_clustering::KMeansInit::Random)
                    .fit(&dataset)
                    .expect("KMeans bad fit");
                let y_pred = model.predict(&dataset).to_vec();

                vec![benefit_of_doubt_acc(&y_true, &y_pred)]
            })
        }
        "kmeans" => {
            let y_true = dataset.targets().to_vec();

            benchmark_runtime(runs, || {
                let y_pred = klaster::KMeans::new_plusplus(n_clusters)
                    .fit_predict(dataset.records().view())
                    .to_vec();

                vec![benefit_of_doubt_acc(&y_true, &y_pred)]
            })
        }
        "hdbscan-ref" => {
            let min_cluster_size: usize = matches
                .get_one::<String>("hdbscan_min_cluster_size")
                .ok_or("missing hdbscan_min_cluster_size")?
                .parse()?;
            let min_samples: usize = matches
                .get_one::<String>("hdbscan_min_samples")
                .ok_or("missing hdbscan_min_samples")?
                .parse()?;
            let config = HdbscanHyperParams::builder()
                .min_cluster_size(min_cluster_size)
                .min_samples(min_samples)
                .build();

            let data: Vec<Vec<f64>> = dataset
                .records()
                .outer_iter()
                .map(|row| row.to_vec())
                .collect();
            let y_true = dataset.targets().to_vec();

            benchmark_runtime(runs, || {
                let model = Hdbscan::new(&data, config.clone());
                let y_pred: Vec<i32> = model.cluster().expect("HDBSCAN bad fit").to_vec();

                vec![benefit_of_doubt_acc(&y_true, &y_pred)]
            })
        }
        "hdbscan" => unimplemented!(),
        "n2d-ref" => panic!("no 3rd party implementation available"),
        "n2d" => unimplemented!(),
        _ => return Err("unknown algorithm".into()),
    };

    let output_dir = Path::new(file!()).parent().ok_or("how?")?.join("output");
    if !output_dir.exists() {
        fs::create_dir(&output_dir)?;
    }
    let timestamp = chrono::Local::now().format("%d.%m.%Y-%H:%M:%S").to_string();
    let filename = format!(
        "{}-{}-{}.json",
        timestamp,
        matches.get_one::<String>("alg").unwrap(),
        dataset_name
    );
    let filepath = output_dir.join(filename);
    fs::write(filepath, serde_json::to_string_pretty(&result)?)?;

    println!("{}", result);

    Ok(())
}
