// Copyright (C) 2025 Piotr Jabłoński
// Extended copyright information can be found in the LICENSE file.

use burn::optim::AdamConfig;
use clap::{Arg, Command};
use klaster::*;
use rand::{rng, seq::SliceRandom};

type MyBackend = burn::backend::Vulkan;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("sdc")
        .about("Train the SDC model and run inference for test data")
        .arg(
            Arg::new("dataset")
                .long("dataset")
                .required(true)
                .value_parser(["mnist", "unipen"])
                .help("Dataset to use"),
        )
        .arg(
            Arg::new("dataset_path")
                .long("dataset-path")
                .required(true)
                .help("Dataset path to find the dataset in"),
        )
        .get_matches();

    let dataset_name = matches
        .get_one::<String>("dataset")
        .expect("dataset argument missing")
        .as_str();
    let dataset_path = matches
        .get_one::<String>("dataset_path")
        .expect("dataset-path argument missing");

    let (dataset, latent_dim) = match dataset_name {
        "mnist" => (Dataset::mnist(dataset_path), 10),
        "unipen" => (Dataset::unipen(dataset_path)?, 128),
        _ => unreachable!(),
    };
    let artifact_dir = "/tmp/sdc";
    let device = &Default::default();

    // Train
    train::<burn::backend::Autodiff<MyBackend>>(
        artifact_dir,
        TrainingConfig::new(
            SDCConfig::new(dbg!(dataset.n_classes()), latent_dim).with_alpha(1.05),
            AutoencoderConfig::new(latent_dim, dataset.item_dims, [1, 32, 64], 32),
            AdamConfig::new(),
        )
        .with_num_epochs(10)
        .with_lr(0.005)
        .with_batch_size(64),
        &dataset,
        device,
    );

    let mut rng = rng();
    // Load subset of the test dataset
    let mut items = dataset.test_items();
    if items.is_empty() {
        items = dataset.train_items();
    }
    items.shuffle(&mut rng);
    let n = std::cmp::min(1 << 13, items.len());

    // Infer
    let (y_pred, y_true) = infer::<MyBackend>(artifact_dir, &dataset, device, items[0..n].to_vec());

    // Print to compare
    println!(
        "Accuracy: {}%\nNMI: {}\nARI: {}",
        (metric::acc_score(&y_pred, &y_true) * 100f64) as i32,
        metric::nmi_score(&y_pred, &y_true),
        metric::ari_score(&y_pred, &y_true)
    );

    Ok(())
}
