// Copyright (C) 2025 Piotr Jabłoński
// Extended copyright information can be found in the LICENSE file.

use burn::optim::AdamConfig;
use clap::{Arg, Command};
use klaster::*;
use rand::{rng, seq::SliceRandom};

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
    train::<burn::backend::Autodiff<burn::backend::Vulkan>>(
        artifact_dir,
        TrainingConfig::new(
            SDCConfig::new(dbg!(dataset.n_classes()), latent_dim).with_alpha(1.05),
            AutoencoderConfig::new(latent_dim, dataset.item_dims, [1, 32, 64], 8),
            AdamConfig::new(),
        )
        .with_num_epochs(10)
        .with_lr(0.00183)
        .with_batch_size(16),
        &dataset,
        device,
    );

    let mut rng = rng();
    let mut items = dataset.test_items();
    if items.is_empty() {
        items = dataset.train_items();
    }
    items.shuffle(&mut rng);
    let n = std::cmp::min(256, items.len());
    infer::<burn::backend::Vulkan>(artifact_dir, &dataset, device, items[0..n].to_vec());

    Ok(())
}
