// Copyright (C) 2025 Piotr Jabłoński
// Extended copyright information can be found in the LICENSE file.

use burn::optim::AdamConfig;
use klaster::sdc::*;
use rand::{rng, seq::SliceRandom};

const DATASET_DIR: &str = "datasets";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = Dataset::unipen(&format!("{DATASET_DIR}/UNIPEN-64x64-grayscale"))?;
    let latent_dim = 128;
    // let dataset = Dataset::mnist(&format!("{DATASET_DIR}/MNIST/raw"));
    // let latent_dim = 10;
    let artifact_dir = "/tmp/sdc";
    let device = &Default::default();
    train::<burn::backend::Autodiff<burn::backend::Wgpu>>(
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
    let mut test_items = dataset.test_items();
    test_items.shuffle(&mut rng);
    infer::<burn::backend::Wgpu>(artifact_dir, &dataset, device, test_items[0..256].to_vec());

    Ok(())
}
