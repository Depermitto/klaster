// Copyright (C) 2025 Piotr Jabłoński
// Extended copyright information can be found in the LICENSE file.

use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    prelude::Backend,
    record::{CompactRecorder, Recorder},
    train::metric::Adaptor,
};

use crate::{
    TrainingConfig,
    metric::ClusteringMetricInput,
    sdc::{
        dataset::{Dataset, ItemRaw},
        model::Centroids,
    },
};

/// Perform inference with a trained SDC model.
///
/// # Arguments
///
/// * `artifact_dir`: Directory where the trained model is saved, expected `artifact_dir/model`,
/// * `dataset`: Dataset to use for inference,
/// * `device`: Device to use for inference,
/// * `items`: Items to perform inference on.
///
/// # Returns
/// Tuple of unaligned predictions and the ground truth: (y_pred, y_true)
///
/// # See also
/// [`crate::sdc::train`], [`crate::metric::acc_score`], [`crate::metric::nmi_score`], [`crate::metric::ari_score`]
pub fn infer<B: Backend>(
    artifact_dir: &str,
    dataset: &Dataset,
    device: &B::Device,
    items: Vec<ItemRaw>,
) -> (Vec<i32>, Vec<i32>) {
    // Load trained model
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), device)
        .expect("Trained model should exist; run train first");
    let model = config
        .model
        .init::<B>(config.autoencoder.init(device), Centroids::Empty, device)
        .load_record(record);

    // Predict clusters
    let batcher = dataset.batcher();
    let batch = batcher.batch(items, device);
    let output = model.forward_clustering(batch.images, batch.targets);

    // Align clusters to labels
    let metric_input: ClusteringMetricInput<B> = output.adapt();
    let y_pred = metric_input.y_pred();
    let y_true = metric_input.y_true();

    (y_pred, y_true)
}
