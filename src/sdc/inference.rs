use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    prelude::Backend,
    record::{CompactRecorder, Recorder},
    train::metric::Adaptor,
};

use crate::sdc::{
    Dataset, TrainingConfig,
    dataset::ItemRaw,
    metric::{ClusteringMetricInput, acc_score, align_clusters},
    model::Centroids,
};

pub fn infer<B: Backend>(
    artifact_dir: &str,
    dataset: &Dataset,
    device: &B::Device,
    items: Vec<ItemRaw>,
) {
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
    let batch = batcher.batch(items.clone(), device);
    let output = model.forward_clustering(batch.images, batch.targets);

    // Align clusters to labels
    let metric_input: ClusteringMetricInput<B> = output.adapt();
    let y_pred = metric_input.y_pred();
    let y_true = metric_input.y_true();
    let aligned_preds = align_clusters(&y_pred, &y_true);

    // Print to compare
    for (p, t) in aligned_preds.iter().zip(y_true.iter()) {
        println!("p: {} | t: {}", p, t);
    }
    println!(
        "Correct: {}%",
        (acc_score(&y_pred, &y_true) * 100f64) as i32
    );
}
