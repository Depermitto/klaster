use crate::KMeans;
use crate::sdc::dataset::Batch;
use crate::sdc::metric::{ARIMetric, ClusteringAccuracyMetric, NMIMetric};
use crate::sdc::model::Centroids;
use crate::sdc::{AutoencoderConfig, Dataset, SDCConfig};
use burn::module::AutodiffModule;
use burn::{
    data::{dataloader::DataLoaderBuilder, dataloader::batcher::Batcher, dataset::InMemDataset},
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::{DType, backend::AutodiffBackend},
    train::{LearnerBuilder, metric::LossMetric},
};
use ndarray::Array2;

#[derive(Config)]
pub struct TrainingConfig {
    pub model: SDCConfig,
    pub autoencoder: AutoencoderConfig,
    pub optimizer: AdamConfig,
    #[config(default = 65)]
    pub num_epochs: usize,
    #[config(default = 8)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub lr: f64,
    #[config(default = 0.6)]
    pub pretraining_period: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    dataset: Dataset,
    device: &B::Device,
) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher = dataset.batcher();
    let dataset_train = InMemDataset::new(dataset.train_items());
    let dataset_test = InMemDataset::new(dataset.test_items());

    let dataloader_train = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_train);

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_test);

    // Pretrain autoencoder
    let autoencoder_trained = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs((config.num_epochs as f64 * config.pretraining_period) as usize)
        .summary()
        .build(
            config.autoencoder.init::<B>(device),
            config.optimizer.init(),
            config.lr,
        )
        .fit(dataloader_train.clone(), dataloader_test.clone());

    // Initialize centroids with K-Means
    let centroids = {
        let whole: Batch<B> = batcher.batch(dataset.train_items(), device);
        let (_, embeddings) = autoencoder_trained.valid().forward(whole.images.valid());

        let embeddings_ndarray = unsafe {
            Array2::from_shape_vec_unchecked(
                embeddings.dims(),
                embeddings
                    .to_data()
                    .convert_dtype(DType::F64)
                    .to_vec()
                    .expect("Tensor data should be converted to ndarray successfully"),
            )
        };
        let kmeans_fitted = KMeans::new_plusplus(config.model.n_clusters).fit(&embeddings_ndarray);
        let centroids = kmeans_fitted.centroids();
        Centroids::Initialized(Tensor::from_data(
            TensorData::new(centroids.to_owned().into_raw_vec(), centroids.shape()),
            device,
        ))
    };

    // Joint training
    LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(ClusteringAccuracyMetric::new())
        .metric_valid_numeric(ClusteringAccuracyMetric::new())
        .metric_train_numeric(NMIMetric::new())
        .metric_valid_numeric(NMIMetric::new())
        .metric_train_numeric(ARIMetric::new())
        .metric_valid_numeric(ARIMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config
                .model
                .init::<B>(autoencoder_trained, centroids, device),
            config.optimizer.init(),
            config.lr,
        )
        .fit(dataloader_train, dataloader_test)
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
