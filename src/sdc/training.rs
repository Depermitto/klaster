use crate::KMeans;
use crate::sdc::model::Centroids;
use crate::sdc::{AutoencoderConfig, MnistBatcher, SDCConfig};
use burn::data::dataset::Dataset;
use burn::tensor::DType;
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        LearnerBuilder,
        metric::{AccuracyMetric, LossMetric},
    },
};
use ndarray::Array2;

#[derive(Config)]
pub struct TrainingConfig {
    pub model: SDCConfig,
    pub autoencoder: AutoencoderConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher = MnistBatcher::default();

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::train());

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::test());

    // Pretrain autoencoder
    let autoencoder_trained = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs * (3 / 10))
        .summary()
        .build(
            config.autoencoder.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        )
        .fit(dataloader_train.clone(), dataloader_test.clone());

    // Initialize centroids with K-Means
    let centroids = {
        let images = MnistDataset::test()
            .iter()
            .chain(MnistDataset::train().iter())
            .map(|item| TensorData::from(item.image).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 2>::from_data(data, &device))
            .map(|tensor| tensor.reshape([1, 1, 28, 28]))
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect();
        let images = Tensor::cat(images, 0);
        let (_, embeddings) = autoencoder_trained.forward(images);

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
            &device,
        ))
    };

    // Joint training
    LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config
                .model
                .init::<B>(autoencoder_trained, centroids, &device),
            config.optimizer.init(),
            config.learning_rate,
        )
        .fit(dataloader_train, dataloader_test)
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
