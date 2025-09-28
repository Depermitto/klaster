use crate::sdc::autoencoder::Autoencoder;
use crate::sdc::clustering::ClusteringOutput;
use crate::sdc::loss::ClusterLoss;
use crate::sdc::mnist_data::MnistBatch;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Distribution;
use burn::train::{TrainOutput, TrainStep, ValidStep};

#[derive(Module, Debug)]
pub struct SDC<B: Backend> {
    autoencoder: Autoencoder<B>,
    centroids: Tensor<B, 2>,
    alpha: f64,
    gamma: f64,
}

#[derive(Config, Debug)]
pub struct SDCConfig {
    pub n_clusters: usize,
    pub latent_dim: usize,
    #[config(default = "1.0")]
    pub alpha: f64,
    #[config(default = "2.0")]
    pub gamma: f64,
}

pub enum Centroids<B: Backend> {
    Random,
    Initialized(Tensor<B, 2>),
}

impl SDCConfig {
    pub fn init<B: Backend>(
        &self,
        autoencoder: Autoencoder<B>,
        centroids: Centroids<B>,
        device: &B::Device,
    ) -> SDC<B> {
        SDC {
            autoencoder,
            centroids: match centroids {
                Centroids::Random => Tensor::random(
                    [self.n_clusters, self.latent_dim],
                    Distribution::Normal(0.0, 0.04),
                    device,
                ),
                Centroids::Initialized(centroids) => centroids,
            },
            alpha: self.alpha,
            gamma: self.gamma,
        }
    }
}

impl<B: Backend> SDC<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 2>) {
        self.autoencoder.forward(x)
    }

    pub fn forward_clustering(
        &self,
        x: Tensor<B, 4>,
        targets: Tensor<B, 1, Int>,
    ) -> ClusteringOutput<B> {
        let (recon, embeddings) = self.forward(x.clone());

        let loss = ClusterLoss::new().forward::<B, 4>(
            x,
            recon,
            embeddings.clone(),
            self.centroids.clone(),
            2.0,
            1.0,
        );

        ClusteringOutput {
            centroids: self.centroids.clone(),
            embeddings,
            loss,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClusteringOutput<B>> for SDC<B> {
    fn step(&self, batch: MnistBatch<B>) -> TrainOutput<ClusteringOutput<B>> {
        let item = self.forward_clustering(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MnistBatch<B>, ClusteringOutput<B>> for SDC<B> {
    fn step(&self, batch: MnistBatch<B>) -> ClusteringOutput<B> {
        self.forward_clustering(batch.images, batch.targets)
    }
}
