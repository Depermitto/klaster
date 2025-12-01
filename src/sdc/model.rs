// Copyright (C) 2025 Piotr Jabłoński
// Extended copyright information can be found in the LICENSE file.

use crate::sdc::autoencoder::Autoencoder;
use crate::sdc::clustering::ClusteringOutput;
use crate::sdc::dataset::Batch;
use crate::sdc::loss::ClusteringLoss;
use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{TrainOutput, TrainStep, ValidStep};

#[derive(Module, Debug)]
pub struct SDC<B: Backend> {
    pub autoencoder: Autoencoder<B>,
    pub centroids: Tensor<B, 2>,
    alpha: f64,
    gamma: f64,
}

/// Configuration for the SDC model.
///
/// # Params
/// - `n_clusters`: Number of clusters to form.
/// - `latent_dim`: Dimensionality of the latent space.
/// - `alpha`: Weighting factor for the clustering loss.
/// - `gamma`: Weighting factor for the reconstruction loss.
///
/// # See also
/// [`SDCConfig::init`]
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
    Empty,
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
                Centroids::Empty => Tensor::zeros([self.n_clusters, self.latent_dim], device),
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
    pub fn forward_clustering(
        &self,
        x: Tensor<B, 4>,
        targets: Tensor<B, 1, Int>,
    ) -> ClusteringOutput<B> {
        let (recon, embeddings) = self.autoencoder.forward(x.clone());

        let loss = ClusteringLoss::new().forward::<B, 4>(
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

impl<B: AutodiffBackend> TrainStep<Batch<B>, ClusteringOutput<B>> for SDC<B> {
    fn step(&self, batch: Batch<B>) -> TrainOutput<ClusteringOutput<B>> {
        let item = self.forward_clustering(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<Batch<B>, ClusteringOutput<B>> for SDC<B> {
    fn step(&self, batch: Batch<B>) -> ClusteringOutput<B> {
        self.forward_clustering(batch.images, batch.targets)
    }
}
