// Copyright (C) 2025 Piotr Jabłoński
// Extended copyright information can be found in the LICENSE file.

use crate::sdc::autoencoder::Autoencoder;
use crate::sdc::clustering::ClusteringOutput;
use crate::sdc::dataset::Batch;
use crate::sdc::loss::ClusteringLoss;
use burn::module::Param;
use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{TrainOutput, TrainStep, ValidStep};

/// SDC model implementation combining an autoencoder and clustering head.
///
/// # Overview
/// Holds the learnable components used during training and inference, including the
/// convolutional autoencoder and cluster centroids.
///
/// # See also
/// [`SDCConfig`], [`crate::Autoencoder`], [`crate::AutoencoderConfig`]
#[derive(Module, Debug)]
pub struct SDC<B: Backend> {
    pub autoencoder: Autoencoder<B>,
    pub centroids: Param<Tensor<B, 2>>,
    alpha: f64,
    gamma: f64,
}

/// Configuration for the SDC model.
///
/// # Params
/// - `n_clusters`: Number of clusters to form,
/// - `latent_dim`: Dimensionality of the latent space,
/// - `alpha`: Weighting factor for the clustering loss,
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
    /// Do not initialize centroids (zero-filled).
    Empty,
    /// Initialize centroids from a random normal distribution.
    Random,
    /// User-provided centroids.
    Initialized(Tensor<B, 2>),
}

impl SDCConfig {
    /// Initialize an [`SDC`] model.
    ///
    /// # Params
    /// - `autoencoder`: Pretrained autoencoder instance,
    /// - `centroids`: Cluster centroids initialization strategy,
    /// - `device`: Target device for model parameters.
    pub fn init<B: Backend>(
        &self,
        autoencoder: Autoencoder<B>,
        centroids: Centroids<B>,
        device: &B::Device,
    ) -> SDC<B> {
        SDC {
            autoencoder,
            centroids: match centroids {
                Centroids::Empty => {
                    Param::from_tensor(Tensor::zeros([self.n_clusters, self.latent_dim], device))
                }
                Centroids::Random => Param::from_tensor(Tensor::random(
                    [self.n_clusters, self.latent_dim],
                    Distribution::Normal(0.0, 0.04),
                    device,
                )),
                Centroids::Initialized(centroids) => Param::from_tensor(centroids),
            },
            alpha: self.alpha,
            gamma: self.gamma,
        }
    }
}

impl<B: Backend> SDC<B> {
    /// Forward pass used for clustering training and evaluation.
    ///
    /// # Data layout
    /// - `x`: [batch, channels, height, width]
    /// - `targets`: \[batch\]
    ///
    /// # See also
    /// [`ClusteringOutput`]
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
            self.centroids.val(),
            self.gamma,
            self.alpha,
        );

        ClusteringOutput {
            centroids: self.centroids.val(),
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
