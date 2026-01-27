// Copyright (C) 2025 Piotr Jabłoński
// Extended copyright information can be found in the LICENSE file.

//! `SDC` (Symbol Deep Clustering) model and components.
//!
//! Provides the [`SDCConfig`] model configuration along with supporting types for
//! autoencoder setup ([`AutoencoderConfig`]), training ([`TrainingConfig`]), inference
//! ([`infer`]), metrics ([`metric`]), and dataset handling ([`dataset::Dataset`]).

mod autoencoder;
mod cdist;
mod clustering;
pub mod dataset;
mod inference;
mod loss;
mod model;
mod training;

pub use autoencoder::{Autoencoder, AutoencoderConfig};
pub use clustering::ClusteringOutput;
pub use inference::infer;
pub use model::{SDC, SDCConfig};
pub use training::{TrainingConfig, train};
