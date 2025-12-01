// Copyright (C) 2025 Piotr Jabłoński
// Extended copyright information can be found in the LICENSE file.

//! `SDC` (Symbol Deep Clustering) model and components.
//!
//! This module provides the main [`SDCConfig`] model, as well as supporting types for
//! autoencoder configuration [`AutoencoderConfig`], training [`TrainingConfig`], and dataset handling [`Dataset`].

mod autoencoder;
mod cdist;
mod clustering;
pub mod dataset;
mod inference;
mod loss;
pub mod metric;
mod model;
mod training;

pub use autoencoder::AutoencoderConfig;
pub use dataset::Dataset;
pub use inference::infer;
pub use model::SDCConfig;
pub use training::{TrainingConfig, train};
