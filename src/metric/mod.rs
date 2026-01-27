// Copyright (C) 2025 Piotr Jabłoński
// Extended copyright information can be found in the LICENSE file.

//! Metrics used for evaluating clustering quality.
//! Provides Accuracy, ARI, and NMI.

mod acc;
mod ari;
mod nmi;

pub use acc::acc_score;
pub use ari::ari_score;
pub use nmi::nmi_score;

pub(crate) use acc::{ClusteringAccuracyMetric, align_clusters};
pub(crate) use ari::ARIMetric;
pub(crate) use nmi::NMIMetric;

use burn::{
    prelude::{Backend, Int, Tensor},
    tensor::DType,
};
use derive_new::new;

/// Input wrapper for clustering metrics.
#[derive(new)]
pub(crate) struct ClusteringMetricInput<B: Backend> {
    clusters: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
}

impl<B: Backend> ClusteringMetricInput<B> {
    fn batch_size(&self) -> usize {
        let [batch_size, _] = self.clusters.dims();
        batch_size
    }

    pub(crate) fn y_pred(&self) -> Vec<i32> {
        let clusters = self.clusters.clone();
        let batch_size = self.batch_size();
        let y_pred = clusters.argmax(1).reshape([batch_size]);
        y_pred
            .to_data()
            .convert_dtype(DType::I32)
            .to_vec::<i32>()
            .unwrap()
    }

    pub(crate) fn y_true(&self) -> Vec<i32> {
        self.targets
            .to_data()
            .convert_dtype(DType::I32)
            .to_vec::<i32>()
            .unwrap()
    }
}
