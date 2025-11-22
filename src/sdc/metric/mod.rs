mod acc;
mod ari;
mod nmi;

pub use acc::{ClusteringAccuracyMetric, acc_score, align_clusters};
pub use ari::{ARIMetric, ari_score};
pub use nmi::{NMIMetric, nmi_score};

use burn::{
    prelude::{Backend, Int, Tensor},
    tensor::DType,
};
use derive_new::new;

#[derive(new)]
pub struct ClusteringMetricInput<B: Backend> {
    clusters: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
}

impl<B: Backend> ClusteringMetricInput<B> {
    fn batch_size(&self) -> usize {
        let [batch_size, _] = self.clusters.dims();
        batch_size
    }

    pub fn y_true(&self) -> Vec<i32> {
        let clusters = self.clusters.clone();
        let batch_size = self.batch_size();
        let y_pred = clusters.argmax(1).reshape([batch_size]);
        y_pred
            .to_data()
            .convert_dtype(DType::I32)
            .to_vec::<i32>()
            .unwrap()
    }

    pub fn y_pred(&self) -> Vec<i32> {
        self.targets
            .to_data()
            .convert_dtype(DType::I32)
            .to_vec::<i32>()
            .unwrap()
    }
}
