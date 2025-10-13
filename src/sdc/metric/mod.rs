mod acc;
mod nmi;

pub use acc::ClusteringAccuracyMetric;
pub use nmi::NMIMetric;

use burn::prelude::{Backend, Int, Tensor};
use derive_new::new;

#[derive(new)]
pub struct ClusteringInput<B: Backend> {
    clusters: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
}

impl<B: Backend> ClusteringInput<B> {
    fn batch_size(&self) -> usize {
        let [batch_size, _] = self.clusters.dims();
        batch_size
    }

    fn outputs_true(&self) -> Vec<i64> {
        let clusters = self.clusters.clone();
        let batch_size = self.batch_size();
        let y_pred = clusters.argmax(1).reshape([batch_size]);
        y_pred.to_data().to_vec::<i64>().unwrap()
    }

    fn outputs_pred(&self) -> Vec<i64> {
        self.targets.to_data().to_vec::<i64>().unwrap()
    }
}
