use crate::sdc::cdist::pairwise_distances_squared;
use crate::sdc::metric::ClusteringInput;
use burn::prelude::{Backend, Int, Tensor};
use burn::tensor::Transaction;
use burn::train::metric::{Adaptor, ItemLazy, LossInput};
use burn_ndarray::NdArray;
use derive_new::new;

#[derive(new)]
pub struct ClusteringOutput<B: Backend> {
    pub centroids: Tensor<B, 2>,
    pub embeddings: Tensor<B, 2>,
    pub loss: Tensor<B, 1>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> ItemLazy for ClusteringOutput<B> {
    type ItemSync = ClusteringOutput<NdArray>;

    fn sync(self) -> Self::ItemSync {
        let [centroids, embeddings, loss, targets] = Transaction::default()
            .register(self.centroids)
            .register(self.embeddings)
            .register(self.loss)
            .register(self.targets)
            .execute()
            .try_into()
            .unwrap();

        let device = &Default::default();

        ClusteringOutput {
            centroids: Tensor::from_data(centroids, device),
            embeddings: Tensor::from_data(embeddings, device),
            loss: Tensor::from_data(loss, device),
            targets: Tensor::from_data(targets, device),
        }
    }
}

impl<B: Backend> Adaptor<ClusteringInput<B>> for ClusteringOutput<B> {
    fn adapt(&self) -> ClusteringInput<B> {
        let dist = pairwise_distances_squared(self.embeddings.clone(), self.centroids.clone());
        let q: Tensor<B, 2> = 1.0 / (1.0 + dist);
        ClusteringInput::new(q, self.targets.clone())
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for ClusteringOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}
