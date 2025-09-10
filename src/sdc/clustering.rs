use crate::sdc::cdist::pairwise_distances_squared;
use burn::prelude::{Backend, Int, Tensor};
use burn::tensor::Transaction;
use burn::train::metric::{AccuracyInput, Adaptor, ItemLazy, LossInput};
use burn_ndarray::NdArray;

pub struct ClusteringOutput<B: Backend> {
    pub centroids: Tensor<B, 2>,
    pub embeddings: Tensor<B, 2>,
    pub loss: Tensor<B, 1>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> ClusteringOutput<B> {
    fn predict_clusters(&self) -> Tensor<B, 2> {
        let dist = pairwise_distances_squared(self.embeddings.clone(), self.centroids.clone());
        let q: Tensor<B, 2> = 1.0 / (1.0 + dist);
        q.sqrt()
    }
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
            .expect("Correct amount of tensor data");

        let device = &Default::default();

        ClusteringOutput {
            centroids: Tensor::from_data(centroids, device),
            embeddings: Tensor::from_data(embeddings, device),
            loss: Tensor::from_data(loss, device),
            targets: Tensor::from_data(targets, device),
        }
    }
}

impl<B: Backend> Adaptor<AccuracyInput<B>> for ClusteringOutput<B> {
    fn adapt(&self) -> AccuracyInput<B> {
        // q = 1.0 / (1.0 + torch.cdist(embeddings, model.centroids) ** 2)
        // y_pred = torch.argmax(q, dim=1).cpu().numpy()
        AccuracyInput::new(self.predict_clusters(), self.targets.clone())
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for ClusteringOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}
