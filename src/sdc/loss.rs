use crate::sdc::cdist::pairwise_distances_squared;
use burn::{
    nn::loss::{MseLoss, Reduction},
    prelude::{Backend, Tensor},
};
use derive_new::new;

#[derive(new)]
pub struct ClusteringLoss;

impl ClusteringLoss {
    pub fn forward<B: Backend, const D: usize>(
        &self,
        logits: Tensor<B, D>,
        targets: Tensor<B, D>,
        embeddings: Tensor<B, 2>,
        centroids: Tensor<B, 2>,
        gamma: f64,
        alpha: f64,
    ) -> Tensor<B, 1> {
        // Focal MSE (downright small errors, common in sparse data)
        let mse_loss = MseLoss::new().forward(logits.clone(), targets.clone(), Reduction::Mean);
        let focal_weight = (logits - targets).abs().powf_scalar(gamma);
        let focal_loss = focal_weight.mul(mse_loss.unsqueeze());
        let recon_loss = focal_loss.mean();

        // Clustering loss
        let dist = pairwise_distances_squared(embeddings, centroids);

        let q: Tensor<B, 2> = 1.0 / (1.0 + dist / alpha);
        let q: Tensor<B, 2> = q.powf_scalar((alpha + 1.0) / 2.0);
        let q: Tensor<B, 2> = q.clone() / q.sum_dim(1);

        let p: Tensor<B, 2> = q.clone().powi_scalar(2) / q.clone().sum_dim(0);
        let p: Tensor<B, 2> = p.clone() / p.sum_dim(1);

        let cluster_loss = ((p.clone() / (q + 1e-8)) + 1e-8)
            .log()
            .mul(p)
            .sum_dim(1)
            .mean();

        // Combined loss with weighting coefficients

        recon_loss + alpha * cluster_loss
    }
}
