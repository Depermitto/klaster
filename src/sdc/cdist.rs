// Copyright (C) 2025 Piotr Jabłoński
// Extended copyright information can be found in the LICENSE file.

use burn::prelude::Backend;
use burn::tensor::Tensor;

pub fn pairwise_distances_squared<B: Backend>(x1: Tensor<B, 2>, x2: Tensor<B, 2>) -> Tensor<B, 2> {
    // Expand dimensions for broadcasting
    let embeddings_expanded: Tensor<B, 3> = x1.unsqueeze_dim(1); // [batch_size, 1, embedding_dim]
    let centroids_expanded: Tensor<B, 3> = x2.unsqueeze_dim(0); // [1, num_clusters, embedding_dim]

    // Compute squared differences and sum along embedding dimension
    let squared_diff: Tensor<B, 3> = (embeddings_expanded - centroids_expanded).powi_scalar(2);
    squared_diff.sum_dim(2).squeeze(2) // [batch_size, num_clusters]
}
