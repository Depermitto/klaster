// Copyright (C) 2025 Piotr Jabłoński
// Extended copyright information can be found in the LICENSE file.

//! Centroid initialization strategies for KMeans clustering.

use ndarray::{Array1, Array2, ArrayView2, Axis, Zip, s};
use rand::distr::Distribution;

use crate::kmeans::closest_centroid;

/// Initialization methods for KMeans clustering.
///
/// - `Forgy`: Randomly selects `k` data points as initial centroids.
/// - `PlusPlus`: Uses the KMeans++ algorithm to choose initial centroids, spreading them out
///   by selecting each new centroid with probability proportional to its squared distance
///   from the nearest existing centroid.
#[derive(Clone, Copy)]
pub enum KMeansInit {
    Forgy,
    PlusPlus,
}

impl KMeansInit {
    /// Initialize centroids for KMeans clustering using the selected method.
    pub fn run(
        &self,
        k_clusters: usize,
        data: ArrayView2<f64>,
        rng: &mut impl rand::Rng,
    ) -> Array2<f64> {
        match self {
            KMeansInit::Forgy => {
                let (samples, _) = data.dim();
                let indices = rand::seq::index::sample(rng, samples, k_clusters).into_vec();
                data.select(Axis(0), &indices)
            }
            KMeansInit::PlusPlus => {
                let (samples, features) = data.dim();
                let mut centroids = Array2::<f64>::zeros((k_clusters, features));
                let mut weights = Array1::<f64>::zeros(samples);

                // Choose the first centroid at random among all the data points
                centroids
                    .row_mut(0)
                    .assign(&data.row(rng.random_range(0..samples)));

                for c_idx in 1..k_clusters {
                    // For each data point, compute the distance to its nearest already-chosen centroid.
                    // The probability of selecting a point as the next centroid is proportional to the
                    // square distance of the closest centroids
                    Zip::from(data.outer_iter())
                        .and(&mut weights)
                        .for_each(|point, weight| {
                            let (_, min_dist) =
                                closest_centroid(point, centroids.slice(s![0..c_idx, ..]));
                            *weight = min_dist;
                        });

                    let p_idx = rand::distr::weighted::WeightedIndex::new(weights.iter())
                        .map(|w_idx| w_idx.sample(rng))
                        .unwrap_or(0);
                    centroids.row_mut(c_idx).assign(&data.row(p_idx));
                }

                centroids
            }
        }
    }
}
