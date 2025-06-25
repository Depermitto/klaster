// Copyright (C) 2025 Piotr Jabłoński
// Extended copyright information can be found in the LICENSE file.

//! KMeans clustering algorithm and related components.
//!
//! This module provides the main [`KMeans`] model, as well as supporting types for
//! centroid initialization ([`init`]) and distance metrics ([`dist`]).

pub mod dist;
pub mod init;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};

pub use crate::kmeans::init::KMeansInit;

/// K-Means clustering model.
///
/// # Overview
/// Performs K-Means clustering on input data, grouping samples into `k_clusters` clusters.
/// The algorithm supports customization of the initialization method, distance metric,
/// convergence criteria, and maximum iteration limit.
///
/// # Params
/// - `k_clusters`: Number of clusters to form (must be ≥ 1),
/// - `max_iter`: Maximum iterations of the algorithm (must be ≥ 1),
/// - `tolerance`: Relative tolerance for convergence (must be ≥ 0.0),
/// - `init_fn`: Cluster center initialization strategy,
/// - `dist_fn`: Distance metric for assignment and convergence.
///
/// # Panics
/// Panic can occur during initialization if:
/// - `k_clusters` is 0
/// - `max_iter` is 0
/// - `tolerance` is negative
pub struct KMeans {
    k_clusters: usize,
    max_iter: usize,
    tolerance: f64,
    init_fn: KMeansInit,
}

impl KMeans {
    /// Create a new KMeans model with random (Forgy) initialization and Euclidean distance.
    pub fn new_random(k_clusters: usize) -> Self {
        assert_ne!(k_clusters, 0);
        Self {
            k_clusters,
            init_fn: KMeansInit::Forgy,
            tolerance: 1e-4,
            max_iter: 300,
        }
    }

    /// Create a new KMeans model with KMeans++ initialization and Euclidean distance.
    pub fn new_plusplus(k_clusters: usize) -> Self {
        assert_ne!(k_clusters, 0);
        Self {
            init_fn: KMeansInit::PlusPlus,
            ..Self::new_random(k_clusters)
        }
    }

    /// Set the convergence tolerance for the KMeans model.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        assert!(tolerance > 0.0);
        self.tolerance = tolerance;
        self
    }

    /// Set the maximum number of iterations for the KMeans model.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        assert_ne!(max_iter, 0);
        self.max_iter = max_iter;
        self
    }

    /// Fit the KMeans model to the input data and return a fitted model.
    ///
    /// # Panics
    /// May occur if input `data` contains invalid values.
    pub fn fit(&self, data: ArrayView2<f64>) -> KMeansFitted {
        let mut rng = rand::rng();

        let mut centroids = self.init_fn.run(self.k_clusters, data, &mut rng);
        let mut memberships = Array1::zeros(data.nrows());

        for _ in 0..self.max_iter {
            // Assignment step
            assign_clusters(data, centroids.view(), &mut memberships);

            // Calculate new centroids (mean of all points assigned to each cluster)
            let new_centroids = {
                let mut new_centroids = Array2::<f64>::zeros((self.k_clusters, data.ncols()));
                let mut counts = Array1::<f64>::zeros(self.k_clusters);

                for (point, &membership) in data.outer_iter().zip(&memberships) {
                    let mut centroid = new_centroids.row_mut(membership);
                    centroid += &point;
                    counts[membership] += 1.0;
                }

                for (mut new_centroid, count) in new_centroids.outer_iter_mut().zip(counts) {
                    if count > 0.0 {
                        new_centroid /= count;
                    }
                }
                new_centroids
            };

            // Convergence check
            let distance = dist::naive_euclidean_sq(centroids.view(), new_centroids.view());
            centroids = new_centroids;
            if distance < self.tolerance {
                break;
            }
        }

        KMeansFitted { centroids }
    }

    /// Fit the KMeans model and return cluster assignments for each sample. This is equivalent to writing
    /// `.fit(data).predict(data)`
    ///
    /// # Panics
    /// May occur if input `data` contains invalid values.
    pub fn fit_predict(&self, data: ArrayView2<f64>) -> Array1<usize> {
        self.fit(data).predict(data)
    }
}

/// A fitted K-Means model containing learned cluster centroids and prediction methods.
///
/// Note: Use the [`centroids`](KMeansFitted::centroids) method to lookup final cluster centroids.
pub struct KMeansFitted {
    centroids: Array2<f64>,
}

impl KMeansFitted {
    /// Get a view of the learned centroids.
    pub fn centroids(&self) -> ArrayView2<f64> {
        self.centroids.view()
    }

    /// Assign clusters to the input data, writing results in-place.
    ///
    /// Note: `data` and `memberships` must agree on the length of their first dimension ([`ndarray::Axis(0)`](ndarray::Axis))
    pub fn predict_inplace(&self, data: ArrayView2<f64>, memberships: &mut Array1<usize>) {
        assert_eq!(data.nrows(), memberships.len());
        assign_clusters(data, self.centroids(), memberships);
    }

    /// Assign clusters to the input data and return the assignments.
    pub fn predict(&self, data: ArrayView2<f64>) -> Array1<usize> {
        let mut memberships = Array1::zeros(data.nrows());
        assign_clusters(data, self.centroids(), &mut memberships);
        memberships
    }
}

fn assign_clusters(
    data: ArrayView2<f64>,
    centroids: ArrayView2<f64>,
    memberships: &mut Array1<usize>,
) {
    Zip::from(data.outer_iter())
        .and(memberships)
        .for_each(|point, membership| {
            let (cluster_assignment, _) = closest_centroid(point, centroids);
            *membership = cluster_assignment;
        });
}

fn closest_centroid(point: ArrayView1<f64>, centroids: ArrayView2<f64>) -> (usize, f64) {
    if point.is_empty() || centroids.is_empty() {
        unreachable!()
    }
    let point_dot = point.dot(&point);

    let mut cluster_assignment = 0;
    let mut min_dist = f64::INFINITY;
    for (c_idx, centroid) in centroids.outer_iter().enumerate() {
        let dist = dist::euclidean_sq_lprecomputed(point, point_dot, centroid);
        if dist < min_dist {
            min_dist = dist;
            cluster_assignment = c_idx;
        }
    }
    (cluster_assignment, min_dist)
}
