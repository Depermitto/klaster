// Copyright (C) 2025 Piotr Jabłoński
// Extended copyright information can be found in the LICENSE file.

//! Distance functions for KMeans clustering.

use ndarray::{ArrayView, Dimension, Zip};

/// Supported distance functions for KMeans clustering.
#[derive(Clone, Copy)]
pub enum KMeansDist {
    /// [Euclidean (L2) distance](https://en.wikipedia.org/wiki/Euclidean_distance).
    /// The "ordinary" straight-line distance between two points.
    ///
    /// Formula: `d(x,y) = sqrt(sum(x - y)^2)`
    Euclidean,

    /// [Manhattan (L1) distance](https://en.wikipedia.org/wiki/Taxicab_geometry).
    /// Also known as "taxicab" or "city block" distance.
    ///
    /// Formula: `d(x,y) = sum(|x - y|)`
    Manhattan,

    /// [Chebyshev (L∞) distance](https://en.wikipedia.org/wiki/Chebyshev_distance).
    /// The greatest difference along any coordinate dimension.
    ///
    /// Formula: `d(x,y) = max(|x - y|)`
    Chebyshev,

    /// [Minkowski distance](https://en.wikipedia.org/wiki/Minkowski_distance).
    /// Generalization of both Euclidean and Manhattan distances, controlled by the parameter `p`.
    ///
    /// For `p=1`, it is Manhattan distance; for `p=2`, it is Euclidean distance;
    /// for `p=∞`, it is Chebyshev distance; `p < 1` violates the triangle inequality and thus,
    /// is not a valid metric and constitutes undefined behaviour.
    ///
    /// Formula: `d(x,y) = (sum(|x - y|^p))^(1/p)`
    Minkowski(f64),

    /// [Cosine distance (1 - cosine similarity)](https://en.wikipedia.org/wiki/Cosine_similarity#Cosine_distance).
    /// Measures the cosine of the angle between two vectors (1 - cosine similarity).
    ///
    /// Formula: `d(x,y) = 1 - sum(x, y) / (|x| * |y|)`
    Cosine,
}

impl KMeansDist {
    /// Compute the distance between two points using the selected metric.
    pub fn run<D>(&self, a: ArrayView<f64, D>, b: ArrayView<f64, D>) -> f64
    where
        D: Dimension,
    {
        match self {
            KMeansDist::Euclidean => Zip::from(a)
                .and(b)
                .fold(0.0, |acc, x, y| acc + (x - y).powi(2))
                .sqrt(),
            KMeansDist::Manhattan => Zip::from(a)
                .and(b)
                .fold(0.0, |acc, x, y| acc + (x - y).abs()),
            KMeansDist::Chebyshev => Zip::from(a)
                .and(b)
                .fold(0.0, |acc, x, y| acc.max((x - y).abs())),
            KMeansDist::Minkowski(p) => Zip::from(a)
                .and(b)
                .fold(0.0, |acc, x, y| acc + (x - y).abs().powf(*p))
                .powf(1.0 / *p),
            KMeansDist::Cosine => {
                let dot = Zip::from(&a).and(&b).fold(0.0, |acc, x, y| acc + x * y);
                let norm_a = Zip::from(a).fold(0.0, |acc, x| acc + x.powi(2)).sqrt();
                let norm_b = Zip::from(b).fold(0.0, |acc, y| acc + y.powi(2)).sqrt();
                1.0 - (dot / (norm_a * norm_b))
            }
        }
    }
}
