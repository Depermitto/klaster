// Copyright (C) 2025 Piotr Jabłoński
// Extended copyright information can be found in the LICENSE file.

//! Euclidean distance and functions used for optimizing distance measuring

use ndarray::{ArrayView, ArrayView1, Dimension, Zip};

// pub fn euclidean_sq<D>(a: ArrayView<f64, D>, b: ArrayView<f64, D>) -> f64
// where
//     D: Dimension,
// {
//     assert_eq!(a.dim(), b.dim());
//     let a_flat = a.to_shape(a.len());
//     let b_flat = b.to_shape(b.len());
//     if let (Ok(a), Ok(b)) = (a_flat, b_flat) {
//         fast_euclidean_sq(a.dot(&a), a.dot(&b), b.dot(&b))
//     } else {
//         fallback_euclidean_sq(a, b)
//     }
// }

#[inline]
pub fn euclidean_sq_precomputed(a: ArrayView1<f64>, aa_dot: f64, b: ArrayView1<f64>) -> f64 {
    aa_dot - 2.0 * a.dot(&b) + b.dot(&b)
}

#[inline]
pub fn naive_euclidean_sq<D>(a: ArrayView<f64, D>, b: ArrayView<f64, D>) -> f64
where
    D: Dimension,
{
    assert_eq!(a.dim(), b.dim());
    let a_flat = a.to_shape(a.len());
    let b_flat = b.to_shape(b.len());
    if let (Ok(a), Ok(b)) = (a_flat, b_flat) {
        euclidean_sq_precomputed(a.view(), a.dot(&a), b.view())
    } else {
        Zip::from(a)
            .and(b)
            .fold(0.0, |acc, x, y| acc + (x - y).powi(2))
    }
}
