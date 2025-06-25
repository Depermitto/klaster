// Copyright (C) 2025 Piotr Jabłoński
// Extended copyright information can be found in the LICENSE file.

//! Euclidean distance and functions used for optimizing distance measuring

use ndarray::{Array1, ArrayView, ArrayView1, ArrayView2, Dimension, Zip};

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
pub fn euclidean_sq(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    fast_euclidean_sq(a.dot(&a), a.dot(&b), b.dot(&b))
}

#[inline]
pub fn euclidean_sq_lprecomputed(a: ArrayView1<f64>, aa_dot: f64, b: ArrayView1<f64>) -> f64 {
    fast_euclidean_sq(aa_dot, a.dot(&b), b.dot(&b))
}

#[inline]
pub fn euclidean_sq_rprecomputed(a: ArrayView1<f64>, b: ArrayView1<f64>, bb_dot: f64) -> f64 {
    fast_euclidean_sq(a.dot(&a), a.dot(&b), bb_dot)
}

#[inline]
fn fast_euclidean_sq(aa_dot: f64, ab_dot: f64, bb_dot: f64) -> f64 {
    aa_dot - 2.0 * ab_dot + bb_dot
}

pub fn precompute_dot_products(array: ArrayView2<f64>) -> Array1<f64> {
    array.outer_iter().map(|point| point.dot(&point)).collect()
}

#[inline]
pub fn dotnd<D>(array: ArrayView<f64, D>) -> f64
where
    D: Dimension,
{
    array
        .to_shape(array.len())
        .map(|flat| flat.dot(&flat))
        .unwrap_or(f64::NAN)
}

#[inline]
pub fn naive_euclidean_sq<D>(a: ArrayView<f64, D>, b: ArrayView<f64, D>) -> f64
where
    D: Dimension,
{
    Zip::from(a)
        .and(b)
        .fold(0.0, |acc, x, y| acc + (x - y).powi(2))
}
