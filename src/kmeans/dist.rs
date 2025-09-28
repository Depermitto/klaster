// Copyright (C) 2025 Piotr Jabłoński
// Extended copyright information can be found in the LICENSE file.

//! Euclidean distance and functions used for optimizing distance measuring

use ndarray::{ArrayBase, Data, Dimension, Ix1, Zip};

#[inline]
pub fn euclidean_sq_precomputed(
    a: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    aa_dot: f64,
    b: &ArrayBase<impl Data<Elem = f64>, Ix1>,
) -> f64 {
    aa_dot - 2.0 * a.dot(b) + b.dot(b)
}

#[inline]
pub fn naive_euclidean_sq<D: Dimension>(
    a: &ArrayBase<impl Data<Elem = f64>, D>,
    b: &ArrayBase<impl Data<Elem = f64>, D>,
) -> f64 {
    assert_eq!(a.dim(), b.dim());
    let a_flat = a.to_shape(a.len());
    let b_flat = b.to_shape(b.len());
    if let (Ok(a), Ok(b)) = (a_flat, b_flat) {
        euclidean_sq_precomputed(&a, a.dot(&a), &b)
    } else {
        Zip::from(a)
            .and(b)
            .fold(0.0, |acc, x, y| acc + (x - y).powi(2))
    }
}
