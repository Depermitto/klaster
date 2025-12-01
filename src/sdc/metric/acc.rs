// Copyright (C) 2025 Piotr Jabłoński
// Extended copyright information can be found in the LICENSE file.

use crate::sdc::metric::ClusteringMetricInput;
use burn::prelude::*;
use burn::train::metric::state::{FormatOptions, NumericMetricState};
use burn::train::metric::{Metric, MetricEntry, MetricMetadata, Numeric};
use std::collections::HashMap;

#[derive(Default)]
pub struct ClusteringAccuracyMetric<B: Backend> {
    state: NumericMetricState,
    _b: B,
}

impl<B: Backend> ClusteringAccuracyMetric<B> {
    pub fn new() -> Self {
        Self::default()
    }
}

pub fn align_clusters<T>(y_pred: &[T], y_true: &[T]) -> Vec<T>
where
    T: std::cmp::Eq + std::hash::Hash + Copy,
{
    let mut cluster_to_labels: HashMap<_, Vec<_>> = HashMap::new();
    for (&pred, &true_label) in y_pred.iter().zip(y_true.iter()) {
        cluster_to_labels.entry(pred).or_default().push(true_label);
    }

    let mut global_counts: HashMap<_, usize> = HashMap::new();
    for &label in y_true {
        *global_counts.entry(label).or_insert(0) += 1;
    }

    let Some((&most_common_global_label, _)) = global_counts.iter().max_by_key(|e| e.1) else {
        return vec![];
    };

    let mut label_mapping: HashMap<_, _> = HashMap::new();
    for (&cluster, labels) in &cluster_to_labels {
        let mut counts = HashMap::new();
        for &label in labels {
            *counts.entry(label).or_insert(0) += 1;
        }
        let majority_label = if let Some((&label, _)) = counts.iter().max_by_key(|e| e.1) {
            label
        } else {
            most_common_global_label
        };
        label_mapping.insert(cluster, majority_label);
    }

    let aligned_preds: Vec<_> = y_pred
        .iter()
        .map(|pred| *label_mapping.get(pred).unwrap_or(&most_common_global_label))
        .collect();

    aligned_preds
}

pub fn acc_score<T>(y_pred: &[T], y_true: &[T]) -> f64
where
    T: std::cmp::Eq + std::hash::Hash + Copy,
{
    assert_eq!(y_pred.len(), y_true.len());
    let n = y_true.len();
    if n == 0 {
        return 0.0;
    }

    let aligned_preds = align_clusters(y_pred, y_true);
    let mut correct = 0usize;
    for (pred, true_label) in aligned_preds.iter().zip(y_true.iter()) {
        if pred == true_label {
            correct += 1;
        }
    }
    (correct as f64) / (n as f64)
}

impl<B: Backend> Metric for ClusteringAccuracyMetric<B> {
    type Input = ClusteringMetricInput<B>;

    fn name(&self) -> String {
        "Accuracy".to_string()
    }

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        self.state.update(
            100.0 * acc_score(&input.y_pred(), &input.y_true()),
            input.batch_size(),
            FormatOptions::new(self.name()).unit("%").precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

impl<B: Backend> Numeric for ClusteringAccuracyMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

#[cfg(test)]
mod tests {
    use super::acc_score;

    #[test]
    fn perfect_labelings() {
        assert_eq!(acc_score(&[0, 0, 1, 1], &[0, 0, 1, 1]), 1.0);
        assert_eq!(acc_score(&[0, 1, 2], &[0, 1, 2]), 1.0);
    }

    #[test]
    fn permuted_labelings() {
        // y_pred: [0, 0, 1, 1], y_true: [1, 1, 0, 0] -> mapping 0->1, 1->0. Accuracy should be 1.0
        assert_eq!(acc_score(&[0, 0, 1, 1], &[1, 1, 0, 0]), 1.0);
        // y_pred: [0, 1, 2], y_true: [2, 0, 1] -> mapping 0->2, 1->0, 2->1. Accuracy should be 1.0
        assert_eq!(acc_score(&[0, 1, 2], &[2, 0, 1]), 1.0);
    }

    #[test]
    fn mid_labelings() {
        assert_eq!(acc_score(&[0, 0, 1, 1], &[0, 1, 0, 1]), 0.5);
        assert_eq!(acc_score(&[0, 1, 0, 1], &[0, 0, 1, 1]), 0.5);
    }

    #[test]
    fn single_cluster() {
        assert_eq!(acc_score(&[0, 0, 0, 0], &[0, 0, 0, 0]), 1.0);
        assert_eq!(acc_score(&[0, 0, 0, 0], &[1, 1, 1, 1]), 1.0);
    }

    #[test]
    fn empty_input() {
        assert_eq!(acc_score::<i32>(&[], &[]), 0.0);
    }

    #[test]
    fn one_element_input() {
        assert_eq!(acc_score(&[0], &[0]), 1.0);
        assert_eq!(acc_score(&[0], &[1]), 1.0);
    }

    #[test]
    fn uneven_clusters() {
        // y_pred: [0, 0, 0, 1], y_true: [0, 0, 1, 1]
        // cluster_to_labels: {0: [0, 0, 1], 1: [1]}
        // global_counts: {0: 2, 1: 2}
        // label_mapping: {0: 0, 1: 1}
        // aligned_preds: [0, 0, 0, 1]
        // correct: 3. Accuracy: 3/4 = 0.75
        assert_eq!(acc_score(&[0, 0, 0, 1], &[0, 0, 1, 1]), 0.75);
    }
}
