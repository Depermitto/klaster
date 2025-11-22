use crate::sdc::metric::ClusteringMetricInput;
use burn::prelude::*;
use burn::train::metric::state::{FormatOptions, NumericMetricState};
use burn::train::metric::{Metric, MetricEntry, MetricMetadata, Numeric};
use std::collections::HashMap;

#[derive(Default)]
pub struct ARIMetric<B: Backend> {
    state: NumericMetricState,
    _b: B,
}

impl<B: Backend> ARIMetric<B> {
    pub fn new() -> Self {
        Self::default()
    }
}

pub fn ari_score<T>(y_pred: &[T], y_true: &[T]) -> f64
where
    T: std::cmp::Eq + std::hash::Hash + Copy,
{
    assert_eq!(y_pred.len(), y_true.len());
    let n = y_true.len() as i64;

    fn combinations(n: i64, k: i64) -> i64 {
        if k < 0 || k > n {
            0
        } else if k == 0 || k == n {
            1
        } else if k > n / 2 {
            combinations(n, n - k)
        } else {
            (1..=k).fold(1, |acc, i| acc * (n - i + 1) / i)
        }
    }

    let mut contingency_table = HashMap::new();
    for (true_label, pred_label) in y_true.iter().zip(y_pred.iter()) {
        *contingency_table
            .entry((true_label, pred_label))
            .or_insert(0) += 1;
    }

    let mut a = HashMap::new();
    let mut b = HashMap::new();

    for ((true_label, pred_label), count) in &contingency_table {
        *a.entry(**true_label).or_insert(0) += *count;
        *b.entry(**pred_label).or_insert(0) += *count;
    }

    let sum_nij_choose_2: i64 = contingency_table
        .values()
        .map(|&nij| combinations(nij, 2))
        .sum();

    let sum_a_choose_2: i64 = a.values().map(|&ai| combinations(ai, 2)).sum();
    let sum_b_choose_2: i64 = b.values().map(|&bj| combinations(bj, 2)).sum();

    let n_choose_2 = combinations(n, 2);
    if n_choose_2 == 0 {
        return 0.0;
    }

    let index = sum_nij_choose_2 as f64;
    let expected_index = (sum_a_choose_2 as f64 * sum_b_choose_2 as f64) / n_choose_2 as f64;
    let max_index = 0.5 * (sum_a_choose_2 as f64 + sum_b_choose_2 as f64);
    let denominator = max_index - expected_index;

    if denominator == 0.0 {
        0.0
    } else {
        (index - expected_index) / denominator
    }
}

impl<B: Backend> Metric for ARIMetric<B> {
    type Input = ClusteringMetricInput<B>;

    fn name(&self) -> String {
        "ARI".to_string()
    }

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        self.state.update(
            ari_score(&input.y_true(), &input.y_pred()),
            input.batch_size(),
            FormatOptions::new(self.name()).precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

impl<B: Backend> Numeric for ARIMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn perfect_labelings() {
        assert_eq!(ari_score(&[0, 0, 1, 1], &[0, 0, 1, 1]), 1.0);

        assert_eq!(ari_score(&[0, 0, 1, 1], &[1, 1, 0, 0]), 1.0);
    }

    #[test]
    fn totally_incomplete() {
        assert_eq!(ari_score(&[0, 0, 0, 0], &[0, 1, 2, 3]), 0.0);
    }

    #[test]
    fn complete_unpure() {
        assert_abs_diff_eq!(
            ari_score(&[0, 0, 1, 2], &[0, 0, 1, 1]),
            0.57,
            epsilon = 1e-2
        );
        assert_abs_diff_eq!(
            ari_score(&[0, 0, 1, 1], &[0, 0, 1, 2]),
            0.57,
            epsilon = 1e-2
        );
    }

    #[test]
    fn negative_discordant_labelings() {
        assert_abs_diff_eq!(
            ari_score(&[0, 0, 1, 1], &[0, 1, 0, 1]),
            -0.5,
            epsilon = 1e-10
        );
    }
}
