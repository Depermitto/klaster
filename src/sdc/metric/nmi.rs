use crate::sdc::metric::ClusteringInput;
use burn::prelude::*;
use burn::train::metric::state::{FormatOptions, NumericMetricState};
use burn::train::metric::{Metric, MetricEntry, MetricMetadata, Numeric};
use std::collections::HashMap;

#[derive(Default)]
pub struct NMIMetric<B: Backend> {
    state: NumericMetricState,
    _b: B,
}

impl<B: Backend> NMIMetric<B> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<B: Backend> Metric for NMIMetric<B> {
    type Input = ClusteringInput<B>;

    fn name(&self) -> String {
        "NMI".to_string()
    }

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let batch_size = input.batch_size();
        let y_pred = input.outputs_pred();
        let y_true = input.outputs_true();

        let n = y_true.len() as f64;

        // Count occurrences
        let mut true_counts = HashMap::new();
        let mut pred_counts = HashMap::new();
        let mut joint_counts = HashMap::new();

        for (&true_label, &pred_label) in y_true.iter().zip(y_pred.iter()) {
            *true_counts.entry(true_label).or_insert(0.0) += 1.0;
            *pred_counts.entry(pred_label).or_insert(0.0) += 1.0;
            *joint_counts.entry((true_label, pred_label)).or_insert(0.0) += 1.0;
        }

        // Compute entropy H(Y)
        let h_true = true_counts
            .values()
            .map(|&count| {
                let p = count / n;
                -p * p.log2()
            })
            .sum::<f64>();

        // Compute entropy H(C)
        let h_pred = pred_counts
            .values()
            .map(|&count| {
                let p = count / n;
                -p * p.log2()
            })
            .sum::<f64>();

        // Compute Mutual Information MI(Y, C)
        let mi = joint_counts
            .iter()
            .map(|(&(y, c), &joint_count)| {
                let p_joint = joint_count / n;
                let p_y = true_counts[&y] / n;
                let p_c = pred_counts[&c] / n;
                p_joint * (p_joint / (p_y * p_c)).log2()
            })
            .sum::<f64>();

        let nmi_score = {
            if h_true == 0.0 || h_pred == 0.0 {
                0.0
            } else {
                // Normalize
                mi / ((h_true * h_pred).sqrt())
            }
        };

        self.state.update(
            nmi_score,
            batch_size,
            FormatOptions::new("NMI").precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

impl<B: Backend> Numeric for NMIMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}
