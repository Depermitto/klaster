use crate::sdc::metric::ClusteringInput;
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

impl<B: Backend> Metric for ClusteringAccuracyMetric<B> {
    type Input = ClusteringInput<B>;

    fn name(&self) -> String {
        "Accuracy".to_string()
    }

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let batch_size = input.batch_size();
        let y_pred = input.outputs_pred();
        let y_true = input.outputs_true();

        // Build cluster-to-true-label mapping
        let mut cluster_to_labels: HashMap<i64, Vec<i64>> = HashMap::new();
        for (&pred, &true_label) in y_pred.iter().zip(y_true.iter()) {
            cluster_to_labels.entry(pred).or_default().push(true_label);
        }

        // Determine most common global label
        let mut global_counts: HashMap<i64, usize> = HashMap::new();
        for &label in &y_true {
            *global_counts.entry(label).or_insert(0) += 1;
        }
        let most_common_global_label = *global_counts.iter().max_by_key(|e| e.1).unwrap().0;

        // Create cluster -> majority label map
        let mut label_mapping: HashMap<i64, i64> = HashMap::new();
        for (&cluster, labels) in &cluster_to_labels {
            let mut counts = HashMap::new();
            for &label in labels {
                *counts.entry(label).or_insert(0) += 1;
            }
            let majority_label = *counts.iter().max_by_key(|e| e.1).unwrap().0;
            label_mapping.insert(cluster, majority_label);
        }

        // Align predicted labels
        let aligned_preds: Vec<i64> = y_pred
            .iter()
            .map(|pred| *label_mapping.get(pred).unwrap_or(&most_common_global_label))
            .collect();

        let mut correct = 0usize;
        for (pred, true_label) in aligned_preds.iter().zip(y_true.iter()) {
            if pred == true_label {
                correct += 1;
            }
        }
        let accuracy = (correct as f64) / (batch_size as f64);

        self.state.update(
            100.0 * accuracy,
            batch_size,
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
