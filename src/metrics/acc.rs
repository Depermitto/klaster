use std::collections::HashMap;
use std::hash::Hash;

pub fn benefit_of_doubt_acc<T, U>(y_true: &[T], y_pred: &[U]) -> f64
where
    T: Eq + Ord + Hash + Copy,
    U: Eq + Ord + Hash + Copy,
{
    let mut label_mapping = HashMap::new();
    for &cluster in y_pred.iter() {
        let mut label_counts = HashMap::new();
        for (&pred, &true_label) in y_pred.iter().zip(y_true) {
            if pred == cluster {
                *label_counts.entry(true_label).or_insert(0) += 1;
            }
        }
        if let Some((&label, _)) = label_counts.iter().max_by_key(|(_, count)| *count) {
            label_mapping.insert(cluster, label);
        }
    }

    let aligned_labels: Vec<T> = y_pred
        .iter()
        .map(|c| *label_mapping.get(c).unwrap())
        .collect();

    let correct = y_true
        .iter()
        .zip(aligned_labels.iter())
        .filter(|(a, b)| a == b)
        .count();
    correct as f64 / y_true.len() as f64
}
