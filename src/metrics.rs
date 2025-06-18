use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::Hash;
use std::{
    fmt::{Display, Formatter, Result},
    time::Duration,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRaport {
    pub elapsed: Duration,
    pub runs: usize,
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

impl Display for BenchmarkRaport {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let col_w = 12;

        writeln!(f, "Raport {{")?;

        writeln!(
            f,
            "\t\x1b[1m{:<10}\x1b[0m {:>col_w$}",
            "Elapsed:",
            format!("{:.3?}", self.elapsed),
            col_w = col_w
        )?;

        writeln!(
            f,
            "\t\x1b[1m{:<10}\x1b[0m {:>col_w$}",
            "Runs:",
            self.runs,
            col_w = col_w
        )?;

        write!(f, "\t\x1b[1m{:<10}\x1b[0m", "Mean:")?;
        for m in &self.mean {
            write!(f, " {:>col_w$.3}", m, col_w = col_w)?;
        }
        writeln!(f)?;

        write!(f, "\t\x1b[1m{:<10}\x1b[0m", "Std:")?;
        for s in &self.std {
            write!(f, " {:>col_w$.3}", s, col_w = col_w)?;
        }
        writeln!(f)?;

        writeln!(f, "}}")?;
        Ok(())
    }
}

pub fn benchmark_runtime<F>(runs: usize, f: F) -> BenchmarkRaport
where
    F: Fn() -> Vec<f64>,
{
    let sample = f();
    let res_len = sample.len();

    let mut timings = Array1::default(runs);
    let mut data = Array2::zeros((runs, res_len));
    for i in 0..runs {
        let start = std::time::Instant::now();
        let res = f();
        for (j, &val) in res.iter().enumerate() {
            data[[i, j]] = val;
        }
        timings[i] = start.elapsed().as_nanos() as u64;
    }

    BenchmarkRaport {
        runs: runs,
        elapsed: Duration::from_nanos(timings.mean().unwrap_or(0)),
        mean: data
            .mean_axis(Axis(0))
            .unwrap_or(Array1::default(runs))
            .to_vec(),
        std: data.std_axis(Axis(0), 1.0).to_vec(),
    }
}

pub fn benefit_of_doubt_acc<T, U>(y_true: &[T], y_pred: &[U]) -> f64
where
    T: Eq + PartialOrd + Hash + Copy,
    U: Eq + PartialOrd + Hash + Copy,
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
