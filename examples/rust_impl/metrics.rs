// Copyright (C) 2025 Piotr Jabłoński
// Extended copyright information can be found in the LICENSE file.

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
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

        writeln!(f, "[")?;

        writeln!(
            f,
            "\t\x1b[1m{:<10}\x1b[0m {:>col_w$}",
            "elapsed:",
            format!("{:.3?}", self.elapsed),
            col_w = col_w
        )?;

        writeln!(
            f,
            "\t\x1b[1m{:<10}\x1b[0m {:>col_w$}",
            "runs:",
            self.runs,
            col_w = col_w
        )?;

        write!(f, "\t\x1b[1m{:<10}\x1b[0m", "mean:")?;
        for m in &self.mean {
            write!(f, " {:>col_w$.3}", m, col_w = col_w)?;
        }
        writeln!(f)?;

        write!(f, "\t\x1b[1m{:<10}\x1b[0m", "std:")?;
        for s in &self.std {
            write!(f, " {:>col_w$.3}", s, col_w = col_w)?;
        }
        writeln!(f)?;

        writeln!(f, "]")?;
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
        runs,
        elapsed: Duration::from_nanos(timings.mean().unwrap_or(0)),
        mean: data
            .mean_axis(Axis(0))
            .unwrap_or(Array1::default(runs))
            .to_vec(),
        std: data.std_axis(Axis(0), 1.0).to_vec(),
    }
}
