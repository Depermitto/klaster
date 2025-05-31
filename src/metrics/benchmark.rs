pub fn benchmark<F>(func: F, n: usize) -> Vec<f64>
where
    F: Fn() -> Vec<f64>,
{
    use ndarray::{Array2, Axis};

    let sample = func();
    let res_len = sample.len();

    let mut data = Array2::<f64>::zeros((n, res_len + 1));
    for i in 0..n {
        let start = std::time::Instant::now();
        let res = func();
        for (j, &val) in res.iter().enumerate() {
            data[[i, j + 1]] = val;
        }
        data[[i, 0]] = start.elapsed().as_secs_f64();
    }
    data.mean_axis(Axis(0)).unwrap().to_vec()
}
