use ndarray::{ArrayView, Dimension, Zip};

#[derive(Clone, Copy)]
pub enum KMeansDistFn {
    Euclidean,
    EuclideanSquared,
    Manhattan,
    Chebyshev,
    Cosine,
}

impl KMeansDistFn {
    pub fn run<D>(&self, a: ArrayView<f64, D>, b: ArrayView<f64, D>) -> f64
    where
        D: Dimension,
    {
        match self {
            KMeansDistFn::Euclidean => Zip::from(a)
                .and(b)
                .fold(0.0, |acc, x, y| acc + (x - y).powi(2))
                .sqrt(),
            KMeansDistFn::EuclideanSquared => Zip::from(a)
                .and(b)
                .fold(0.0, |acc, x, y| acc + (x - y).powi(2)),
            KMeansDistFn::Manhattan => Zip::from(a)
                .and(b)
                .fold(0.0, |acc, x, y| acc + (x - y).abs()),
            KMeansDistFn::Chebyshev => Zip::from(a)
                .and(b)
                .fold(0.0, |acc, x, y| acc.max((x - y).abs())),
            KMeansDistFn::Cosine => {
                let dot = Zip::from(&a).and(&b).fold(0.0, |acc, x, y| acc + x * y);
                let norm_a = Zip::from(a).fold(0.0, |acc, x| acc + x.powi(2)).sqrt();
                let norm_b = Zip::from(b).fold(0.0, |acc, y| acc + y.powi(2)).sqrt();
                1.0 - (dot / (norm_a * norm_b))
            }
        }
    }
}
