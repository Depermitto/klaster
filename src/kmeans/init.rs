use ndarray::{Array2, ArrayView2, Axis};
use ndarray_rand::rand;

#[derive(Clone)]
pub enum KMeansInitFn {
    Forgy,
}

impl KMeansInitFn {
    pub fn run(
        &self,
        k_clusters: usize,
        data: ArrayView2<f64>,
        rng: &mut impl rand::Rng,
    ) -> Array2<f64> {
        match self {
            KMeansInitFn::Forgy => {
                let samples = data.dim().0;
                let indices = rand::seq::index::sample(rng, samples, k_clusters).into_vec();
                data.select(Axis(0), &indices)
            }
        }
    }
}
