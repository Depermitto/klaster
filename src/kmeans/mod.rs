mod dist;
mod init;

use ndarray::{Array1, Array2, ArrayView2};
use ndarray_rand::rand;

pub use crate::kmeans::{dist::KMeansDistFn, init::KMeansInitFn};

pub struct KMeans {
    k_clusters: usize,
    max_iter: usize,
    tolerance: f64,
    init_fn: KMeansInitFn,
    dist_fn: KMeansDistFn,
}

impl KMeans {
    pub fn new(k_clusters: usize) -> Self {
        Self {
            k_clusters,
            init_fn: KMeansInitFn::Forgy,
            dist_fn: KMeansDistFn::EuclideanSquared,
            tolerance: 1e-4,
            max_iter: 300,
        }
    }

    pub fn init_fn(self, init_fn: KMeansInitFn) -> Self {
        Self { init_fn, ..self }
    }

    pub fn dist_fn(self, dist_fn: KMeansDistFn) -> Self {
        Self { dist_fn, ..self }
    }

    pub fn tolerance(self, tolerance: f64) -> Self {
        Self { tolerance, ..self }
    }

    pub fn max_iter(self, max_iter: usize) -> Self {
        Self { max_iter, ..self }
    }

    pub fn fit(&self, data: ArrayView2<f64>) -> KMeansFitted {
        let mut rng = rand::thread_rng();
        let mut centroids = self.init_fn.run(self.k_clusters, data, &mut rng);
        let mut memberships = Array1::zeros(data.nrows());

        for _ in 0..self.max_iter {
            // Assignment step
            assign_clusters(data, centroids.view(), &mut memberships, self.dist_fn);

            // Calculate new centroids (mean of all points assigned to each cluster)
            let new_centroids = {
                let mut counts = Array1::<f64>::zeros(self.k_clusters);
                let mut new_centroids = Array2::<f64>::zeros((self.k_clusters, data.ncols()));

                for (point, &membership) in data.outer_iter().zip(&memberships) {
                    let mut centroid = new_centroids.row_mut(membership);
                    centroid += &point;
                    counts[membership] += 1.0;
                }

                for (mut new_centroid, count) in new_centroids.outer_iter_mut().zip(counts) {
                    if count > 0.0 {
                        new_centroid /= count;
                    }
                }
                new_centroids
            };

            // Convergence check
            let distance = self.dist_fn.run(centroids.view(), new_centroids.view());
            centroids = new_centroids;
            if distance < self.tolerance {
                break;
            }
        }

        KMeansFitted {
            centroids,
            dist_fn: self.dist_fn,
        }
    }

    pub fn fit_predict(&self, data: ArrayView2<f64>) -> Array1<usize> {
        self.fit(data).predict(data)
    }
}

pub struct KMeansFitted {
    centroids: Array2<f64>,
    dist_fn: KMeansDistFn,
}

impl KMeansFitted {
    pub fn centroids(&self) -> ArrayView2<f64> {
        self.centroids.view()
    }

    pub fn predict_inplace(&self, data: ArrayView2<f64>, memberships: &mut Array1<usize>) {
        assign_clusters(data, self.centroids(), memberships, self.dist_fn);
    }

    pub fn predict(&self, data: ArrayView2<f64>) -> Array1<usize> {
        let mut memberships = Array1::zeros(data.nrows());
        assign_clusters(data, self.centroids(), &mut memberships, self.dist_fn);
        memberships
    }
}

fn assign_clusters(
    data: ArrayView2<f64>,
    centroids: ArrayView2<f64>,
    memberships: &mut Array1<usize>,
    dist_fn: KMeansDistFn,
) {
    for (point, membership) in data.outer_iter().zip(memberships) {
        let cluster_assignment = (0..centroids.nrows()).min_by(|&x, &y| {
            dist_fn
                .run(point, centroids.row(x))
                .partial_cmp(&dist_fn.run(point, centroids.row(y)))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        *membership = cluster_assignment.unwrap();
    }
}
