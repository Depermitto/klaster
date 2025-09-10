mod clustering;
mod loss;
mod mnist_data;
mod model;
mod training;
mod cdist;

pub use mnist_data::MnistBatcher;
pub use model::SDCConfig;
pub use training::{TrainingConfig, train};
