mod clustering;
mod loss;
mod mnist_data;
mod model;
mod training;
mod cdist;
mod autoencoder;

pub use mnist_data::MnistBatcher;
pub use model::SDCConfig;
pub use autoencoder::AutoencoderConfig;
pub use training::{TrainingConfig, train};
