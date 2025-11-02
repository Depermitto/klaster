use burn::optim::AdamConfig;
use klaster::sdc::*;

const DATASET_DIR: &str = "/home/dev-main/datasets";
const TRN_LEN: u32 = 5_000;
const WIDTH: usize = 28;
const HEIGHT: usize = 28;

fn main() {
    let mnist::Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = mnist::MnistBuilder::new()
        .base_path(format!("{DATASET_DIR}/MNIST/raw/").as_str())
        .training_set_length(TRN_LEN)
        .finalize();

    let device = &Default::default();
    let latent_dim = 10;
    train::<burn::backend::Autodiff<burn_ndarray::NdArray>>(
        "/tmp/sdc",
        TrainingConfig::new(
            SDCConfig::new(10, latent_dim).with_alpha(1.05),
            AutoencoderConfig::new(latent_dim),
            Dataset::new(
                dataset::Split::new(trn_img, trn_lbl),
                dataset::Split::new(tst_img, tst_lbl),
                dataset::Dims::new(WIDTH, HEIGHT),
            ),
            AdamConfig::new(),
        )
        .with_lr(0.00183)
        .with_batch_size(8)
        .with_num_workers(6),
        device,
    );
}
