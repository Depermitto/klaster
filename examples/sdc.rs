use burn::optim::AdamConfig;
use klaster::sdc::*;

const DATASET_DIR: &str = "/home/dev-main/datasets";
const SUBSET: usize = 1_000;
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
        .finalize();
    let trn_img = trn_img[..SUBSET * WIDTH * HEIGHT].to_vec();
    let trn_lbl = trn_lbl[..SUBSET].to_vec();
    let tst_img = tst_img[..SUBSET * WIDTH * HEIGHT].to_vec();
    let tst_lbl = tst_lbl[..SUBSET].to_vec();

    let device = dbg!(Default::default());
    train::<burn::backend::Autodiff<burn_ndarray::NdArray>>(
        "/tmp/sdc",
        TrainingConfig::new(
            SDCConfig::new(10, 8).with_alpha(1.05),
            AutoencoderConfig::new(8),
            DatasetConfig::new(
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
