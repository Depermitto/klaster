use std::collections::HashSet;

use burn::optim::AdamConfig;
use klaster::sdc::{dataset::DatasetSplit, *};

const DATASET_DIR: &str = "datasets";
const MNIST_TRN_LEN: u32 = 5_000;
const MNIST_WIDTH: usize = 28;
const MNIST_HEIGHT: usize = 28;
const UNIPEN_WIDTH: usize = 64;
const UNIPEN_HEIGHT: usize = 64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // let mnist::Mnist {
    //     trn_img,
    //     trn_lbl,
    //     tst_img,
    //     tst_lbl,
    //     ..
    // } = mnist::MnistBuilder::new()
    //     .base_path(format!("{DATASET_DIR}/MNIST/raw/").as_str())
    //     .training_set_length(MNIST_TRN_LEN)
    //     .finalize();
    // let n_classes = 10;

    let unipen_dir = format!("{DATASET_DIR}/UNIPEN-64x64-grayscale");
    let mut records = Vec::new();
    let mut targets = Vec::new();

    for entry in walkdir::WalkDir::new(unipen_dir)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.is_file()
            && let Some(label) = path
                .parent()
                .and_then(|parent| parent.file_name())
                .and_then(|label_str| label_str.to_str().and_then(|s| s.parse::<usize>().ok()))
        {
            let img = image::ImageReader::open(path)?.decode()?.to_luma8();
            let img_vec = img.into_raw();
            records.extend(img_vec.iter().map(|&x| x as u8));
            targets.push(label as u8);
        }
    }
    let n_classes = targets.iter().collect::<HashSet<_>>().len();
    assert_eq!(n_classes, 93);

    let device = &Default::default();
    let latent_dim = 32;
    train::<burn::backend::Autodiff<burn::backend::Wgpu>>(
        "/tmp/sdc",
        TrainingConfig::new(
            SDCConfig::new(n_classes, latent_dim).with_alpha(1.05),
            AutoencoderConfig::new(latent_dim, [UNIPEN_HEIGHT, UNIPEN_WIDTH], [1, 32, 64], 8),
            AdamConfig::new(),
        )
        .with_lr(0.00183)
        .with_batch_size(16)
        .with_num_workers(6),
        Dataset::new(
            DatasetSplit::new(records, targets),
            DatasetSplit::empty(),
            // DatasetSplit::new(trn_img, trn_lbl),
            // DatasetSplit::new(tst_img, tst_lbl),
            [UNIPEN_HEIGHT, UNIPEN_WIDTH],
        ),
        device,
    );

    Ok(())
}
