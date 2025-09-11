use burn::optim::AdamConfig;
use klaster::sdc::*;

fn main() {
    let device = Default::default();
    println!("{:?}", device);
    let artifact_dir = "/tmp/sdc";
    train::<burn::backend::Autodiff<burn::backend::Vulkan>>(
        artifact_dir,
        TrainingConfig::new(
            SDCConfig::new(10, 8),
            AutoencoderConfig::new(8),
            AdamConfig::new(),
        ),
        device,
    );
}
