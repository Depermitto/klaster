use crate::sdc::dataset::Batch;
use burn::nn::loss::{MseLoss, Reduction};
use burn::tensor::backend::AutodiffBackend;
use burn::train::{RegressionOutput, TrainOutput, TrainStep, ValidStep};
use burn::{
    nn::{
        GroupNorm, GroupNormConfig, LeakyRelu, Linear, LinearConfig, PaddingConfig2d, Sigmoid,
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Autoencoder<B: Backend> {
    encoder: Encoder<B>,
    decoder: Decoder<B>,
}

#[derive(Module, Debug)]
struct Encoder<B: Backend> {
    conv1: Conv2d<B>,
    norm1: GroupNorm<B>,
    conv2: Conv2d<B>,
    norm2: GroupNorm<B>,
    linear: Linear<B>,
    leaky_relu: LeakyRelu,
}

#[derive(Module, Debug)]
struct Decoder<B: Backend> {
    linear: Linear<B>,
    conv_trans1: ConvTranspose2d<B>,
    norm1: GroupNorm<B>,
    conv_trans2: ConvTranspose2d<B>,
    leaky_relu: LeakyRelu,
    sigmoid: Sigmoid,
}

#[derive(Config, Debug)]
pub struct AutoencoderConfig {
    pub latent_dim: usize,
    #[config(default = "[1, 32, 64]")]
    pub channels: [usize; 3],
    #[config(default = "[7, 7]")]
    pub feature_map_size: [usize; 2],
    #[config(default = "8")]
    pub groups: usize,
    #[config(default = "0.01")]
    pub leaky_relu_slope: f64,
    #[config(default = "[3, 3]")]
    pub kernel_size: [usize; 2],
    #[config(default = "[2, 2]")]
    pub stride: [usize; 2],
    #[config(default = "[1, 1]")]
    pub padding: [usize; 2],
    #[config(default = "[1, 1]")]
    pub output_padding: [usize; 2],
}

impl AutoencoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Autoencoder<B> {
        let [input_ch, hidden_ch, output_ch] = self.channels;
        let [h, w] = self.feature_map_size;
        let flat_features = output_ch * h * w;

        Autoencoder {
            encoder: Encoder {
                conv1: Conv2dConfig::new([input_ch, hidden_ch], self.kernel_size)
                    .with_stride(self.stride)
                    .with_padding(PaddingConfig2d::Explicit(self.padding[0], self.padding[1]))
                    .init(device),
                norm1: GroupNormConfig::new(self.groups, hidden_ch).init(device),
                conv2: Conv2dConfig::new([hidden_ch, output_ch], self.kernel_size)
                    .with_stride(self.stride)
                    .with_padding(PaddingConfig2d::Explicit(self.padding[0], self.padding[1]))
                    .init(device),
                norm2: GroupNormConfig::new(self.groups, output_ch).init(device),
                linear: LinearConfig::new(flat_features, self.latent_dim).init(device),
                leaky_relu: LeakyRelu {
                    negative_slope: self.leaky_relu_slope,
                },
            },
            decoder: Decoder {
                linear: LinearConfig::new(self.latent_dim, flat_features).init(device),
                conv_trans1: ConvTranspose2dConfig::new([output_ch, hidden_ch], self.kernel_size)
                    .with_stride(self.stride)
                    .with_padding(self.padding)
                    .with_padding_out(self.output_padding)
                    .init(device),
                norm1: GroupNormConfig::new(self.groups, hidden_ch).init(device),
                conv_trans2: ConvTranspose2dConfig::new([hidden_ch, input_ch], self.kernel_size)
                    .with_stride(self.stride)
                    .with_padding(self.padding)
                    .with_padding_out(self.output_padding)
                    .init(device),
                leaky_relu: LeakyRelu {
                    negative_slope: self.leaky_relu_slope,
                },
                sigmoid: Sigmoid::new(),
            },
        }
    }
}

impl<B: Backend> Autoencoder<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 2>) {
        // Encoder
        let x = self.encoder.conv1.forward(x);
        let x = self.encoder.norm1.forward(x);
        let x = self.encoder.leaky_relu.forward(x);

        let x = self.encoder.conv2.forward(x);
        let x = self.encoder.norm2.forward(x);
        let x = self.encoder.leaky_relu.forward(x);

        let [batch_size, channels, height, width] = x.dims();
        let x = x.reshape([batch_size, channels * height * width]);
        let embeddings = self.encoder.linear.forward(x);

        // Decoder
        let x = self.decoder.linear.forward(embeddings.clone());
        let x = x.reshape([batch_size, channels, height, width]);

        let x = self.decoder.conv_trans1.forward(x);
        let x = self.decoder.norm1.forward(x);
        let x = self.decoder.leaky_relu.forward(x);

        let x = self.decoder.conv_trans2.forward(x);
        let recon = self.decoder.sigmoid.forward(x);

        (recon, embeddings)
    }

    pub fn forward_regression(&self, x: Tensor<B, 4>) -> RegressionOutput<B> {
        let (recon, _) = self.forward(x.clone());
        let loss = MseLoss::new().forward(recon.clone(), x.clone(), Reduction::Mean);

        RegressionOutput {
            loss,
            output: recon.flatten(1, 3),
            targets: x.flatten(1, 3),
        }
    }
}

impl<B: AutodiffBackend> TrainStep<Batch<B>, RegressionOutput<B>> for Autoencoder<B> {
    fn step(&self, batch: Batch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regression(batch.images);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<Batch<B>, RegressionOutput<B>> for Autoencoder<B> {
    fn step(&self, batch: Batch<B>) -> RegressionOutput<B> {
        self.forward_regression(batch.images)
    }
}
