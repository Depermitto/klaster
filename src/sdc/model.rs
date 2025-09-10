use crate::sdc::clustering::ClusteringOutput;
use crate::sdc::loss::ClusterLoss;
use crate::sdc::mnist_data::MnistBatch;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Distribution;
use burn::train::{TrainOutput, TrainStep, ValidStep};
use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig}, GroupNorm, GroupNormConfig, LeakyRelu, Linear, LinearConfig, PaddingConfig2d,
        Sigmoid,
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct SDC<B: Backend> {
    autoencoder: Autoencoder<B>,
    centroids: Tensor<B, 2>,
    alpha: f64,
    gamma: f64,
}

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
pub struct SDCConfig {
    pub n_clusters: usize,
    pub latent_dim: usize,

    #[config(default = "1.0")]
    pub alpha: f64,

    #[config(default = "2.0")]
    pub gamma: f64,

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

impl SDCConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SDC<B> {
        let [input_ch, hidden_ch, output_ch] = self.channels;
        let [h, w] = self.feature_map_size;
        let flat_features = output_ch * h * w;

        SDC {
            autoencoder: Autoencoder {
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
                    conv_trans1: ConvTranspose2dConfig::new(
                        [output_ch, hidden_ch],
                        self.kernel_size,
                    )
                    .with_stride(self.stride)
                    .with_padding(self.padding)
                    .with_padding_out(self.output_padding)
                    .init(device),
                    norm1: GroupNormConfig::new(self.groups, hidden_ch).init(device),
                    conv_trans2: ConvTranspose2dConfig::new(
                        [hidden_ch, input_ch],
                        self.kernel_size,
                    )
                    .with_stride(self.stride)
                    .with_padding(self.padding)
                    .with_padding_out(self.output_padding)
                    .init(device),
                    leaky_relu: LeakyRelu {
                        negative_slope: self.leaky_relu_slope,
                    },
                    sigmoid: Sigmoid::new(),
                },
            },
            centroids: Tensor::random(
                [self.n_clusters, self.latent_dim],
                Distribution::Normal(0.0, 0.04),
                device,
            ),
            alpha: self.alpha,
            gamma: self.gamma,
        }
    }
}

impl<B: Backend> SDC<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 2>) {
        // Encoder
        let encoder = &self.autoencoder.encoder;
        let x = encoder.conv1.forward(x);
        let x = encoder.norm1.forward(x);
        let x = encoder.leaky_relu.forward(x);

        let x = encoder.conv2.forward(x);
        let x = encoder.norm2.forward(x);
        let x = encoder.leaky_relu.forward(x);

        let [batch_size, channels, height, width] = x.dims();
        let x = x.reshape([batch_size, channels * height * width]);
        let embeddings = encoder.linear.forward(x);

        // Decoder
        let decoder = &self.autoencoder.decoder;
        let x = decoder.linear.forward(embeddings.clone());
        let x = x.reshape([batch_size, channels, height, width]);

        let x = decoder.conv_trans1.forward(x);
        let x = decoder.norm1.forward(x);
        let x = decoder.leaky_relu.forward(x);

        let x = decoder.conv_trans2.forward(x);
        let recon = decoder.sigmoid.forward(x);

        (recon, embeddings)
    }

    pub fn forward_clustering(
        &self,
        x: Tensor<B, 4>,
        targets: Tensor<B, 1, Int>,
    ) -> ClusteringOutput<B> {
        let (recon, embeddings) = self.forward(x.clone());
        let loss = ClusterLoss::new().forward::<B, 4>(
            x,
            recon,
            embeddings.clone(),
            self.centroids.clone(),
            2.0,
            1.0,
        );
        // let loss = MseLoss::new().forward(x, recon, Reduction::Mean);

        ClusteringOutput {
            centroids: self.centroids.clone(),
            embeddings,
            loss,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClusteringOutput<B>> for SDC<B> {
    fn step(&self, batch: MnistBatch<B>) -> TrainOutput<ClusteringOutput<B>> {
        let item = self.forward_clustering(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MnistBatch<B>, ClusteringOutput<B>> for SDC<B> {
    fn step(&self, batch: MnistBatch<B>) -> ClusteringOutput<B> {
        self.forward_clustering(batch.images, batch.targets)
    }
}
