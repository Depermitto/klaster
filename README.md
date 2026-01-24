# Klaster

*Klaster* is a clustering library for the Rust programming language built as part of an engineering thesis at [Warsaw University of Technology](https://eng.pw.edu.pl/). It focuses on two complementary approaches: a fast, classical K-Means implementation for low-dimensional/tabular data, and an original deep-clustering model SDC designed for image datasets representing monochrome symbol images such as handwritten digits. The goal is to narrow the gap between Rust and the more mature Python ecosystem by providing well-documented clustering tools in a single, cohesive library.

## Getting Started

```shell
cargo add klaster
```

### For developers

Clone the repository and build the project:

```sh
git clone https://github.com/Depermitto/klaster
cd klaster
cargo test
```

## Symbol Deep Clustering

SDC is deep-clustering model tuned for monochrome images of symbols (e.g. handwritten characters/digits). It uses a convolutional autoencoder and jointly optimizes the latent representation with clustering assignments, inspired by DEC/IDEC and DCEC. The reconstruction loss is adapted with focal loss to emphasize hard-to-reconstruct strokes over background.

**MNIST** (10 classes)

| Method | Accuracy | ARI | NMI |
| --- | --- | --- | --- |
| K-Means (raw pixels) | ~0.59 | ~0.37 | ~0.50 |
| SDC (after joint optimization) | 82.9% - 84.0% | 0.48 - 0.53 | 0.825 - 0.842 |

**UNIPEN** (93 classes)

| Method | Accuracy | ARI | NMI |
| --- | --- | --- | --- |
| K-Means (raw pixels) | ~0.21 | ~0.06 | ~0.32 |
| SDC | 67.9% - 68.4% | 0.101 - 0.111 | 0.873 - 0.875 |

Note: See thesis chapter 4 for full experimental context.

## Using the library

### Example 1: Training and running SDC

The SDC example trains a model and then performs inference on a shuffled batch of items.

```rust
// See examples/sdc.rs for the full runnable example.
use klaster::*;

let dataset_name = /* provided via cmdline argument `dataset` */;
let dataset_path = /* provided via cmdline argument `dataset-path`*/;

let (dataset, latent_dim) = match dataset_name.as_str() {
    "mnist" => (Dataset::mnist(dataset_path), 10),
    "unipen" => (Dataset::unipen(dataset_path)?, 128),
    _ => unreachable!(),
};

let artifact_dir = "/tmp/sdc";
let device = &Default::default();
train::<burn::backend::Autodiff<burn::backend::Vulkan>>(
    artifact_dir,
    TrainingConfig::new(
        SDCConfig::new(dataset.n_classes(), latent_dim).with_alpha(1.05),
        AutoencoderConfig::new(latent_dim, dataset.item_dims, [1, 32, 64], 8),
        burn::optim::AdamConfig::new(),
    )
    .with_num_epochs(10)
    .with_lr(0.00183)
    .with_batch_size(16),
    &dataset,
    device,
);

infer::<burn::backend::Vulkan>(artifact_dir, &dataset, device, dataset.test_items());
Ok(())
```

Run the example for MNIST:

```sh
cargo run --example sdc -- --dataset mnist --dataset-path /path/to/mnist
```

or for UNIPEN:

```sh
cargo run --example sdc -- --dataset unipen --dataset-path /path/to/unipen
```

This example uses the Vulkan backend in training and inference. See [burn](https://github.com/tracel-ai/burn)'s GitHub page or the [Burn Book](https://burn.dev/books/burn) for all supported backends.

### KMeans

*Klaster* bundles a complete implementation of the K-Means clustering algorithm, that targets performance (albeit still lacking behind *scikit-learn*) while preserving the standard algorithmic behavior and quality of results. It parallelizes the assignment step across CPU cores using `rayon`, keeps data in `ndarray` arrays and exposes both K-Means++ and Forgy initialization via a builder-style API. Distance computations are optimized by precomputing sample norms and relying on dot products. Extra care has been put into making the public API as easy to use as possible.

**Experiment results**

| Dataset | Implementation | Accuracy | ARI | NMI | Time |
| --- | --- | --- | --- | --- | --- |
| BCW | sklearn | 0.854 | 0.491 | 0.279 | 3.734 ms |
| BCW | linfa | 0.854 | 0.491 | 0.467 | 2.915 ms |
| BCW | klaster | 0.854 | 0.491 | 0.467 | 0.313 ms |
| Red Wine | sklearn | 0.483 | -0.002 | 0.053 | 6.289 ms |
| Red Wine | linfa | 0.482 ± 0.002 | -0.001 ± 0.002 | 0.038 ± 0.003 | 19.596 ms |
| Red Wine | klaster | 0.484 ± 0.004 | -0.003 ± 0.003 | 0.039 ± 0.003 | 1.708 ms |
| MNIST | sklearn | 0.589 | 0.369 | 1.119 | 2.948 s |
| MNIST | linfa | 0.585 ± 0.005 | 0.362 ± 0.002 | 0.495 ± 0.004 | 107.391 s |
| MNIST | klaster | 0.593 ± 0.017 | 0.375 ± 0.021 | 0.495 ± 0.015 | 5.614 s |
| UNIPEN | sklearn | 0.211 | 0.063 | 1.367 | 56.797 s |
| UNIPEN | linfa | 0.216 ± 0.003 | 0.068 ± 0.006 | 0.323 ± 0.004 | 5445.459 s |
| UNIPEN | klaster | 0.210 ± 0.005 | 0.064 ± 0.001 | 0.318 ± 0.001 | 470.3 s |

### Example 2: Running KMeans

```rust
// See examples/sdc.rs for the full runnable example.
use klaster::*;

let expected_centroids = ndarray::array![[-1., 1., 1.], [8., 2., 2.]];
let k_clusters = expected_centroids.nrows();

let mut rng = ndarray_rand::rand::thread_rng();
let data = linfa_datasets::generate::blobs(300, &expected_centroids, &mut rng);

let model_fitted = KMeans::new_plusplus(k_clusters)
    .with_max_iter(100)
    .with_tolerance(1e-6)
    .fit(&data);
println!("{:?}", model_fitted.centroids());
```

Run the example:

```sh
cargo run --example kmeans
```

## Documentation

- API docs: <https://docs.rs/klaster>
- See `src/kmeans/` for detailed K-Means documentation.
- See `src/sdc/` for SDC architecture, configuration, and training helpers.
- See `src/sdc/metric/` for Accuracy, NMI, ARI metrics.

## Contributing

Contributions are welcome! Please open issues or pull requests.

## License

Copyright (C) 2025 Piotr Jabłoński, Institute of Computer Science - Warsaw University of Technology

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
