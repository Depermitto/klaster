// Copyright (C) 2025 Piotr Jabłoński
// Extended copyright information can be found in the LICENSE file.

use std::collections::HashSet;

use burn::data::dataloader::batcher::Batcher;
use burn::prelude::{Backend, ElementConversion, Int, Tensor, TensorData};
use derive_new::new;
use serde::{Deserialize, Serialize};

/// Dataset for training and testing. Wraps raw image bytes and labels into train/test splits
/// and exposes helpers to build batches for SDC training.
///
/// # Fields
/// - `train_split`: The training data split,
/// - `test_split`: The testing data split,
/// - `item_dims`: The dimensions of each item in the dataset.
///
/// # See also
/// [`Dataset::mnist`], [`Dataset::unipen`]
#[derive(new, Debug, Clone)]
pub struct Dataset {
    train_split: DatasetSplit,
    test_split: DatasetSplit,
    pub item_dims: [usize; 2],
}

impl Dataset {
    /// Read the [unipen](https://github.com/sueiras/handwritting_characters_database) dataset.
    ///
    /// # Params
    /// - `unipen_path`: Path to the dataset root directory.
    ///
    /// # Errors
    /// Returns an error if a file cannot be read or decoded.
    pub fn unipen(unipen_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut records = Vec::new();
        let mut targets = Vec::new();

        for entry in walkdir::WalkDir::new(unipen_path)
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
                records.extend(img_vec);
                targets.push(label as u8);
            }
        }

        Ok(Self {
            train_split: DatasetSplit::new(records, targets),
            test_split: DatasetSplit::empty(),
            item_dims: [64, 64],
        })
    }

    /// Read the MNIST dataset from a local directory.
    ///
    /// # Params
    /// - `mnist_path`: Path to the MNIST root directory.
    pub fn mnist(mnist_path: &str) -> Self {
        let mnist::Mnist {
            trn_img,
            trn_lbl,
            tst_img,
            tst_lbl,
            ..
        } = mnist::MnistBuilder::new().base_path(mnist_path).finalize();

        Self {
            train_split: DatasetSplit::new(trn_img, trn_lbl),
            test_split: DatasetSplit::new(tst_img, tst_lbl),
            item_dims: [28, 28],
        }
    }
}

/// Container for raw data.
///
/// # Panics
/// Can occur if the `images` buffer does not align with `labels` and `item_dims`.
///
/// # See also
/// [`DatasetSplit::new`], [`DatasetSplit::empty`]
#[derive(Debug, Clone)]
pub struct DatasetSplit {
    images: Vec<u8>,
    labels: Vec<u8>,
}

impl DatasetSplit {
    /// Create a new split from raw image bytes and labels.
    pub fn new(images: impl Into<Vec<u8>>, labels: impl Into<Vec<u8>>) -> Self {
        Self {
            images: images.into(),
            labels: labels.into(),
        }
    }

    /// Create an empty split.
    pub fn empty() -> Self {
        Self {
            images: Vec::new(),
            labels: Vec::new(),
        }
    }
}

impl Dataset {
    fn items(split: &DatasetSplit, dims: [usize; 2]) -> Vec<ItemRaw> {
        let size = dims[0] * dims[1];
        assert_eq!(split.images.len(), split.labels.len() * size);

        let items: Vec<_> = split
            .images
            .chunks_exact(size)
            .zip(&split.labels)
            .map(|(image_bytes, &label)| ItemRaw::new(Vec::from(image_bytes), label))
            .collect();
        items
    }

    /// Get items used in training.
    ///
    /// # Panics
    /// Can occur if the train split data length does not match `item_dims`.
    pub fn train_items(&self) -> Vec<ItemRaw> {
        Self::items(&self.train_split, self.item_dims)
    }

    /// Get items used in validation.
    ///
    /// # Panics
    /// Can occur if the test split data length does not match `item_dims`.
    pub fn test_items(&self) -> Vec<ItemRaw> {
        Self::items(&self.test_split, self.item_dims)
    }

    /// Get the number of unique classes in the training split.
    pub fn n_classes(&self) -> usize {
        self.train_split.labels.iter().collect::<HashSet<_>>().len()
    }

    /// Get this dataset's [`DatasetBatcher`].
    #[must_use]
    pub(crate) fn batcher(&self) -> DatasetBatcher {
        let data = &self.train_split.images;

        let mut sum = 0.0f64;
        let mut sum_squares = 0.0f64;
        let count = data.len() as f64;

        for &pixel in data {
            let pixel_normalized = f64::from(pixel) / 255.0; // NORMALIZE HERE
            sum += pixel_normalized;
            sum_squares += pixel_normalized * pixel_normalized;
        }

        let mean = (sum / count) as f32;
        let variance = ((sum_squares / count) - (sum / count).powi(2)) as f32;
        let std = variance.sqrt();

        dbg!("mean={}, std={}", mean, std);
        DatasetBatcher::new(self.item_dims, mean, std)
    }
}

/// Converts raw CPU data into tensors on the target [`burn::prelude::Device`] and
/// automatically applies normalization using dataset mean and standard deviation.
#[derive(new, Debug, Clone, Copy)]
pub(crate) struct DatasetBatcher {
    dims: [usize; 2],
    mean: f32,
    std: f32,
}

/// Single raw dataset item.
#[derive(new, Deserialize, Serialize, Debug, Clone)]
pub struct ItemRaw {
    pub image_bytes: Vec<u8>,
    pub label: u8,
}

/// Batch of data on a [`burn::prelude::Device`].
///
/// # Data layout
/// - `images`: [batch, channels, height, width]
/// - `targets`: [batch]
#[derive(Clone, Debug)]
pub(crate) struct Batch<B: Backend> {
    pub(crate) images: Tensor<B, 4>,
    pub(crate) targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, ItemRaw, Batch<B>> for DatasetBatcher {
    fn batch(&self, items: Vec<ItemRaw>, device: &B::Device) -> Batch<B> {
        let targets = items
            .iter() // ref
            .map(|item| {
                Tensor::<B, 1, Int>::from_data([i64::from(item.label).elem::<B::IntElem>()], device)
            })
            .collect();

        let images = items
            .into_iter() // own
            .map(|item| TensorData::new(item.image_bytes, self.dims).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 2>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, 1, self.dims[0], self.dims[1]]))
            // Normalize: scale between [0,1] and make the mean=0 and std=1
            .map(|tensor| ((tensor / 255) - self.mean) / self.std)
            .collect();

        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);

        Batch { images, targets }
    }
}
