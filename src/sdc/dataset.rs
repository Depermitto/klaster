use burn::data::dataloader::batcher::Batcher;
use burn::prelude::{Backend, ElementConversion, Int, Tensor, TensorData};
use derive_new::new;
use serde::{Deserialize, Serialize};

#[derive(new, Debug, Clone)]
pub struct Dataset {
    train_split: DatasetSplit,
    test_split: DatasetSplit,
    item_dims: [usize; 2],
}

#[derive(Debug, Clone)]
pub struct DatasetSplit {
    images: Vec<u8>,
    labels: Vec<u8>,
}

impl DatasetSplit {
    pub fn new(images: impl Into<Vec<u8>>, labels: impl Into<Vec<u8>>) -> Self {
        Self {
            images: images.into(),
            labels: labels.into(),
        }
    }

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

    pub(crate) fn train_items(&self) -> Vec<ItemRaw> {
        Self::items(&self.train_split, self.item_dims)
    }

    pub(crate) fn test_items(&self) -> Vec<ItemRaw> {
        Self::items(&self.test_split, self.item_dims)
    }

    #[must_use]
    pub fn batcher(&self) -> DatasetBatcher {
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

#[derive(new, Debug, Clone, Copy)]
pub struct DatasetBatcher {
    dims: [usize; 2],
    mean: f32,
    std: f32,
}

#[derive(new, Deserialize, Serialize, Debug, Clone)]
pub(crate) struct ItemRaw {
    image_bytes: Vec<u8>,
    label: u8,
}

#[derive(Clone, Debug)]
pub(crate) struct Batch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
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
