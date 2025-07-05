# Klaster

klaster is a Rust library for efficient and flexible clustering, designed for research and production use, with a focus on performance and extensibility.

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

### Example: Running KMeans

```rust
use klaster::KMeans;
use ndarray::array;

let data = array![[1.0, 2.0], [1.1, 2.1], [8.0, 8.0], [8.1, 8.1]];
let model = KMeans::new_plusplus(2)
    .with_tolerance(1e-4)
    .with_max_iter(100);
let assignments = model.fit_predict(data.view());
println!("Cluster assignments: {:?}", assignments);
```

## Documentation

- API docs: <https://docs.rs/klaster>
- See [src/kmeans/mod.rs](src/kmeans/mod.rs) for detailed documentation and usage examples.

## Contributing

Contributions are welcome! Please open issues or pull requests.

## License

Copyright (C) 2025 Piotr Jabłoński
Distributed under the terms of the MIT license. See [LICENSE](LICENSE)
