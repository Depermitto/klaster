pub mod hdbscan;
pub mod kmeans;
pub mod metrics;
pub mod n2d;

#[cfg(test)]
mod tests {
    #[test]
    fn sanity() {
        assert_eq!(2 + 2, 4);
    }
}
