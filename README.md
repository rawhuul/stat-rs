# stat-rs

This library provides a comprehensive implementation for statistical operations on fixed-size numerical series in Rust, designed for both `std` and `no-std` environments. It features functions for various statistical analyses, including descriptive statistics, measures of central tendency, dispersion, and outlier detection.

## Features

- **Fixed-Size Series**: Efficient storage and operations on fixed-length numerical arrays using `ArrayVec`.
- **Descriptive Statistics**: Calculate mean, median, mode, variance, standard deviation, and more.
- **Quartile Analysis**: Functions to compute quartiles, interquartile ranges, and outliers.
- **Statistical Tests**: Methods for correlation, covariance, and other statistical tests.
- **Utility Functions**: Additional functions like normalization, sigmoid, and softmax transformations.

## Usage

Here's a quick example of how to create a series and perform some statistical calculations:

```rust
let data: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
let series = Series::from(data);

// Returns None, only if Series is empty
let mean = series.mean().unwrap();
let median = series.median().unwrap();
let variance = series.variance().unwrap();

println!("Mean: {}", mean);
println!("Median: {}", median);
println!("Variance: {}", variance);
```
