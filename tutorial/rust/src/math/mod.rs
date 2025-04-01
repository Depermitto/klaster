//! A simple math module providing basic mathematical operations.
//!
//! This module includes functions for computing Fibonacci numbers, factorials,
//! exponentiation, and checking for prime numbers. It is designed to contain a bunch
//! of wikipedia definitions and have enough text to generate seemingly exhaustive documentation.
//!
//! # Examples
//!
//! ```
//! use example::math::*;
//!
//! assert_eq!(fibonacci(10), 55);
//! assert_eq!(factorial(5), 120);
//! assert_eq!(fast_pow(3, 16), 43046721);
//! assert_eq!(is_prime(29), true);
//! ```

/// Computes the nth number in the Fibonacci sequence.
///
/// The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones,
/// starting from 0 and 1. The sequence begins as follows:
///
/// > 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, ...
///
/// # Arguments
///
/// * `n` - Position in the Fibonacci sequence to compute.
///
/// # Examples
///
/// ```
/// use example::math::fibonacci;
///
/// assert_eq!(fibonacci(0), 0);
/// assert_eq!(fibonacci(1), 1);
/// assert_eq!(fibonacci(10), 55);
/// ```
pub fn fibonacci(n: u32) -> u32 {
    let mut a = 0;
    let mut b = 1;
    for _ in 0..n {
        let tmp = b;
        b = a + b;
        a = tmp;
    }
    a
}

/// Computes the factorial of a given number.
///
/// The factorial of a non-negative integer `n` is the product of all positive integers less than or equal to `n`.
/// It is denoted by `n!`. For example:
///
/// > 5! = 5 * 4 * 3 * 2 * 1 = 120
///
/// # Arguments
///
/// * `n` - The number to compute the factorial for. Must be a non-negative integer.
///
/// # Examples
///
/// ```
/// use example::math::factorial;
///
/// assert_eq!(factorial(0), 1);
/// assert_eq!(factorial(5), 120);
/// ```
pub fn factorial(n: u64) -> u64 {
    (2..=n).fold(1, |acc, e| acc * e)
}

/// Computes the power of a number raised to a given exponent by squaring.
///
/// Exponentiating by squaring is a general method for fast computation of
/// large positive integer powers of a number. Some variants are commonly
/// referred to as square-and-multiply algorithms or binary exponentiation.
///
/// # Arguments
///
/// * `n` - The base number.
/// * `power` - The exponent to raise the base to.
///
/// # Examples
///
/// ```
/// use example::math::fast_pow;
///
/// assert_eq!(fast_pow(2, 3), 8);
/// assert_eq!(fast_pow(5, 2), 25);
/// assert_eq!(fast_pow(47, 0), 1);
/// assert_eq!(fast_pow(17, 5), 1419857);
/// ```
pub fn fast_pow(n: u64, power: u64) -> u64 {
    if power == 0 {
        1
    } else if power % 2 == 0 {
        fast_pow(n * n, power / 2)
    } else {
        n * fast_pow(n * n, (power - 1) / 2)
    }
}

/// Checks if a given number is prime.
///
/// A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.
/// This function checks for primality by testing divisibility from 2 up to `n-1`.
///
/// # Arguments
///
/// * `n` - The number to check for primality.
///
/// # Examples
///
/// ```
/// use example::math::is_prime;
///
/// assert_eq!(is_prime(2), true);
/// assert_eq!(is_prime(4), false);
/// assert_eq!(is_prime(29), true);
/// ```
pub fn is_prime(n: u32) -> bool {
    for i in 2..n {
        if n % i == 0 {
            return false;
        }
    }
    true
}

/// Represents a complex number type. Implements basic addition, multiplication and division operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub real: f64,
    pub imag: f64,
}

impl Complex {
    /// Adds two complex numbers.
    /// # Example
    /// ```
    /// use example::math::Complex;
    ///
    /// let a = Complex { real: 1.0, imag: 2.0 };
    /// let b = Complex { real: 3.0, imag: 4.0 };
    ///
    /// let sum = a.add(b);
    /// assert_eq!(sum, Complex { real: 4.0, imag: 6.0 });
    /// ```
    pub fn add(self, rhs: Complex) -> Complex {
        Complex {
            real: self.real + rhs.real,
            imag: self.imag + rhs.imag,
        }
    }

    /// Multiplies two complex numbers.
    /// # Example
    /// ```
    /// use example::math::Complex;
    ///
    /// let a = Complex { real: 1.0, imag: 2.0 };
    /// let b = Complex { real: 3.0, imag: 4.0 };
    ///
    /// let product = a.mul(b);
    /// assert_eq!(product, Complex { real: -5.0, imag: 10.0 });
    /// ```
    pub fn mul(self, rhs: Complex) -> Complex {
        Complex {
            real: self.real * rhs.real - self.imag * rhs.imag,
            imag: self.real * rhs.imag + self.imag * rhs.real,
        }
    }

    /// Divides two complex numbers.
    /// Returns `None` if division by zero occurs.
    /// # Example
    /// ```
    /// use example::math::Complex;
    ///
    /// let a = Complex { real: 1.0, imag: 2.0 };
    /// let b = Complex { real: 3.0, imag: 4.0 };
    /// let quotient = a.div(b).unwrap();
    /// assert_eq!(quotient, Complex { real: 0.44, imag: 0.08 });
    /// ```
    pub fn div(self, rhs: Complex) -> Option<Complex> {
        let denom = rhs.real * rhs.real + rhs.imag * rhs.imag;
        if denom == 0.0 {
            return None;
        }

        Some(Complex {
            real: (self.real * rhs.real + self.imag * rhs.imag) / denom,
            imag: (self.imag * rhs.real - self.real * rhs.imag) / denom,
        })
    }
}
