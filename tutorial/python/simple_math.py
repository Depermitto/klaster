"""
A simple math module providing basic mathematical operations.

This module includes functions for computing Fibonacci numbers, factorials,
exponentiation, and checking for prime numbers. It is designed to contain a bunch
of wikipedia definitions and have enough text to generate seemingly exhaustive documentation.

Examples:
    >>> fibonacci(10)
    55
    >>> factorial(5)
    120
    >>> fast_pow(3, 16)
    43046721
    >>> is_prime(29)
    True
"""


def fibonacci(n: int) -> int:
    """
    Computes the nth number in the Fibonacci sequence.

    The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones,
    starting from 0 and 1. The sequence begins as follows:

    > 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, ...

    Args:
        n: Position in the Fibonacci sequence to compute.

    Examples:
        >>> fibonacci(0)
        0
        >>> fibonacci(1)
        1
        >>> fibonacci(10)
        55
    """
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def factorial(n: int) -> int:
    """
    Computes the factorial of a given number.

    The factorial of a non-negative integer `n` is the product of all positive integers less than or equal to `n`.
    It is denoted by `n!`. For example:

    > 5! = 5 * 4 * 3 * 2 * 1 = 120

    Args:
        n: The number to compute the factorial for. Must be a non-negative integer.

    Examples:
        >>> factorial(0)
        1
        >>> factorial(5)
        120
    """
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def fast_pow(n: int, power: int) -> int:
    """
    Computes the power of a number raised to a given exponent by squaring.

    Exponentiating by squaring is a general method for fast computation of
    large positive integer powers of a number. Some variants are commonly
    referred to as square-and-multiply algorithms or binary exponentiation.

    Args:
        n: The base number.
        power: The exponent to raise the base to.

    Examples:
        >>> fast_pow(2, 3)
        8
        >>> fast_pow(5, 2)
        25
        >>> fast_pow(47, 0)
        1
        >>> fast_pow(17, 5)
        1419857
    """
    if power == 0:
        return 1
    elif power % 2 == 0:
        return fast_pow(n * n, power // 2)
    else:
        return n * fast_pow(n * n, (power - 1) // 2)


def is_prime(n: int) -> bool:
    """
    Checks if a given number is prime.

    A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.
    This function checks for primality by testing divisibility from 2 up to `n-1`.

    Args:
        n: The number to check for primality.

    Examples:
        >>> is_prime(2)
        True
        >>> is_prime(4)
        False
        >>> is_prime(29)
        True
    """
    if n <= 1:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True


class Complex:
    """
    Represents a complex number type. Implements basic addition, multiplication and division operations.

    Attributes:
        real: The real part of the complex number.
        imag: The imaginary part of the complex number.
    """

    def __init__(self, real: float, imag: float):
        self.real = real
        self.imag = imag

    def __eq__(self, other: "Complex") -> bool:
        return self.real == other.real and self.imag == other.imag

    def __repr__(self) -> str:
        return f"Complex(real={self.real}, imag={self.imag})"

    def add(self, rhs: "Complex") -> "Complex":
        """
        Adds two complex numbers.

        Examples:
            >>> a = Complex(1.0, 2.0)
            >>> b = Complex(3.0, 4.0)
            >>> a.add(b)
            Complex(real=4.0, imag=6.0)
        """
        return Complex(self.real + rhs.real, self.imag + rhs.imag)

    def mul(self, rhs: "Complex") -> "Complex":
        """
        Multiplies two complex numbers.

        Examples:
            >>> a = Complex(1.0, 2.0)
            >>> b = Complex(3.0, 4.0)
            >>> a.mul(b)
            Complex(real=-5.0, imag=10.0)
        """
        real = self.real * rhs.real - self.imag * rhs.imag
        imag = self.real * rhs.imag + self.imag * rhs.real
        return Complex(real, imag)

    def div(self, rhs: "Complex") -> "Complex | None":
        """
        Divides two complex numbers.
        Returns None if division by zero occurs.

        Examples:
            >>> a = Complex(1.0, 2.0)
            >>> b = Complex(3.0, 4.0)
            >>> a.div(b)
            Complex(real=0.44, imag=0.08)
        """
        denom = rhs.real * rhs.real + rhs.imag * rhs.imag
        if denom == 0.0:
            return None

        real = (self.real * rhs.real + self.imag * rhs.imag) / denom
        imag = (self.imag * rhs.real - self.real * rhs.imag) / denom
        return Complex(real, imag)
