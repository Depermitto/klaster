# Generacja dokumentacji z kodu dla projektów w Rust

Piotr Jabłoński <piotr.jablonski10.stud@pw.edu.pl>

Poniższa instrukcja przedstawia sposób dokumentowania kodu w Rust, umożliwiający generowanie dokumentacji w postaci plików HTML. Wykorzystuje się do tego ekosystem Cargo. Większość omawianych elementów jest na tyle uniwersalna, że występuje również w innych językach i zachowuje się podobnie.

Ogólny poradnik budowania dokumentacji znajduje się pod adresem <https://doc.rust-lang.org/rust-by-example/meta/doc.html>

## Format komentarzy

Podstawową komendą do wygenerowania dokumentacji jest komenda `cargo doc`. Program `cargo` buduje dokumentację całego projektu na podstawie komentarzy w kodzie. Dokumentacja jest generowana w postaci plików HTML w folderze `${PROJ_DIR}/target/doc/${CRATE_NAME}/index.html`. Z każdego modułu tworzona jest sekcja, która zawiera dokumentację funkcji i struktur.

### Ważne uwagi przed rozpoczęciem

1. Dokumentacja generuje się tylko dla publicznych elementów (oznaczonych `pub`)
2. Komentarze dokumentacyjne powinny wyjaśniać **co** robi funkcja/struktura, a nie **jak** to robi
3. Dobre praktyki zalecają umieszczanie przykładów użycia

### Komentarze dla funkcji i struktur muszą się zaczynać `///`

```rust
/// Represents a complex number type. Implements basic addition, multiplication and division operations.
///
/// # Fields
/// - `real` - real part of complex number
/// - `imag` - imaginary part of complex number
pub struct Complex {
   pub real: f64,
   pub imag: f64,
}

/// Adds two integers and returns the result.
///
/// # Arguments
/// * `a` - First integer
/// * `b` - Second integer
///
/// # Returns
/// Sum of a and b
pub fn add(a: i32, b: i32) -> i32 { a + b }
```

### Komentarze dla modułów muszą zaczynać się od `//!`

```rust
//! Module for simple mathematical operations.
//!
//! Provides basic implementations for complex numbers and arithmetic operations.
//!
//! # Examples
//! ```
//! use my_crate::math::Complex;
//! let num = Complex { real: 3.0, imag: 4.0 };
//! ```
```

### Formatowanie z Markdown

W komentarzach można używać:

- Nagłówków (`#`, `##` etc.)
- Pogrubienia (`**text**`) i kursywy (`*text*`)
- List (`- item` lub `1. item`)
- Linków (`[text](url)` lub `<url>`)
- Bloków kodu (``````code``````)

### Przykłady użycia i testy

Przykłady w dokumentacji mogą służyć jako testy:

```rust
/// Adds two complex numbers.
///
/// # Example
/// ```
/// use my_crate::math::Complex;
///
/// let a = Complex { real: 1.0, imag: 2.0 };
/// let b = Complex { real: 3.0, imag: 4.0 };
///
/// let sum = a.add(b);
/// assert_eq!(sum.real, 4.0);
/// assert_eq!(sum.imag, 6.0);
/// ```
pub fn add(&self, other: Complex) -> Complex {
    Complex {
        real: self.real + other.real,
        imag: self.imag + other.imag,
    }
}
```

Aby uruchomić testy z dokumentacji:

```sh
cargo test --doc
```

## Generowanie dokumentacji

Podstawowe komendy:

```sh
cargo doc

cargo doc --open

cargo doc --all # generuje dokumentację ze zależnościami
```

Dodatkowe opcje:

- `--no-deps` - tylko dla bieżącego projektu
- `--document-private-items` - uwzględnia również niepubliczne elementy
- `--release` - tryb na produkcję

## Dokumentacja dla interfejsów

Przykład dokumentowania interfejsu w Rust (`trait`):

```rust
/// Trait for mathematical operations
pub trait MathOps {
    /// Adds two values of implementing type
    fn add(&self, other: Self) -> Self;

    /// Multiplies two values
    fn multiply(&self, other: Self) -> Self;
}
```

## Przydatne tagi sekcji

Oprócz `# Examples` można używać:

- `# Panics` - kiedy funkcja może spanikować
- `# Notes` - dodatkowe uwagi
- `# Warning` - ostrzeżenia

## Linkowanie między elementami

Można linkować do innych elementów dokumentacji używając `[`Type`]`:

```rust
/// This function returns [`Complex`] number
/// See also: [`add`] function
```

[Więcej informacji](https://doc.rust-lang.org/rustdoc/index.html)
