# Generacja dokumentacji z kodu dla projektów w Rust

Piotr Jabłoński <piotr.jablonski10.stud@pw.edu.pl>

Poniższa instrukcja przedstawia sposób dokumentowania kodu w Rust, umożliwiający generowanie dokumentacji w postaci plików HTML. Wykorzystuje się do tego ekosystem Cargo. Większość omawianych elementów jest na tyle uniwersalna, że występuje również w innych językach i zachowuje się podobnie.

Ogólny poradnik budowania dokumentacji znajduje się pod adresem <https://doc.rust-lang.org/rust-by-example/meta/doc.html>

## Format komentarzy

Podstawową komendą do wygenerowania dokumentacji jest komenda `cargo doc`. Program `cargo` buduje dokumentację całego projektu na podstawie komentarzy w kodzie. Z każdego modułu tworzona jest sekcja, która zawiera dokumentację funkcji i struktur. Aby dokumentacja została poprawnie wygenerowana należy się stosować do odpowiedniego formatu komentarzy:

### Komentarze dla funkcji i struktur muszą się zaczynać `///`

```rust
/// Represents a complex number type. Implements basic addition, multiplication and division operations.
pub struct Complex {
   pub real: f64,
   pub imag: f64,
}

/// Adds number a to b and returns the result.
pub fn add(a: i32, b: i32) -> i32 { a + b }
```

Dokument HTML uwzględni tylko publiczne funkcje i struktury (w tym też ich pola), dlatego bardzo ważne jest pamiętanie o dodaniu słowa kluczowego `pub`.

### Komentarze dla modułów muszą zaczynać się od `//!`

```rust
//! Contains struct and function definitions for simple mathematical operations.
//!
//! Examples: ...
//! Limitations: ...
```

Dodanie ich jest opcjonalne, lecz zalecane.

### Komentarze wspierają język **Markdown**

```rust
/// # Heading 1
/// ## Heading 2
/// ...
/// **bold text**
/// *italics*
/// ...
/// <google.com>
/// [Link](adres.do.strony)
/// ...
```

### W komentarzach można umieszczać przykłady użycia

```rust
/// # Examples
/// ```
/// let a = Complex { real: 12.0, imag: 16.0 };
/// let b = add(a.real as i32, 8);
/// ```
```

#### Jeżeli w przykładach użycia użyjemy asercji, to `cargo` pozwoli uruchomić je jako test jednostkowy

```rust
/// Adds two complex numbers.
/// # Example
/// ```
/// use example::math::Complex; // nie wolno zapomnieć o imporcie
///
/// let a = Complex { real: 1.0, imag: 2.0 };
/// let b = Complex { real: 3.0, imag: 4.0 };
///
/// let sum = a.add(b);
/// assert_eq!(sum, Complex { real: 4.0, imag: 6.0 });
/// ```
```

Wtedy kiedy wywołamy komendę do uruchomienia testów,

```sh
cargo test

...
test src/math/mod.rs - math::Complex::add (line 139) ... ok
test src/math/mod.rs - math::Complex::mul (line 157) ... ok
test src/math/mod.rs - math::Complex::div (line 176) ... ok
...
```

Nasze przykłady użycia zostaną zarejestrowane jako testy jednostkowe.

## Generowanie dokumentu

Zakładając odpowiedni format komentarzy, aby zbudować dokument HTML wystarczy uruchomić polecenie

```sh
cargo doc --open
```

Opcja `--open` uruchomi przegląrkę internetową na stronie naszej dokumentacji

![wygenerowana dokumentacja z projektu przykładowego](images/rust-docs.png)

[Więcej informacji](https://doc.rust-lang.org/rust-by-example/meta/doc.html#doc-comments)
