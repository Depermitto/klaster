# Generowanie dokumentacji z kodu

Piotr Jabłoński <piotr.jablonski10.stud@pw.edu.pl>

Poniższa instrukcja przedstawia sposób konfigurowania środowiska do pracy z MkDocs oraz sposoby dokumentowania kodu w językach Rust/Python, umożliwiając generowanie dokumentacji w postaci plików HTML. Wykorzystuje do tego ekosystemy *cargo* oraz *mkdocstrings*. Większość omawianych elementów jest na tyle uniwersalna, że występuje również w innych językach i zachowuje się podobnie.

## Generowanie dokumentacji przy pomocy narzędzia MkDocs

Ogólny poradnik dotyczący pracy z *mkdocs* znajduje się pod adresem:
<https://mkdocs.org/getting-started/>,
a dokumentacja *mkdocstrings* dostępna jest pod:
<https://mkdocstrings.github.io/usage/>

### Omówienie narzędzia

[MkDocs](https://mkdocs.org/) to prosty generator statycznych stron przeznaczony do tworzenia dokumentacji. Pliki dokumentacyjne zapisywane są w formacie Markdown i przetwarzane na pliki HTML. Proces jest konfigurowany poprzez plik **mkdocs.yml**. Do generowania dokumentacji bezpośrednio z kodu źródłowego wykorzystywane jest narzędzie *mkdocstrings*, a do pracy z językiem Python - *mkdocstrings-python*.

Obecnie obsługiwanych jest wiele języków programowania, w tym:

- [Python](https://mkdocstrings.github.io/python/)
- [C](https://mkdocstrings.github.io/c/)
- [Crystal](https://mkdocstrings.github.io/crystal/)
- [TypeScript](https://mkdocstrings.github.io/typescript/)

### Konfiguracja MkDocs

> Wymagana jest instalacja MkDocs przez `pip` w środowisku globalnym/wirtualnym (`pip install mkdocs`)

Aby rozpocząć nowy projekt, należy wykonać następujące polecenia:

```shell
mkdocs new my-project
cd my-project
```

Struktura wygenerowanego projektu:

```shell
.
├── docs/             # folder z dokumentacją
│   └── index.md      # główna strona dokumentacji
├── mkdocs.yml        # plik konfiguracyjny
└── site/             # wygenerowana statyczna strona HTML
    ├── 404.html
    ├── assets/
    ├── index.html
    └── ...
```

#### Konfiguracja pliku mkdocs.yml

Plik konfiguracyjny wykorzystuje format YAML. Podstawowe sekcje do skonfigurowania:

- `site_name` - nazwa strony
- `theme` - motyw graficzny (domyślnie dostępne: *mkdocs* i *readthedocs*)
- `nav` - struktura nawigacji na bocznym pasku

Przykładowa konfiguracja początkowa:

```yaml
site_name: Generate Documentation from Code
use_directory_urls: false  # ułatwia przeglądanie dokumentacji bez serwera HTTP

theme:
  name: mkdocs

nav:
  - Home: index.md
```

Aby wygenerować dokumentację, należy wykonać polecenie `mkdocs build` w katalogu projektu. Wygenerowana strona HTML znajdzie się w folderze *site/*.

#### Dodawanie nowych stron

Aby dodać nową stronę:

1. Utwórz plik Markdown w folderze *docs/*
2. Dodaj odnośnik w sekcji *nav* pliku *mkdocs.yml*

Przykład:

```yaml
nav:
  - Home: index.md
  - Document like a snake: howtopython.md
  - Document like a crab: howtorust.md
  - Writing mkdocs: howtomkdocs.md
```

### Praca z językiem Python

MkDocs nie obsługuje natywnie dokumentacji z plików Python (.py). Wymagana jest integracja z *mkdocstrings*.

> **Instalacja**: `pip install mkdocstrings[python]`

Konfiguracja odbywa się w sekcji *plugins*:

```yaml
plugins:
  - mkdocstrings:
      default_handler: python
      handlers:
        python:
          options:
            show_source: true  # pokazuje kod źródłowy
            heading_level: 2   # poziom nagłówków
```

Następnie należy dodać referencję do dokumentowanego modułu w pliku Markdown:

```markdown
# Python Documentation

::: path.to.module
    options:
      show_submodules: true  # uwzględnia podmoduły (domyślnie tylko __init__.py)
```

Zachowuje on się jak zwykła strona, z tym że w miejsce `:::` "wklejona" zostanie dokumentacja kodu. Sekcja mkdocstring musi zawierać identyfikator modułu z opcjonalną konfiguracją handlera.

### Integracja z Rust

MkDocs nie posiada natywnej obsługi języka Rust. Można jednak:

1. Wygenerować dokumentację za pomocą `cargo doc`
2. Dodać link do wygenerowanej dokumentacji:

```markdown
[Rust API Reference](file:///path/to/rust/doc/index.html)
```

Lub w pliku konfiguracyjnym:

```yaml
nav:
  - Rust API Reference: /path/to/rust/doc/index.html
```

### Pełny przykład projektu

Jako przykład użycia przedstawię strukturę **tego projektu**:

```shell
.
├── docs/
│   ├── api.md
│   ├── howtomkdocs.md
│   ├── howtopython.md
│   ├── howtorust.md
│   └── index.md
├── mkdocs.yml
├── python/
│   ├── __init__.py
│   └── simple_math.py
├── rust/
│   ├── Cargo.toml
│   └── src/
└── site/
    └── ...
```

Plik konfiguracyjny:

```yaml
site_name: Generate Documentation from Code
use_directory_urls: false

theme:
  name: material
  palette:
    - scheme: default
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      toggle:
        icon: material/brightness-4
        name: Switch to light mode

plugins:
  - mkdocstrings:
      default_handler: python
      handlers:
        python:
          options:
            show_source: true
            heading_level: 2

nav:
  - Home: index.md
  - Python API: api.md
  - Python Guide: howtopython.md
  - Rust Guide: howtorust.md
  - MkDocs Guide: howtomkdocs.md
```

## Generacja dokumentacji z kodu dla projektów w Rust

### Format komentarzy

Podstawową komendą do wygenerowania dokumentacji jest komenda `cargo doc`. Program `cargo` buduje dokumentację całego projektu na podstawie komentarzy w kodzie. Dokumentacja jest generowana w postaci plików HTML w folderze `${PROJ_DIR}/target/doc/${CRATE_NAME}/index.html`. Z każdego modułu tworzona jest sekcja, która zawiera dokumentację funkcji i struktur.

#### Ważne uwagi przed rozpoczęciem

1. Dokumentacja generuje się tylko dla publicznych elementów (oznaczonych `pub`)
2. Komentarze dokumentacyjne powinny wyjaśniać **co** robi funkcja/struktura, a nie **jak** to robi
3. Dobre praktyki zalecają umieszczanie przykładów użycia

#### Komentarze dla funkcji i struktur muszą się zaczynać `///`

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

#### Komentarze dla modułów muszą zaczynać się od `//!`

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

#### Formatowanie z Markdown

W komentarzach można używać:

- Nagłówków (`#`, `##` etc.)
- Pogrubienia (`**text**`) i kursywy (`*text*`)
- List (`- item` lub `1. item`)
- Linków (`[text](url)` lub `<url>`)
- Bloków kodu (``````code``````)

#### Przykłady użycia i testy

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

### Generowanie dokumentacji

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

### Dokumentacja dla interfejsów

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

### Przydatne tagi sekcji

Oprócz `# Examples` można używać:

- `# Panics` - kiedy funkcja może spanikować
- `# Notes` - dodatkowe uwagi
- `# Warning` - ostrzeżenia

### Linkowanie między elementami

Można linkować do innych elementów dokumentacji używając `[`Type`]`:

```rust
/// This function returns [`Complex`] number
/// See also: [`add`] function
```

[Więcej informacji](https://doc.rust-lang.org/rustdoc/index.html)

## Generacja dokumentacji z kodu dla projektów w Pythonie

### Format komentarzy

Podstawowym formatem dokumentacji w Pythonie są docstringi zgodne z konwencją PEP 257. mkdocstrings obsługuje zarówno format Google Style, NumPy Style jak i reStructuredText.

#### Ważne uwagi przed rozpoczęciem

1. Dokumentacja generuje się dla wszystkich elementów, chyba że zostaną oznaczone jako prywatne (przez prefixy `_` i `__`)
2. Docstringi powinny wyjaśniać **co** robi funkcja/klasa, a nie **jak** to robi
3. Dobre praktyki zalecają umieszczanie przykładów użycia

#### Docstringi dla funkcji i klas

```python
class Complex:
    """Represents a complex number type. Implements basic arithmetic operations.

    Attributes:
        real (float): Real part of complex number
        imag (float): Imaginary part of complex number
    """

    def __init__(self, real: float, imag: float):
        self.real = real
        self.imag = imag

def add(a: int, b: int) -> int:
    """Adds two integers and returns the result.

    Args:
        a: First integer
        b: Second integer

    Returns:
        Sum of a and b
    """
    return a + b
```

#### Docstringi dla modułów

```python
"""Module for simple mathematical operations.

Provides basic implementations for complex numbers and arithmetic operations.

Examples:
    >>> from my_module import Complex
    >>> num = Complex(3.0, 4.0)
"""
```

#### Formatowanie z Markdown

W docstringach można używać:

- Nagłówków (`#`, `##` etc.)
- Pogrubienia (`**text**`) i kursywy (`*text*`)
- List (`- item` lub `1. item`)
- Linków (`[text](url)` lub `<url>`)
- Bloków kodu (``````code``````)

#### Przykłady użycia i testy

Przykłady w dokumentacji mogą służyć jako testy doctest:

```python
def add_complex(a: 'Complex', b: 'Complex') -> 'Complex':
    """Adds two complex numbers.

    Example:
        >>> from my_module import Complex, add_complex
        >>> a = Complex(1.0, 2.0)
        >>> b = Complex(3.0, 4.0)
        >>> sum = add_complex(a, b)
        >>> sum.real
        4.0
        >>> sum.imag
        6.0
    """
    return Complex(a.real + b.real, a.imag + b.imag)
```

Aby uruchomić testy z dokumentacji:

```sh
python -m doctest -v my_module.py
```

### Przydatne sekcje

Oprócz `Examples` można używać:

- `Raises`: kiedy funkcja może rzucić wyjątek
- `Notes`: dodatkowe uwagi
- `Warnings`: ostrzeżenia
- `See Also`: powiązane funkcje

### Linkowanie między elementami

mkdocstrings automatycznie linkuje typy w sygnaturach. Można też ręcznie dodać linki:

```python
def create_complex() -> 'Complex':
    """This function returns a :class:`Complex` number.
    See also: :func:`add` function
    """
```

[Więcej informacji](https://peps.python.org/pep-0257/)
