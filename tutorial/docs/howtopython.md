# Generacja dokumentacji z kodu dla projektów w Pythonie

Piotr Jabłoński <piotr.jablonski10.stud@pw.edu.pl>

Poniższa instrukcja przedstawia sposób dokumentowania kodu w Pythonie, umożliwiający generowanie dokumentacji w postaci plików HTML przy użyciu ekosystemu mkdocstrings. Większość omawianych elementów jest na tyle uniwersalna, że występuje również w innych językach i zachowuje się podobnie.

## Format komentarzy

Podstawowym formatem dokumentacji w Pythonie są docstringi zgodne z konwencją PEP 257. mkdocstrings obsługuje zarówno format Google Style, NumPy Style jak i reStructuredText.

### Ważne uwagi przed rozpoczęciem

1. Dokumentacja generuje się dla wszystkich elementów, chyba że zostaną oznaczone jako prywatne (przez prefixy `_` i `__`)
2. Docstringi powinny wyjaśniać **co** robi funkcja/klasa, a nie **jak** to robi
3. Dobre praktyki zalecają umieszczanie przykładów użycia

### Docstringi dla funkcji i klas

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

### Docstringi dla modułów

```python
"""Module for simple mathematical operations.

Provides basic implementations for complex numbers and arithmetic operations.

Examples:
    >>> from my_module import Complex
    >>> num = Complex(3.0, 4.0)
"""
```

### Formatowanie z Markdown

W docstringach można używać:

- Nagłówków (`#`, `##` etc.)
- Pogrubienia (`**text**`) i kursywy (`*text*`)
- List (`- item` lub `1. item`)
- Linków (`[text](url)` lub `<url>`)
- Bloków kodu (``````code``````)

### Przykłady użycia i testy

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

## Przydatne sekcje

Oprócz `Examples` można używać:

- `Raises`: kiedy funkcja może rzucić wyjątek
- `Notes`: dodatkowe uwagi
- `Warnings`: ostrzeżenia
- `See Also`: powiązane funkcje

## Linkowanie między elementami

mkdocstrings automatycznie linkuje typy w sygnaturach. Można też ręcznie dodać linki:

```python
def create_complex() -> 'Complex':
    """This function returns a :class:`Complex` number.
    See also: :func:`add` function
    """
```

[Więcej informacji](https://peps.python.org/pep-0257/)
