# Generowanie dokumentacji przy pomocy narzędzia MkDocs

Piotr Jabłoński <piotr.jablonski10.stud@pw.edu.pl>

Poniższa instrukcja przedstawia sposób konfigurowania środowiska do pracy z MkDocs. Wykorzystuje się do tego narzędzia *mkdocs* oraz *mkdocstrings*.

Ogólny poradnik dotyczący pracy z *mkdocs* znajduje się pod adresem:  
<https://mkdocs.org/getting-started/>,  
a dokumentacja *mkdocstrings* dostępna jest pod:  
<https://mkdocstrings.github.io/usage/>

## Omówienie narzędzia

[MkDocs](https://mkdocs.org/) to prosty generator statycznych stron przeznaczony do tworzenia dokumentacji. Pliki dokumentacyjne zapisywane są w formacie Markdown i przetwarzane na pliki HTML. Proces jest konfigurowany poprzez plik **mkdocs.yml**. Do generowania dokumentacji bezpośrednio z kodu źródłowego wykorzystywane jest narzędzie *mkdocstrings*, a do pracy z językiem Python - *mkdocstrings-python*. 

Obecnie obsługiwanych jest wiele języków programowania, w tym:
- [Python](https://mkdocstrings.github.io/python/)
- [C](https://mkdocstrings.github.io/c/)
- [Crystal](https://mkdocstrings.github.io/crystal/)
- [TypeScript](https://mkdocstrings.github.io/typescript/)

## Konfiguracja MkDocs

> Wymagana jest instalacja MkDocs przez `pip` w środowisku globalnym/wirtualnym (`pip install mkdocs`)

Aby rozpocząć nowy projekt, należy wykonać następujące polecenia:

```shell
mkdocs new my-project
cd my-project
```

Struktura wygenerowanego projektu:

```
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

### Konfiguracja pliku mkdocs.yml

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

### Dodawanie nowych stron

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

## Praca z językiem Python

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

## Integracja z Rust

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

## Pełny przykład projektu

Jako przykład użycia przedstawię strukturę **tego projektu**:

```
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
