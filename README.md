# Kotogram

[![Python Canary](https://github.com/jomof/kotogram/actions/workflows/python_canary.yml/badge.svg?branch=main)](https://github.com/jomof/kotogram/actions/workflows/python_canary.yml)
[![TypeScript Canary](https://github.com/jomof/kotogram/actions/workflows/typescript_canary.yml/badge.svg?branch=main)](https://github.com/jomof/kotogram/actions/workflows/typescript_canary.yml)
[![PyPI Version](https://img.shields.io/pypi/v/kotogram.svg)](https://pypi.org/project/kotogram/)
[![npm Version](https://img.shields.io/npm/v/kotogram.svg)](https://www.npmjs.com/package/kotogram)
[![Python Support](https://img.shields.io/pypi/pyversions/kotogram.svg)](https://pypi.org/project/kotogram/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A dual Python/TypeScript library for Japanese text parsing and encoding using the kotogram compact format.

## Overview

Kotogram provides tools for parsing Japanese text into a compact, linguistically-rich format that encodes part-of-speech, conjugation, and pronunciation information. The library features:

- **Abstract parser interface** (`JapaneseParser`) for backend implementations
- **Sudachi implementation** (`SudachiJapaneseParser`) using SudachiPy with full dictionary
- **Kotogram format** - compact representation preserving linguistic features
- **Bidirectional conversion** between Japanese text and kotogram format
- **Dual-language support** - Python and TypeScript implementations (TypeScript coming soon)
- **Production-quality CI/CD** with comprehensive testing and publishing workflows

## Project Structure

```
kotogram/
├── kotogram/                    # Python package
│   ├── __init__.py             # Package exports and version
│   ├── japanese_parser.py      # Abstract JapaneseParser interface
│   └── sudachi_japanese_parser.py # Sudachi implementation
├── src/                         # TypeScript source
│   ├── kotogram.ts             # Kotogram conversion functions
│   └── index.ts                # Package exports
├── tests-py/                    # Python tests
│   └── test_japanese_parser.py # Japanese parser tests
├── tests-ts/                    # TypeScript tests
│   └── kotogram.test.ts
├── .github/workflows/           # CI/CD workflows
│   ├── python_canary.yml       # Python build & test
│   ├── typescript_canary.yml   # TypeScript build & test
│   ├── python_publish.yml      # Publish to PyPI
│   └── typescript_publish.yml  # Publish to npm
├── version.txt                  # Single source of truth for version
├── publish.sh                  # Version bump and publish script
├── pyproject.toml              # Python package configuration
├── package.json                # TypeScript package configuration
└── tsconfig.json               # TypeScript compiler configuration
```

## Quick Start

### Japanese Text Parsing

Parse Japanese text into kotogram format with full linguistic information:

**Python**:
```python
from kotogram import SudachiJapaneseParser, kotogram_to_japanese

# Initialize parser (requires sudachipy and sudachidict_full)
parser = SudachiJapaneseParser(dict_type='full')

# Convert Japanese to kotogram
japanese = "猫を食べる"
kotogram = parser.japanese_to_kotogram(japanese)
# Result: ⌈ˢ猫ᵖn:common_noun⌉⌈ˢをᵖprt:case_particle⌉⌈ˢ食べるᵖv:general:e-ichidan-ba:terminal⌉

# Convert back to Japanese
reconstructed = kotogram_to_japanese(kotogram)
# Result: "猫を食べる"

# With spaces between tokens
spaced = kotogram_to_japanese(kotogram, spaces=True)
# Result: "猫 を 食べる"

# With furigana (IME-style readings in brackets)
with_furigana = kotogram_to_japanese(kotogram, furigana=True)
# Result: "猫[ねこ]を食べる[たべる]"

# Combine options
spaced_furigana = kotogram_to_japanese(kotogram, spaces=True, furigana=True)
# Result: "猫[ねこ] を 食べる[たべる]"
```

**TypeScript**:
```typescript
import { kotogramToJapanese, splitKotogram } from 'kotogram';

// Convert Japanese to kotogram (requires Python parser)
const kotogram = "⌈ˢ猫ᵖn:common_noun⌉⌈ˢをᵖprt:case_particle⌉⌈ˢ食べるᵖv:general:e-ichidan-ba:terminal⌉";

// Convert back to Japanese
const reconstructed = kotogramToJapanese(kotogram);
// Result: "猫を食べる"

// With spaces between tokens
const spaced = kotogramToJapanese(kotogram, { spaces: true });
// Result: "猫 を 食べる"

// With furigana (IME-style readings in brackets)
const withFurigana = kotogramToJapanese(kotogram, { furigana: true });
// Result: "猫[ねこ]を食べる[たべる]"

// Split into tokens
const tokens = splitKotogram(kotogram);
// Result: ["⌈ˢ猫ᵖn:common_noun⌉", "⌈ˢをᵖprt:case_particle⌉", "⌈ˢ食べるᵖv:general:e-ichidan-ba:terminal⌉"]
```

### Kotogram Format

The kotogram format encodes rich linguistic information in a compact representation:

```
⌈ˢ食べるᵖv:general:e-ichidan-ba:terminalᵇ食べるᵈ食べるʳタベル⌉
  │  │    │ │       │            │         │      │      │
  │  │    │ │       │            │         │      │      └─ pronunciation (ʳ)
  │  │    │ │       │            │         │      └─ lemma (ᵈ)
  │  │    │ │       │            │         └─ base form (ᵇ)
  │  │    │ │       │            └─ conjugation form
  │  │    │ │       └─ conjugation type
  │  │    │ └─ POS detail
  │  │    └─ part-of-speech (ᵖ)
  │  └─ surface form (ˢ)
  └─ token boundary markers (⌈⌉)
```

## Development

### Python Development

```bash
# Install in development mode
pip install -e .

# Run tests
python -m pytest tests-py/

# Run type checking
mypy kotogram/

# Build package
python -m build
```

### TypeScript Development

```bash
# Install dependencies
npm install

# Build
npm run build

# Run tests
npm test

# Type check
npx tsc --noEmit
```

## Testing

### Python Tests

Tests are located in [tests-py/](tests-py/) and use the `unittest` framework. They are also compatible with `pytest`.

Run tests:
```bash
python -m unittest discover -s tests-py -p 'test_*.py' -v
# or
python -m pytest tests-py/ -v
```

### TypeScript Tests

Tests are located in [tests-ts/](tests-ts/) and use Node.js built-in test runner.

Run tests:
```bash
npm test
```

## GitHub Workflows

### Canary Builds

These workflows run on every push, pull request, and daily at 2 AM UTC:

- **[.github/workflows/python_canary.yml](.github/workflows/python_canary.yml)**
  - **Testing**: Runs on Python 3.8, 3.9, 3.10, 3.11, 3.12 with unittest and pytest
  - **Code Coverage**: Tracks test coverage and uploads to Codecov
  - **Code Quality**:
    - Black for code formatting
    - isort for import sorting
    - flake8 for linting (complexity limit: 10)
    - pylint for advanced code quality (minimum score: 8.0)
    - mypy for strict type checking
  - **Security**:
    - bandit for security vulnerability scanning
    - safety for dependency vulnerability checks
  - **Best Practices**:
    - Checks for print() statements (should use logging)
    - Detects TODO/FIXME comments
    - Validates README.md and LICENSE files exist
  - **Package Validation**:
    - Ensures no TypeScript/JavaScript files leak into Python package
    - Verifies package contents and structure

- **[.github/workflows/typescript_canary.yml](.github/workflows/typescript_canary.yml)**
  - **Testing**: Runs on Node.js 18, 20, 22
  - **Type Checking**: Strict TypeScript type checking with --noEmit
  - **Code Quality**:
    - ESLint for linting (if configured)
    - Prettier for code formatting (if configured)
    - Circular dependency detection with madge
  - **Performance**:
    - Bundle size analysis (warns if >100KB)
  - **Security**:
    - npm audit for dependency vulnerabilities
  - **Best Practices**:
    - Checks for console.log() statements
    - Detects TODO/FIXME comments
    - Warns about `any` types (encourages type safety)
    - Validates package.json metadata (description, keywords, repository, license)
    - Validates README.md and LICENSE files exist
  - **Package Validation**:
    - Ensures no Python files leak into TypeScript package
    - Verifies dist/ directory contents

### Publishing Workflows

These workflows are triggered when a version tag (e.g., `v0.0.1`) is pushed:

- **[.github/workflows/python_publish.yml](.github/workflows/python_publish.yml)**
  - Verifies version consistency across [version.txt](version.txt), [kotogram/__init__.py](kotogram/__init__.py), and [pyproject.toml](pyproject.toml)
  - Builds and publishes to PyPI using trusted publishing
  - Verifies installation from PyPI

- **[.github/workflows/typescript_publish.yml](.github/workflows/typescript_publish.yml)**
  - Verifies version consistency across [version.txt](version.txt) and [package.json](package.json)
  - Builds and publishes to npm with provenance
  - Verifies installation from npm

## Version Management

### Single Source of Truth

The file [version.txt](version.txt) contains the current version number (e.g., `0.0.1`). This version must be kept in sync across:
- [version.txt](version.txt)
- [kotogram/__init__.py](kotogram/__init__.py) (`__version__` variable)
- [pyproject.toml](pyproject.toml) (`version` field)
- [package.json](package.json) (`version` field)

The publish workflows automatically verify this consistency before publishing.

### Publishing a New Version

Use the [publish.sh](publish.sh) script to bump the version and trigger publication:

```bash
# Bump patch version (0.0.1 -> 0.0.2)
./publish.sh patch

# Bump minor version (0.0.1 -> 0.1.0)
./publish.sh minor

# Bump major version (0.0.1 -> 1.0.0)
./publish.sh major
```

The script will:
1. Increment the version number
2. Update all version files
3. Commit the changes
4. Create a git tag (e.g., `v0.0.2`)
5. Push the commit and tag to GitHub

This triggers both [python_publish.yml](.github/workflows/python_publish.yml) and [typescript_publish.yml](.github/workflows/typescript_publish.yml) workflows.

## Badges

The README includes status badges for build status, package versions, and license:

```markdown
[![Python Canary](https://github.com/jomof/kotogram/actions/workflows/python_canary.yml/badge.svg?branch=main)](https://github.com/jomof/kotogram/actions/workflows/python_canary.yml)
[![TypeScript Canary](https://github.com/jomof/kotogram/actions/workflows/typescript_canary.yml/badge.svg?branch=main)](https://github.com/jomof/kotogram/actions/workflows/typescript_canary.yml)
[![PyPI Version](https://img.shields.io/pypi/v/kotogram.svg)](https://pypi.org/project/kotogram/)
[![npm Version](https://img.shields.io/npm/v/kotogram.svg)](https://www.npmjs.com/package/kotogram)
[![Python Support](https://img.shields.io/pypi/pyversions/kotogram.svg)](https://pypi.org/project/kotogram/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
```

**Note**: Update the username in badge URLs if you fork this to your own repository.

## Configuration Requirements

### PyPI Publishing

To publish to PyPI, configure trusted publishing:

1. Go to PyPI → Your Account → Publishing
2. Add a new publisher with:
   - Repository: `jomof/kotogram`
   - Workflow: `python_publish.yml`
   - Environment: `pypi`

### npm Publishing

To publish to npm, you need an npm access token:

1. Create an automation token on npmjs.com
2. Add it as a GitHub secret named `NPM_TOKEN`
3. Configure the `npm` environment in your repository settings

## API Reference

### JapaneseParser (Abstract Base Class)

Abstract interface for Japanese text parsing implementations.

```python
from kotogram import JapaneseParser

class JapaneseParser(ABC):
    @abstractmethod
    def japanese_to_kotogram(self, text: str) -> str:
        """Convert Japanese text to kotogram compact representation."""
        pass
```

### SudachiJapaneseParser

Sudachi-based implementation using SudachiPy with the full dictionary.

```python
from kotogram import SudachiJapaneseParser

# Initialize with full dictionary (recommended)
parser = SudachiJapaneseParser(dict_type='full')

# Or use smaller dictionaries for faster loading
parser_small = SudachiJapaneseParser(dict_type='small')
parser_core = SudachiJapaneseParser(dict_type='core')

# Enable validation mode for debugging unmapped features
parser_strict = SudachiJapaneseParser(dict_type='full', validate=True)
# This will raise descriptive KeyError if any Sudachi features
# are missing from the mapping dictionaries

# Parse Japanese text
kotogram = parser.japanese_to_kotogram("今日は良い天気です")
```

**Parameters**:
- `dict_type` (default: `'full'`): Dictionary type to use ('small', 'core', or 'full')
- `validate` (default: `False`): When `True`, raises descriptive `KeyError` exceptions when encountering unmapped linguistic features. The error message includes:
  - The name of the mapping dictionary (e.g., `POS_MAP`, `CONJUGATED_TYPE_MAP`)
  - The unmapped key value

**Validation Mode Example**:
```python
# With validate=True, unmapped features raise detailed errors
parser = SudachiJapaneseParser(dict_type='full', validate=True)
try:
    kotogram = parser.japanese_to_kotogram("未知の単語")
except KeyError as e:
    # Error message: "Missing mapping in POS_MAP: key='未知品詞' not found."
    print(f"Unmapped feature detected: {e}")
```

### Helper Functions

```python
from kotogram import kotogram_to_japanese, split_kotogram

# Convert kotogram back to Japanese
japanese = kotogram_to_japanese(kotogram_str)
japanese_with_spaces = kotogram_to_japanese(kotogram_str, spaces=True)

# Split kotogram into individual tokens
tokens = split_kotogram(kotogram_str)
```

### Mapping Constants

Global mapping constants are available in `japanese_parser` module:

```python
from kotogram.japanese_parser import (
    POS_MAP,              # Part-of-speech mappings
    POS1_MAP,             # POS detail level 1
    POS2_MAP,             # POS detail level 2
    CONJUGATED_TYPE_MAP,  # Conjugation type mappings
    CONJUGATED_FORM_MAP,  # Conjugation form mappings
    POS_TO_CHARS,         # POS to character mappings
    CHAR_TO_POS,          # Character to POS mappings
)
```

## License

MIT

## Contributing

This is a template project. Feel free to fork and adapt it for your own dual-language libraries!
