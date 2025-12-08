# Kotogram

A template for creating dual Python/TypeScript libraries that can be published to both PyPI and npm.

## Overview

This project demonstrates how to maintain a library with identical functionality in both Python and TypeScript, with automated testing and publishing workflows for both ecosystems.

## Project Structure

```
kotogram/
├── kotogram/              # Python package
│   ├── __init__.py       # Package exports and version
│   ├── codec.py          # Abstract Codec interface
│   └── reversing_codec.py # ReversingCodec implementation
├── src/                   # TypeScript source
│   ├── codec.ts          # Codec interface
│   ├── reversing-codec.ts # ReversingCodec implementation
│   └── index.ts          # Package exports
├── tests-py/              # Python tests
│   └── test_reversing_codec.py
├── tests-ts/              # TypeScript tests
│   └── reversing-codec.test.ts
├── .github/workflows/     # CI/CD workflows
│   ├── python_canary.yml      # Python build & test
│   ├── typescript_canary.yml  # TypeScript build & test
│   ├── python_publish.yml     # Publish to PyPI
│   └── typescript_publish.yml # Publish to npm
├── version.txt           # Single source of truth for version
├── publish.sh           # Version bump and publish script
├── pyproject.toml       # Python package configuration
├── package.json         # TypeScript package configuration
└── tsconfig.json        # TypeScript compiler configuration
```

## Example Components

### Codec Interface

The library includes a simple `Codec` interface that defines encoding and decoding operations:

**Python** ([kotogram/codec.py](kotogram/codec.py)):
```python
from abc import ABC, abstractmethod

class Codec(ABC):
    @abstractmethod
    def encode(self, text: str) -> str:
        pass

    @abstractmethod
    def decode(self, text: str) -> str:
        pass
```

**TypeScript** ([src/codec.ts](src/codec.ts)):
```typescript
export interface Codec {
  encode(text: string): string;
  decode(text: string): string;
}
```

### ReversingCodec Implementation

A concrete implementation that reverses strings for both encoding and decoding:

**Python** ([kotogram/reversing_codec.py](kotogram/reversing_codec.py)):
```python
class ReversingCodec(Codec):
    def encode(self, text: str) -> str:
        return text[::-1]

    def decode(self, text: str) -> str:
        return text[::-1]
```

**TypeScript** ([src/reversing-codec.ts](src/reversing-codec.ts)):
```typescript
export class ReversingCodec implements Codec {
  encode(text: string): string {
    return text.split("").reverse().join("");
  }

  decode(text: string): string {
    return text.split("").reverse().join("");
  }
}
```

## Development

### Python Development

```bash
# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/

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
  - Tests on Python 3.8, 3.9, 3.10, 3.11, 3.12
  - Runs linting (black, isort, flake8)
  - Type checking with mypy
  - Builds package and uploads artifacts

- **[.github/workflows/typescript_canary.yml](.github/workflows/typescript_canary.yml)**
  - Tests on Node.js 18, 20, 22
  - Type checking with TypeScript compiler
  - Builds package and uploads artifacts

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

## Configuration Requirements

### PyPI Publishing

To publish to PyPI, configure trusted publishing:

1. Go to PyPI → Your Account → Publishing
2. Add a new publisher with:
   - Repository: `yourusername/kotogram`
   - Workflow: `python_publish.yml`
   - Environment: `pypi`

### npm Publishing

To publish to npm, you need an npm access token:

1. Create an automation token on npmjs.com
2. Add it as a GitHub secret named `NPM_TOKEN`
3. Configure the `npm` environment in your repository settings

## Usage Examples

### Python

```python
from kotogram import ReversingCodec

codec = ReversingCodec()
encoded = codec.encode("hello")  # "olleh"
decoded = codec.decode(encoded)  # "hello"
```

### TypeScript

```typescript
import { ReversingCodec } from 'kotogram';

const codec = new ReversingCodec();
const encoded = codec.encode("hello");  // "olleh"
const decoded = codec.decode(encoded);  // "hello"
```

## License

MIT

## Contributing

This is a template project. Feel free to fork and adapt it for your own dual-language libraries!
