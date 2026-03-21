# Kotogram

Dual Python/TypeScript library for Japanese text parsing and encoding using a compact kotogram format. Includes a neural style classifier for formality, gender, grammaticality, and register detection.

## Architecture

- **`src/`** - TypeScript source (kotogram utilities, analysis enums)
- **`kotogram/`** - Python source (parser, model, analysis, CLI)
- **`train/`** - Neural model training pipeline
- **`tests-ts/`** - TypeScript tests
- **`tests-py/`** - Python tests (80+ files)
- **`models/style/`** - Pre-trained style classification model

### Kotogram Format

Compact encoding using Unicode markers:
- `⌈⌉` - Token boundaries
- `ˢ` - Surface form, `ᵖ` - POS/features, `ᵇ` - base orthography, `ᵈ` - lemma, `ʳ` - reading

Example: `⌈ˢ猫ᵖnounᵇ猫ʳネコ⌉⌈ˢをᵖprtʳヲ⌉`

## Key Commands

### TypeScript
```bash
npm run build       # Compile to dist/
npm run test        # Compile and run tests
npm run lint        # GTS linter
npm run fix         # Auto-fix lint issues
```

### Python
```bash
pytest tests-py/                    # Run all Python tests
pytest tests-py/ -x                 # Stop on first failure
python -m kotogram                  # CLI entry point
```

## Cross-Language Parity

Python and TypeScript implementations must stay in sync. `TOKEN_SHORTHANDS` compression maps and token parsing logic must match exactly between `src/kotogram.ts` and `kotogram/kotogram.py`.

## Dependencies

- **Python**: `sudachipy` (Japanese morphological analysis), `torch` (neural models), `mlflow` (experiment tracking)
- **TypeScript**: `gts` (Google style linter), Node.js built-in test runner

Install Python deps: `pip install -r requirements.txt`
Install TS deps: `npm install`

## Neural Style Model

Located in `models/style/`. Multi-task transformer for:
- Formality level (VERY_FORMAL → VERY_CASUAL)
- Gender markers (MASCULINE, FEMININE, NEUTRAL)
- Register/dialect (SONKEIGO, KENJOGO, KANSAIBEN, etc.)
- Grammaticality

Training uses Knowledge Component (KC) sparse learning. See `train/` for pipeline details.
