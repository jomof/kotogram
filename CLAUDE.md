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

## Corpus DB (data/corpus.db)

Stored in Google Cloud Storage (bucket `jomof-public-files`, prefix `kotogram-datasets/corpus/`), **not** in git. The well-known local path is `data/corpus.db`.

### Workflow
```bash
python -m scripts.dataset corpus-download latest   # Pull corpus.db from GCS
# ... curate / crawl / cleanup ...
python -m scripts.label --source-db data/corpus.db  # Labels & stamps content hash
python -m scripts.dataset build                     # Checks hash, builds .pt bundle
python -m scripts.dataset upload                    # VACUUM+gzip+push corpus.db & .pt to GCS
# Commit dataset.lock
```

### Consistency guards
- Labeling writes `label_content_hash` into the `metadata` table in corpus.db.
- `dataset build` and `dataset upload` verify corpus.db hasn't changed since labeling.
- `dataset upload` automatically uploads corpus.db (VACUUM + gzip) if not already in GCS.
- `dataset.lock` records `corpus_hash` alongside `dataset_id` and `chive_id`.

### GP info
Labeling writes `gp_info.json` into the label cache so training never needs corpus.db.

## Neural Style Model

Located in `models/style/`. Multi-task transformer for:
- Formality level (VERY_FORMAL → VERY_CASUAL)
- Gender markers (MASCULINE, FEMININE, NEUTRAL)
- Register/dialect (SONKEIGO, KENJOGO, KANSAIBEN, etc.)
- Grammaticality

Training uses Knowledge Component (KC) sparse learning. See `train/` for pipeline details.
