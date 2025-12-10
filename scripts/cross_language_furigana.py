#!/usr/bin/env python3
"""Cross-language validation for kotogram_to_japanese with furigana.

This script validates that the Python and TypeScript implementations of
kotogram_to_japanese produce identical results when furigana=true.

Process:
1. Read all sentences from jpn_sentences.tsv
2. Convert each to kotogram using both MeCab and Sudachi parsers
3. Write kotograms to temp files
4. Run Python kotogram_to_japanese(furigana=True) on all kotograms
5. Run TypeScript kotogramToJapanese({ furigana: true }) on all kotograms
6. Compare results line by line
"""

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kotogram import MecabJapaneseParser, SudachiJapaneseParser, kotogram_to_japanese


def generate_kotograms(data_path: Path) -> tuple[list[str], list[str]]:
    """Generate kotograms for all sentences using both parsers."""
    mecab_parser = MecabJapaneseParser()
    sudachi_parser = SudachiJapaneseParser(dict_type='full')

    mecab_kotograms = []
    sudachi_kotograms = []

    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 3:
                continue
            sentence = row[2]

            try:
                mecab_kotograms.append(mecab_parser.japanese_to_kotogram(sentence))
            except Exception as e:
                print(f"MeCab error for '{sentence[:30]}...': {e}", file=sys.stderr)
                mecab_kotograms.append("")

            try:
                sudachi_kotograms.append(sudachi_parser.japanese_to_kotogram(sentence))
            except Exception as e:
                print(f"Sudachi error for '{sentence[:30]}...': {e}", file=sys.stderr)
                sudachi_kotograms.append("")

    return mecab_kotograms, sudachi_kotograms


def python_convert_furigana(kotograms: list[str]) -> list[str]:
    """Convert kotograms to Japanese with furigana using Python."""
    results = []
    for kotogram in kotograms:
        if not kotogram:
            results.append("")
            continue
        try:
            results.append(kotogram_to_japanese(kotogram, furigana=True))
        except Exception as e:
            print(f"Python error: {e}", file=sys.stderr)
            results.append(f"ERROR: {e}")
    return results


def typescript_convert_furigana(kotograms_file: Path, output_file: Path) -> bool:
    """Convert kotograms to Japanese with furigana using TypeScript."""
    project_root = Path(__file__).parent.parent

    script = f'''
import {{ kotogramToJapanese }} from './dist/kotogram.js';
import {{ readFileSync, writeFileSync }} from 'fs';

const kotograms = readFileSync('{kotograms_file}', 'utf-8').split('\\n');
const results = kotograms.map(k => {{
    if (!k) return '';
    try {{
        return kotogramToJapanese(k, {{ furigana: true }});
    }} catch (e) {{
        return 'ERROR: ' + e.message;
    }}
}});
writeFileSync('{output_file}', results.join('\\n'));
'''

    # Script must be in project root to find dist/kotogram.js
    script_file = project_root / 'ts_convert_temp.mjs'
    script_file.write_text(script)

    try:
        result = subprocess.run(
            ['node', str(script_file)],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"TypeScript error: {result.stderr}", file=sys.stderr)
            return False
        return True
    finally:
        script_file.unlink()


def compare_results(
    python_results: list[str],
    typescript_results: list[str],
    kotograms: list[str],
) -> tuple[int, int, list[dict]]:
    """Compare Python and TypeScript results."""
    total = len(python_results)
    matches = 0
    mismatches = []

    for i, (py, ts, kg) in enumerate(zip(python_results, typescript_results, kotograms)):
        if not kg:  # Skip empty kotograms
            continue
        if py == ts:
            matches += 1
        else:
            mismatches.append({
                'line': i + 1,
                'kotogram': kg[:100] + ('...' if len(kg) > 100 else ''),
                'python': py,
                'typescript': ts,
            })

    return matches, total, mismatches


def main():
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "jpn_sentences.tsv"

    print("Cross-language furigana validation")
    print("=" * 60)
    print(f"Data: {data_path}")
    print()

    # Step 1: Generate kotograms
    print("Step 1: Generating kotograms from Japanese sentences...")
    mecab_kotograms, sudachi_kotograms = generate_kotograms(data_path)
    print(f"  Generated {len(mecab_kotograms)} MeCab kotograms")
    print(f"  Generated {len(sudachi_kotograms)} Sudachi kotograms")
    print()

    all_mismatches = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for parser_name, kotograms in [("MeCab", mecab_kotograms), ("Sudachi", sudachi_kotograms)]:
            print(f"Step 2: Converting {parser_name} kotograms with Python...")
            python_results = python_convert_furigana(kotograms)
            print(f"  Converted {len(python_results)} kotograms")

            # Write kotograms to temp file for TypeScript
            kotograms_file = tmpdir / f'{parser_name.lower()}_kotograms.txt'
            kotograms_file.write_text('\n'.join(kotograms))

            ts_output_file = tmpdir / f'{parser_name.lower()}_ts_results.txt'

            print(f"Step 3: Converting {parser_name} kotograms with TypeScript...")
            if not typescript_convert_furigana(kotograms_file, ts_output_file):
                print(f"  ❌ TypeScript conversion failed for {parser_name}")
                continue

            typescript_results = ts_output_file.read_text().split('\n')
            print(f"  Converted {len(typescript_results)} kotograms")

            # Step 4: Compare results
            print(f"Step 4: Comparing {parser_name} results...")
            matches, total, mismatches = compare_results(
                python_results, typescript_results, kotograms
            )

            if not mismatches:
                print(f"  ✅ All {matches} {parser_name} conversions match!")
            else:
                print(f"  ❌ {len(mismatches)} mismatches found out of {total} {parser_name} conversions")
                print()
                print(f"  First 5 {parser_name} mismatches:")
                for m in mismatches[:5]:
                    print(f"    Line {m['line']}:")
                    print(f"      Kotogram: {m['kotogram']}")
                    print(f"      Python:     {m['python']}")
                    print(f"      TypeScript: {m['typescript']}")
                    print()
                all_mismatches.extend(mismatches)
            print()

    # Return exit code based on results
    if all_mismatches:
        sys.exit(1)


if __name__ == "__main__":
    main()
