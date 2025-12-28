"""Test to ensure TypeScript Enums are in sync with Python Enums."""

import re
import unittest

from kotogram.constants import FormalityLevel, GenderLevel, RegisterLevel


class TestCrossLanguageSync(unittest.TestCase):
    """Verifies that TypeScript Enums match Python Enums."""

    def setUp(self):
        """Load TypeScript source for analysis."""
        with open("src/analysis.ts", "r", encoding="utf-8") as f:
            self.ts_content = f.read()

    def _extract_ts_enum(self, enum_name: str) -> dict:
        """Extract enum values from TypeScript source."""
        # Match enum Name { ... }
        pattern = rf"export enum {enum_name} \{{(.*?)\}}"
        match = re.search(pattern, self.ts_content, re.DOTALL)
        if not match:
            self.fail(f"Could not find TS enum {enum_name}")

        enum_body = match.group(1)
        # Match KEY = "value" or KEY = 'value'
        pairs = re.findall(r"(\w+)\s*=\s*['\"]([^'\"]+)['\"]", enum_body)
        return dict(pairs)

    def test_formality_level_sync(self):
        """Verify FormalityLevel sync."""
        ts_values = self._extract_ts_enum("FormalityLevel")
        py_values = {e.name: e.value for e in FormalityLevel}
        self.assertEqual(ts_values, py_values, "FormalityLevel enums are out of sync")

    def test_gender_level_sync(self):
        """Verify GenderLevel sync."""
        ts_values = self._extract_ts_enum("GenderLevel")
        py_values = {e.name: e.value for e in GenderLevel}
        self.assertEqual(ts_values, py_values, "GenderLevel enums are out of sync")

    def test_register_level_sync(self):
        """Verify RegisterLevel sync."""
        ts_values = self._extract_ts_enum("RegisterLevel")
        py_values = {e.name: e.value for e in RegisterLevel}
        self.assertEqual(ts_values, py_values, "RegisterLevel enums are out of sync")

    def test_grammar_analysis_fields_sync(self):
        """Verify GrammarAnalysis public fields sync."""
        from dataclasses import fields

        from kotogram.analysis import GrammarAnalysis

        # Extract Python fields
        py_fields = {f.name for f in fields(GrammarAnalysis)}

        # Extract TypeScript fields from GrammarAnalysisData interface
        pattern = r"export interface GrammarAnalysisData \{(.*?)\}"
        match = re.search(pattern, self.ts_content, re.DOTALL)
        if not match:
            self.fail("Could not find TS interface GrammarAnalysisData")

        interface_body = match.group(1)
        # Match fieldName: type;
        ts_fields = set(re.findall(r"(\w+)\??\s*:", interface_body))

        self.assertEqual(
            ts_fields,
            py_fields,
            f"GrammarAnalysis fields are out of sync. TS has {ts_fields}, Py has {py_fields}",
        )


if __name__ == "__main__":
    unittest.main()
