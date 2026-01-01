"""
Site customization hook for Kotogram tests/scripts.

This file is automatically imported by python on startup if tests-py is in PYTHONPATH.
It enables the ParameterRecorder if configured via environment variables.
"""

import instrumentation

# This check is redundant as auto_enable checks env vars, but safe.
instrumentation.auto_enable()
