"""Data integrity exception for pipeline defense-in-depth checks.

Every raise site must include: stage name, offending value(s),
which invariant was violated, and where to look upstream.
"""


class DataIntegrityException(Exception):
    """Raised when a pipeline stage detects a data integrity violation.

    Each instance should carry enough context to diagnose the upstream bug
    from the error message alone, without re-running the pipeline or
    attaching a debugger.
    """
