import os
import sys

# Whitelist hooks (marking these as used)
__all__ = ["pytest_configure", "pytest_sessionstart", "pytest_sessionfinish"]

# Add current directory to path to allow importing sibling module
sys.path.append(os.path.dirname(__file__))

import train_record_conftest  # pylint: disable=wrong-import-position


def pytest_configure(config):
    train_record_conftest.pytest_configure(config)

    # Monkeypatch terminal reporter to suppress "generated xml file" message
    plugin = config.pluginmanager.getplugin("terminalreporter")
    if plugin:
        original_write_line = plugin.write_line

        def write_line_wrapper(content, **kwargs):
            if "generated xml file:" not in content:
                original_write_line(content, **kwargs)

        plugin.write_line = write_line_wrapper


def pytest_sessionstart(session):
    train_record_conftest.pytest_sessionstart(session)


def pytest_sessionfinish(session, exitstatus):
    train_record_conftest.pytest_sessionfinish(session, exitstatus)
