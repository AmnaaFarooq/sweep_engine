"""
Shared logger configuration.

Entry points (data_loader.main, later backtester / app) call setup_logging()
exactly once. Library modules just call get_logger(__name__) and never
configure handlers, so importing a module (e.g. under pytest) has no side
effects such as creating log files.

Log lines are timestamped in UTC and level-tagged.
"""
from __future__ import annotations

import logging
import sys
import time

import config as cfg

_configured = False


def setup_logging(log_to_file: bool = True) -> None:
    """Configure the root logger (console + optional file). Idempotent."""
    global _configured
    if _configured:
        return

    formatter = logging.Formatter(cfg.LOG_FORMAT, datefmt=cfg.LOG_DATE_FORMAT)
    formatter.converter = time.gmtime          # timestamps in UTC, not local time

    root = logging.getLogger()
    root.setLevel(cfg.LOG_LEVEL)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_to_file:
        cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
        # utf-8 explicitly: the Windows default (cp1252) can choke on odd characters.
        file_handler = logging.FileHandler(cfg.LOG_PATH, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger that inherits the shared configuration."""
    return logging.getLogger(name)
