#!/usr/bin/env python3
"""
Tests for the CsvEvaluationMode file logging setup.

The module logger of sources/benchmark_evaluation/csv_mode.py must write its
records to logs/evaluation._csv_mode.log, attach its file handler only once
per process, and keep working when the root logger was never configured.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.benchmark_evaluation import csv_mode


def _detach_file_handlers(logger: logging.Logger) -> None:
    """Remove and close any rotating file handlers left by a previous test."""
    for handler in list(logger.handlers):
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            logger.removeHandler(handler)
            handler.close()


def _fresh_module_logger() -> logging.Logger:
    logger = logging.getLogger("sources.benchmark_evaluation.csv_mode")
    _detach_file_handlers(logger)
    return logger


def test_log_file_lives_in_logs_directory():
    assert csv_mode.EVAL_LOG_FILE == Path("logs") / "evaluation._csv_mode.log"


def test_logger_writes_records_to_log_file(tmp_path, monkeypatch):
    log_file = tmp_path / "logs" / "evaluation._csv_mode.log"
    monkeypatch.setattr(csv_mode, "EVAL_LOG_FILE", log_file)
    logger = _fresh_module_logger()

    csv_mode._attach_eval_log_file_handler(logger)
    logger.info("hello eval log")
    logger.debug("debug records reach the file too")

    content = log_file.read_text(encoding="utf-8")
    assert "hello eval log" in content
    assert "debug records reach the file too" in content
    assert "INFO" in content
    _detach_file_handlers(logger)


def test_handler_attached_only_once(tmp_path, monkeypatch):
    log_file = tmp_path / "logs" / "evaluation._csv_mode.log"
    monkeypatch.setattr(csv_mode, "EVAL_LOG_FILE", log_file)
    logger = _fresh_module_logger()

    csv_mode._attach_eval_log_file_handler(logger)
    csv_mode._attach_eval_log_file_handler(logger)

    file_handlers = [h for h in logger.handlers
                     if isinstance(h, logging.handlers.RotatingFileHandler)]
    assert len(file_handlers) == 1
    _detach_file_handlers(logger)
