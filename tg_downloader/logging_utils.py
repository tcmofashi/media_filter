"""Logging helpers for timestamped console output."""

from __future__ import annotations

import io
import logging
import sys
from datetime import datetime
from typing import TextIO


_NOISE_FILTER_INSTALLED = False


class _ExternalNoiseFilter(logging.Filter):
    """Suppress third-party log spam that we already handle in app-level logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pylint: disable=broad-except
            message = str(record.msg)

        if record.name == "asyncio" and "socket.send() raised exception." in message:
            return False

        if not record.name.startswith("pyrogram"):
            return True

        if "FILE_REFERENCE_EXPIRED" in message:
            return False
        if "Retrying \"" in message and ("Connection lost" in message or "Broken pipe" in message):
            return False

        if record.exc_info:
            exc = record.exc_info[1]
            if exc is not None:
                text = str(exc)
                if "FILE_REFERENCE_EXPIRED" in text:
                    return False
                if isinstance(exc, OSError) and ("Connection lost" in text or "Broken pipe" in text):
                    return False

        return True


class TimestampedTextIO(io.TextIOBase):
    """Prefix each output line with the current local timestamp."""

    def __init__(self, wrapped: TextIO) -> None:
        self._wrapped = wrapped
        self._at_line_start = True

    @property
    def encoding(self) -> str | None:
        return getattr(self._wrapped, "encoding", None)

    @property
    def errors(self) -> str | None:
        return getattr(self._wrapped, "errors", None)

    @property
    def buffer(self) -> io.BufferedWriter | None:
        return getattr(self._wrapped, "buffer", None)

    def fileno(self) -> int:
        return self._wrapped.fileno()

    def flush(self) -> None:
        self._wrapped.flush()

    def isatty(self) -> bool:
        return self._wrapped.isatty()

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if not text:
            return 0

        written = 0
        for chunk in text.splitlines(keepends=True):
            if self._at_line_start:
                self._wrapped.write(datetime.now().strftime("[%Y-%m-%d %H:%M:%S] "))
            self._wrapped.write(chunk)
            written += len(chunk)
            self._at_line_start = chunk.endswith("\n")
            if self._at_line_start:
                self._wrapped.flush()
        return written


def install_timestamped_output() -> None:
    """Wrap stdout and stderr with timestamped line-prefix streams."""
    if not isinstance(sys.stdout, TimestampedTextIO):
        sys.stdout = TimestampedTextIO(sys.stdout)
    if not isinstance(sys.stderr, TimestampedTextIO):
        sys.stderr = TimestampedTextIO(sys.stderr)
    configure_external_logging()


def configure_external_logging() -> None:
    """Trim noisy third-party logging so downloader output stays readable."""
    global _NOISE_FILTER_INSTALLED
    if _NOISE_FILTER_INSTALLED:
        return

    noise_filter = _ExternalNoiseFilter()
    for logger_name in (
        "asyncio",
        "pyrogram.client",
        "pyrogram.dispatcher",
        "pyrogram.methods.advanced.save_file",
        "pyrogram.session.session",
    ):
        logging.getLogger(logger_name).addFilter(noise_filter)

    _NOISE_FILTER_INSTALLED = True
