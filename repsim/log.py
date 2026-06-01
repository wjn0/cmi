"""Logging setup for long-running runs.

Bare ``print`` is block-buffered when stdout is a file (as under SLURM), so its
output can be held back for minutes and a running job looks stuck. A
``logging.StreamHandler`` flushes after every record, so progress appears in the
log file in real time. Modules log via ``log = logging.getLogger(__name__)``.
"""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Ensure root logging emits at ``level`` through a flushing handler.

    The real fix for unreliable SLURM logs is logging itself:
    ``StreamHandler.emit`` flushes after every record, unlike bare ``print``
    (block-buffered to a file). So when a handler already exists -- e.g. Hydra
    has configured its console + ``run.log`` file handlers -- we leave them in
    place and only set the level, rather than clobbering Hydra's ``run.log``. In
    a standalone context (no handlers, e.g. ``scripts/cache_check.py``) we add a
    flushing stdout handler. Safe to call more than once.

    Args:
        level: Minimum level to emit (defaults to ``INFO``).
    """
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root.addHandler(handler)
    root.setLevel(level)
