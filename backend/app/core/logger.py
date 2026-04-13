"""统一日志配置。"""

from __future__ import annotations

import logging
import sys

_configured = False


def setup_logger(name: str = "archiver", level: int = logging.INFO) -> logging.Logger:
    global _configured
    logger = logging.getLogger(name)

    if not _configured:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-7s %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(level)
        _configured = True

    return logger


log = setup_logger()
