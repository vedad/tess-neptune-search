#!/usr/bin/env python

import logging

__all__ = ["get_logger"]

def get_logger(
        name,
        filename,
        format="%(asctime)s|%(message)s",
        level="DEBUG"
        ):
    handler = logging.FileHandler(filename=filename, mode="w") 
    handler.setFormatter(logging.Formatter(format))

    logger = logging.getLogger(name)
    if level == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif level == "INFO":
        logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger