#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=bad-indentation, import-error

"""
Created on Mon Jul  8 16:15:56 2024

@author: avicenna
"""

import itertools as it
import logging
import json
from pathlib import Path

from typing import Union, Optional

import fractions
import numpy as np


class BadDilutionFormat(Exception):
    pass


class BadTableFormat(Exception):
    pass


class BadModelInput(Exception):
    pass


class UnacceptableInput(Exception):
    """
    THIS CASTLE IS IN AN UNACCEPTABLE CONDITION!
    """  # - Earl of Lemongrab


class InternalError(Exception):
    """
    ONNNNNE MILLION YEARS DUNGEON!
    """  # - Earl of Lemongrab

    def __init__(self, message=""):

        super().__init__(
            "Something went wrong internally: "
            + message
            + " Contact the author."
        )


NTs = ["A", "G", "C", "T"]
NT_pairs = ["".join(x) for x in list(it.product(NTs, NTs)) if x[0] != x[1]]

cdir = Path(__file__).parent.resolve()
with open(Path(cdir,"logging.json"), "r") as fp:
  logging.config.dictConfig(json.load(fp))


def set_logger_props(
    level: Union[str, int], file: Optional[str]=None, mode: Optional[str] = "a"
):
    """
    sets level or adds stream for Thor.models which is the only logger
    in this module.
    """
    logger = logging.getLogger("Thor.models")
    logger.setLevel(level)

    if file:
        handler = logging.FileHandler(file, mode=mode, encoding="utf-8")
        logger.addHandler(handler)


def _catch_log(logger):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception("Error in %s: %s", func.__name__, e)
                raise

        return wrapper

    return decorator


def _get_end_dilution(dils):

    try:
        dils = [float(1 / fractions.Fraction(str(x))) for x in set(dils)]
    except ValueError as e:
        raise BadDilutionFormat(
            f"Invalid dilution in {set(dils)}: "
            "They should be of the form 1/X, '', nan or None"
        ) from e

    return int(np.max(dils))


def _transformation(x):
    return np.log2((1 / x) / 10)


def _convert_dilution(dilution, transformation, start_dilution):

    if (isinstance(dilution, str) and dilution in ["", "nan", "NONE"]) or (
        isinstance(dilution, float) and np.isnan(dilution)
    ):
        return np.nan

    try:
        return transformation(start_dilution) - transformation(
            float(fractions.Fraction(dilution))
        )
    except ValueError as e:
        raise BadDilutionFormat(
            f"Invalid dilution {dilution}: "
            "It should be of the form 1/X, '', nan or None"
        ) from e


def _log2(x):

    if isinstance(x, float) and np.isinf(x):
        return x

    x = str(x)

    if x[0] in ["<", ">"]:
        return np.log2(float(x[1:]) / 10)

    return np.log2(float(x) / 10)


log2 = np.vectorize(_log2)
