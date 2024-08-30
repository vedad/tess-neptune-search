#!/usr/bin/env python

__all__ = ["validate_wotan_args", "validate_search_args"]

BREAK_TOLERANCE = 0.5
EDGE_CUTOFF = 0.5
MAX_SPLINES = 1000
STDEV_CUT = 2
RETURN_TREND = True

NOVERLAP = 3


def validate_wotan_args(**kwargs):
    kwargs["break_tolerance"] = BREAK_TOLERANCE
    kwargs["edge_cutoff"] = EDGE_CUTOFF
    kwargs["return_trend"] = RETURN_TREND
    if kwargs["method"] == "spline":
        kwargs.pop("window_length")
        kwargs["max_splines"] = MAX_SPLINES
        kwargs["stdev_cut"] = STDEV_CUT
    return kwargs

def validate_search_args(**kwargs):
    kwargs["noverlap"] = NOVERLAP
    return kwargs