"""ISIS-style exclusive spectral selection using Sherpa's analysis axis."""

from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np


def xnotice(ids=None, lo=None, hi=None, *, session=None):
    """Exclusively notice an x-axis interval in the current Sherpa units.

    ``xnotice(1, 12, 13)`` restricts dataset 1 to that range. When
    ``set_analysis('wavelength')`` is active its bounds are Angstrom; for
    energy they are keV, and for channel they are channel numbers. ``ids``
    may be a dataset ID, a sequence of IDs, or None for Sherpa's default ID.

    This intentionally replaces each selected dataset's existing notice
    filter. It uses Sherpa's ``ignore_id`` and ``notice_id`` so background
    filters, detector grouping, quality masks, and bin-edge semantics follow
    the loaded Sherpa version. Other datasets are left alone.
    """
    if session is None:
        from sherpa.astro import ui as session

    if ids is None:
        selected = (session.get_default_id(),)
    elif isinstance(ids, (str, Integral)) and not isinstance(ids, (bool, np.bool_)):
        selected = (ids,)
    elif isinstance(ids, Sequence) or isinstance(ids, np.ndarray):
        selected = tuple(ids)
    else:
        raise TypeError("ids must be a dataset ID or a sequence of dataset IDs")
    if not selected or any(not isinstance(key, (str, Integral)) or isinstance(key, (bool, np.bool_))
                           or isinstance(key, str) and not key for key in selected) or \
            len(set(selected)) != len(selected):
        raise ValueError("provide one or more distinct, nonempty Sherpa dataset IDs")
    if not isinstance(lo, Real) or not isinstance(hi, Real) or \
            isinstance(lo, (bool, np.bool_)) or isinstance(hi, (bool, np.bool_)) or \
            not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError("provide finite x-axis bounds with lo < hi")

    # Validate all selected IDs before changing any of their filters.
    for key in selected:
        data = session.get_data(key)
        if not hasattr(data, "channel") or not hasattr(data, "units"):
            raise TypeError(f"dataset {key!r} must be a Sherpa PHA spectrum")
    for key in selected:
        session.ignore_id(key)
        session.notice_id(key, float(lo), float(hi))
