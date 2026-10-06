# casacure drop-in replacement for casacore: satisfy the
# `lazy_import("casacore.tables")` in dask-ms and `from casacore.tables import
# ...` in consumers by delegating to the casacure binding.
from . import tables  # noqa: F401
from . import quanta  # noqa: F401
from . import measures  # noqa: F401

# Backend marker: the backend-switching tests assert this instead of matching
# the substring "casacure" in `casacore.__file__`, which is also true whenever
# this checkout lives under a directory named `casacure` — including this one,
# where it made the "real casacore" branch fail unconditionally.
__casacure_shim__ = True
