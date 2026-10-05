# casacure drop-in replacement for casacore: satisfy the
# `lazy_import("casacore.tables")` in dask-ms and `from casacore.tables import
# ...` in consumers by delegating to the casacure binding.
from . import tables  # noqa: F401
from . import quanta  # noqa: F401
from . import measures  # noqa: F401
