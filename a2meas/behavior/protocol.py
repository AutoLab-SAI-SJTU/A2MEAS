"""Official BEHAVIOR msgpack NumPy encoding."""

import functools
from typing import Any
import msgpack
import numpy as np


def _pack_array(obj: Any) -> Any:
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def _unpack_array(obj: Any) -> Any:
    if b"__ndarray__" in obj or b"__npgeneric__" in obj:
        dtype = np.dtype(obj[b"dtype"])
        if dtype.kind in ("V", "O", "c"):
            raise ValueError("Unsupported array dtype")
        if b"__ndarray__" in obj:
            return np.ndarray(buffer=obj[b"data"], dtype=dtype, shape=obj[b"shape"])
        return dtype.type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)
