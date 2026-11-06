"""Wire encoding for role-to-role payloads: a safe record, with pickle opt-in.

What this is for:
    Every CADET app turns an application object (a dict with a camera frame in
    it, or a pipeline :class:`~cadet.pipeline.messages.Message`) into the bytes a
    :class:`~cadet.core.transport.Transport` carries. The original scripts did
    that with ``zlib.compress(pickle.dumps(obj))``, which means the receiver runs
    ``pickle.loads`` on whatever arrived on its socket - arbitrary code execution
    by anyone who can connect to the port. This module is the one place a payload
    is encoded, so the encoding is chosen by config instead of being written into
    every app.

What it contains:
    - :func:`encode` / :func:`decode`: encode an object to a payload and back,
      using the process's selected encoding.
    - :data:`SAFE` / :data:`PICKLE`: the two encodings. ``safe`` is a
      self-describing record that executes nothing; ``pickle`` is the original's
      format, kept because reproducing the original is the fidelity default and
      because an object with no safe representation can still be carried.
    - :func:`set_wire_encoding` / :func:`wire_encoding` /
      :func:`set_wire_encoding_from_config`: the per-process selection, set once
      from ``transport.encoding`` by the role runtimes.
    - :func:`register_wire_types`: the whitelist of classes the safe encoding may
      reconstruct. Only a registered class can come back off the wire, and it is
      rebuilt by setting attributes, never by calling code named in the payload.
    - :func:`record_length`: how many bytes the record at the front of a buffer
      occupies, which is what lets ``transport.mode: stream`` recover message
      boundaries from the encoding instead of from a transport length prefix.
    - :class:`WireEncodingError`: raised for a payload this process will not
      decode, or an object the safe encoding cannot express.

    The safe record is::

        b"CDT1" | uint32 body length | zlib(uint32 header length | header | blobs)

    The header is JSON describing the object tree. Binary leaves (``bytes`` and
    numpy arrays) are *not* in the header - they are appended after it as raw
    bytes and referenced by offset and length, so a camera frame is never copied
    through a text encoder. The whole body is then zlib-compressed, exactly as
    the original compressed its pickle, so the payload size is comparable.

    The magic prefix is what makes the two encodings tell themselves apart: a
    record that does not start with it is a pickle payload, and :func:`decode`
    refuses one unless this process opted in. That refusal, not the new format by
    itself, is the fix for the defect.

How it fits into CADET:
    Called by the example apps' encode/decode helpers (``examples/*/app.py``), by
    :mod:`cadet.pipeline.runtime` at a stage boundary, and by
    :mod:`cadet.roles.load`. The transport below it carries bytes and does not
    care which encoding produced them, with one exception: stream mode needs
    :func:`record_length`, which only the safe record offers.

How to configure it:
    ``transport.encoding`` in the config (``safe`` or ``pickle``). The role
    runtimes call :func:`set_wire_encoding_from_config` once, before any traffic.
    A process that never calls it encodes with :data:`DEFAULT_ENCODING`.
"""

import json
import struct
import zlib
from typing import Any, Dict, List, Optional, Tuple

# The two encodings. `safe` cannot execute anything it decodes; `pickle` is the
# original scripts' format and can.
SAFE = "safe"
PICKLE = "pickle"
ENCODINGS = (SAFE, PICKLE)

# What a process encodes with when nothing selected an encoding. The config
# default is the one that matters; this is only the library fallback for code
# (a test, a tool) that never loaded a config.
DEFAULT_ENCODING = SAFE

# Safe-record framing. Fixed by the format, not configurable: a reader has to
# know these to find the record's end.
MAGIC = b"CDT1"
_LEN = ">I"
_LEN_BYTES = struct.calcsize(_LEN)
# Bytes before the compressed body: the magic plus the body length.
RECORD_PREFIX_BYTES = len(MAGIC) + _LEN_BYTES

# Compression level: zlib's default, which is what the original's bare
# `zlib.compress(...)` used, so the compressed size is the same work.
_ZLIB_LEVEL = zlib.Z_DEFAULT_COMPRESSION

# The reserved header key that marks a tagged node. A NUL character cannot occur
# in a field name an app would choose, so a plain dict is never mistaken for a
# tagged node; `encode` refuses a dict that uses it rather than silently
# producing a record that decodes to something else.
_TAG = "\u0000t"

# Tag values, short because the header is JSON and every byte of it is
# compressed and sent.
_T_TUPLE = "t"
_T_BYTES = "y"
_T_ARRAY = "a"
_T_OBJECT = "o"
_T_MAP = "m"


class WireEncodingError(Exception):
    """A payload this process will not decode, or an object it cannot encode."""


class _Registry:
    """The classes the safe encoding may rebuild, by name.

    A name, not an import path: decoding must never be able to reach a class
    that was not registered by code in this repository, which is the difference
    between this and pickle.
    """

    def __init__(self):
        self._by_name: Dict[str, type] = {}
        self._by_cls: Dict[type, str] = {}

    def add(self, name: str, cls: type) -> None:
        known = self._by_name.get(name)
        if known is not None and known is not cls:
            raise WireEncodingError(
                "wire type {!r} is already registered to {}".format(name, known)
            )
        self._by_name[name] = cls
        self._by_cls[cls] = name

    def name_for(self, cls: type) -> Optional[str]:
        return self._by_cls.get(cls)

    def cls_for(self, name: str) -> type:
        cls = self._by_name.get(name)
        if cls is None:
            raise WireEncodingError(
                "wire type {!r} is not registered; a payload may only rebuild a "
                "class this build declared".format(name)
            )
        return cls


_TYPES = _Registry()


def register_wire_types(types: Dict[str, type]) -> None:
    """Whitelist ``{name: class}`` for the safe encoding.

    Called at import time by the module that owns the classes (see
    :mod:`cadet.pipeline.messages`), so a process that can hold such an object
    can also encode and decode one, and a process that cannot refuses by name
    rather than importing something to find out.
    """
    for name, cls in types.items():
        _TYPES.add(name, cls)


def registered_wire_types() -> Tuple[str, ...]:
    """Names the safe encoding can rebuild, sorted (for tests and reports)."""
    return tuple(sorted(_TYPES._by_name))


# --------------------------------------------------------------------------
# per-process encoding selection
# --------------------------------------------------------------------------
_selected = DEFAULT_ENCODING


def set_wire_encoding(encoding: str) -> None:
    """Select this process's encoding (``safe`` or ``pickle``)."""
    if encoding not in ENCODINGS:
        raise WireEncodingError(
            "unknown wire encoding {!r}; expected one of {}".format(encoding, ENCODINGS)
        )
    global _selected
    _selected = encoding


def wire_encoding() -> str:
    """Return this process's encoding."""
    return _selected


def set_wire_encoding_from_config(config) -> str:
    """Select the encoding named by a resolved config, and return it.

    The single call the role runtimes make, so ``transport.encoding`` reaches the
    apps without each of them reading the config.
    """
    encoding = getattr(getattr(config, "transport", None), "encoding", DEFAULT_ENCODING)
    set_wire_encoding(encoding)
    return encoding


# --------------------------------------------------------------------------
# encode
# --------------------------------------------------------------------------
class _Blobs:
    """Binary leaves in the order the header refers to them.

    Each leaf's node carries its own ``o``ffset and ``n``umber of bytes, so the
    decoder slices it directly out of the body instead of walking the list.

    The buffers are kept as they came - a ``bytes`` or a memoryview over an
    array - and never concatenated: the compressor reads each one in turn, so a
    frame is compressed straight out of the array's own memory instead of being
    copied into a body first. That copy was measurable on an 800x600 frame.
    """

    __slots__ = ("chunks", "offset")

    def __init__(self):
        self.chunks: List[Any] = []
        self.offset = 0

    def add(self, data, size: int) -> Tuple[int, int]:
        offset = self.offset
        self.chunks.append(data)
        self.offset += size
        return offset, size


def encode(obj: Any, encoding: Optional[str] = None) -> bytes:
    """Encode ``obj`` into one wire payload."""
    enc = encoding or _selected
    if enc == PICKLE:
        return encode_pickle(obj)
    return encode_safe(obj)


def encode_safe(obj: Any) -> bytes:
    """Encode ``obj`` as a safe record, whatever this process selected."""
    blobs = _Blobs()
    header = json.dumps(_to_node(obj, blobs), separators=(",", ":")).encode("utf-8")
    # Fed to the compressor piece by piece rather than assembled first: the
    # pieces are the header and the binary leaves' own buffers, and a camera
    # frame is large enough that one avoidable copy of it shows up in a
    # measurement.
    compressor = zlib.compressobj(_ZLIB_LEVEL)
    parts = [compressor.compress(struct.pack(_LEN, len(header)) + header)]
    for chunk in blobs.chunks:
        parts.append(compressor.compress(chunk))
    parts.append(compressor.flush())
    packed = b"".join(parts)
    return MAGIC + struct.pack(_LEN, len(packed)) + packed


def encode_pickle(obj: Any) -> bytes:
    """Encode ``obj`` the way the original scripts did."""
    import pickle  # local: an opt-in path should not be imported just to refuse it

    return zlib.compress(pickle.dumps(obj), _ZLIB_LEVEL)


def _to_node(value: Any, blobs: _Blobs) -> Any:
    """Return the JSON-able header node for ``value``, appending binary leaves."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        # Kept as it is: the compressor reads any buffer, so a JPEG does not
        # need copying to be encoded.
        buffer = value if isinstance(value, bytes) else memoryview(value)
        size = len(value) if not isinstance(value, memoryview) else value.nbytes
        offset, size = blobs.add(buffer, size)
        return {_TAG: _T_BYTES, "o": offset, "n": size}
    if isinstance(value, list):
        return [_to_node(item, blobs) for item in value]
    if isinstance(value, tuple):
        return {_TAG: _T_TUPLE, "v": [_to_node(item, blobs) for item in value]}
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            if _TAG in value:
                raise WireEncodingError(
                    "a dict key may not be the reserved wire tag; rename the field "
                    "or set transport.encoding: pickle"
                )
            return {key: _to_node(item, blobs) for key, item in value.items()}
        # Non-string keys cannot be a JSON object's, so the mapping travels as
        # pairs. No app needs this today; it is here so a payload shape change
        # cannot turn working code into a refusal.
        return {
            _TAG: _T_MAP,
            "v": [[_to_node(k, blobs), _to_node(v, blobs)] for k, v in value.items()],
        }
    node = _numpy_node(value, blobs)
    if node is not None:
        return node
    name = _TYPES.name_for(type(value))
    if name is not None:
        return {_TAG: _T_OBJECT, "c": name, "v": _to_node(_state_of(value), blobs)}
    raise WireEncodingError(
        "{} cannot be expressed in the safe wire encoding; convert it at the "
        "boundary (plain values, bytes or a numpy array), register it with "
        "register_wire_types, or set transport.encoding: pickle".format(type(value).__name__)
    )


def _numpy_node(value: Any, blobs: _Blobs) -> Optional[Any]:
    """Return a node for a numpy array or scalar, or ``None`` if it is neither."""
    np = _numpy()
    if np is None:
        return None
    if isinstance(value, np.ndarray):
        # C-contiguous so the bytes are the array in row-major order, which is
        # what `frombuffer` + `reshape` reads back.
        contiguous = np.ascontiguousarray(value)
        # A memoryview over the array's own bytes: no copy, and the compressor
        # reads it directly.
        buffer = memoryview(contiguous).cast("B")
        offset, size = blobs.add(buffer, buffer.nbytes)
        return {
            _TAG: _T_ARRAY,
            "d": contiguous.dtype.str,
            "s": list(contiguous.shape),
            "o": offset,
            "n": size,
        }
    if isinstance(value, np.generic):
        # A numpy scalar travels as a Python scalar: nothing reads it back as a
        # numpy type, and keeping it would mean a dtype per float.
        return _to_node(value.item(), blobs)
    return None


def _state_of(value: Any) -> Dict[str, Any]:
    """Return the attribute state of a registered object, as a plain dict."""
    state = getattr(value, "__dict__", None)
    if state is not None:
        return dict(state)
    # A __slots__ class (StageStamp) has no __dict__; its state is its slots.
    out: Dict[str, Any] = {}
    for cls in type(value).__mro__:
        for slot in getattr(cls, "__slots__", ()) or ():
            if hasattr(value, slot):
                out[slot] = getattr(value, slot)
    return out


# --------------------------------------------------------------------------
# decode
# --------------------------------------------------------------------------
def decode(payload: bytes, encoding: Optional[str] = None) -> Any:
    """Decode one wire payload.

    A safe record is always decoded, whichever encoding this process sends with,
    so a reader is never the reason a peer's safe payload is rejected. A payload
    that is *not* a safe record can only be a pickle, and is refused unless
    pickle was opted into - that refusal is what closes the defect.
    """
    data = bytes(payload)
    if data[: len(MAGIC)] == MAGIC:
        return decode_safe(data)
    enc = encoding or _selected
    if enc != PICKLE:
        raise WireEncodingError(
            "refusing to unpickle {} bytes from the wire: transport.encoding is "
            "{!r}. Unpickling socket data executes code chosen by the peer; set "
            "transport.encoding: pickle only for a run that needs the original's "
            "wire format".format(len(data), enc)
        )
    import pickle  # local: see encode_pickle

    return pickle.loads(zlib.decompress(data))


def decode_safe(payload: bytes) -> Any:
    """Decode a safe record."""
    data = bytes(payload)
    if data[: len(MAGIC)] != MAGIC:
        raise WireEncodingError("not a safe wire record (bad magic)")
    (packed_len,) = struct.unpack(_LEN, data[len(MAGIC) : RECORD_PREFIX_BYTES])
    end = RECORD_PREFIX_BYTES + packed_len
    if len(data) < end:
        raise WireEncodingError(
            "truncated safe wire record: {} body bytes present, {} declared".format(
                len(data) - RECORD_PREFIX_BYTES, packed_len
            )
        )
    try:
        # Sliced through a memoryview: a bytes slice of a megabyte payload is a
        # megabyte copy, and the decompressor does not need one.
        body = zlib.decompress(memoryview(data)[RECORD_PREFIX_BYTES:end])
    except zlib.error as exc:
        raise WireEncodingError("corrupt safe wire record: {}".format(exc))
    (header_len,) = struct.unpack(_LEN, body[:_LEN_BYTES])
    blob_start = _LEN_BYTES + header_len
    header = json.loads(body[_LEN_BYTES:blob_start].decode("utf-8"))
    return _from_node(header, body, blob_start)


def record_length(buffer: bytes) -> Optional[int]:
    """Bytes the record at the front of ``buffer`` occupies, or ``None``.

    ``None`` means "not enough bytes yet", which is the answer a stream reader
    needs: it is how ``transport.mode: stream`` finds message boundaries without
    a transport length prefix. A buffer that does not start with the safe magic
    raises, because a pickle payload carries no length and cannot be delimited in
    a stream at all.
    """
    # Only the record's own prefix is read, and only that is copied: a stream
    # reader calls this on a growing buffer after every read, so copying the
    # buffer here would make reading one record quadratic in its size.
    prefix = bytes(buffer[: len(MAGIC)])
    if prefix != MAGIC[: len(prefix)]:
        raise WireEncodingError(
            "stream does not begin with a safe wire record; transport.mode: stream "
            "requires transport.encoding: safe"
        )
    if len(buffer) < RECORD_PREFIX_BYTES:
        return None
    (packed_len,) = struct.unpack(_LEN, bytes(buffer[len(MAGIC) : RECORD_PREFIX_BYTES]))
    total = RECORD_PREFIX_BYTES + packed_len
    return total if len(buffer) >= total else None


def _from_node(node: Any, body: bytes, blob_start: int) -> Any:
    """Rebuild a value from its header node, reading blobs out of ``body``."""
    if isinstance(node, list):
        return [_from_node(item, body, blob_start) for item in node]
    if not isinstance(node, dict):
        return node
    tag = node.get(_TAG)
    if tag is None:
        return {key: _from_node(item, body, blob_start) for key, item in node.items()}
    if tag == _T_TUPLE:
        return tuple(_from_node(item, body, blob_start) for item in node["v"])
    if tag == _T_MAP:
        return {
            _from_node(k, body, blob_start): _from_node(v, body, blob_start)
            for k, v in node["v"]
        }
    if tag == _T_BYTES:
        start = blob_start + node["o"]
        return body[start : start + node["n"]]
    if tag == _T_ARRAY:
        return _array(node, body, blob_start)
    if tag == _T_OBJECT:
        cls = _TYPES.cls_for(node["c"])
        # Built without running __init__: the payload supplies state, never a
        # call. Attributes are set one by one so a __slots__ class works too.
        obj = object.__new__(cls)
        for key, item in _from_node(node["v"], body, blob_start).items():
            setattr(obj, key, item)
        return obj
    raise WireEncodingError("unknown wire tag {!r}".format(tag))


def _array(node: Dict[str, Any], body: bytes, blob_start: int):
    """Rebuild a numpy array from its node, as one copy out of the body.

    ``frombuffer`` over the body is a view and would be read-only, which some
    model backends refuse to take; the copy makes it writable and owns its
    memory, which is what ``pickle.loads`` also produced.
    """
    np = _numpy()
    if np is None:
        raise WireEncodingError("numpy is required to decode an array payload")
    dtype = np.dtype(node["d"])
    count = node["n"] // dtype.itemsize
    view = np.frombuffer(body, dtype=dtype, count=count, offset=blob_start + node["o"])
    return view.reshape(tuple(node["s"])).copy()


_np = None
_np_tried = False


def _numpy():
    """Return numpy if it is installed, else ``None`` (imported once)."""
    global _np, _np_tried
    if not _np_tried:
        _np_tried = True
        try:
            import numpy

            _np = numpy
        except ImportError:
            _np = None
    return _np
