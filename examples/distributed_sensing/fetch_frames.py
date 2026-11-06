"""Fetch replay frames for the distributed sensing example into a local, git-ignored cache.

What this is for:
    The example's replay config needs a folder of images. This script downloads
    a small subset of a public dataset at run time and writes it under
    ``datasets/``, which ``.gitignore`` already excludes, so no image ever
    reaches the index. Nothing here is redistributed with the repository.

What it contains:
    - :func:`stream_entries`: read a remote zip's entries in stream order,
      stopping as soon as enough have been taken, so a 778 MB archive costs only
      the first few megabytes.
    - :func:`fetch`: write the first ``limit`` images of an archive into a
      destination folder.
    - a command line: ``--limit``, ``--dest``, ``--url``.

How it fits into CADET:
    Run it before ``examples/distributed_sensing/replay.yaml``, whose ``source.path`` points at
    the default destination. The synthetic configs need nothing from here.

How to configure it:
    ``--url`` takes any zip of images served over HTTP; the default is the COCO
    validation split, whose terms the example's README states. ``--limit`` caps
    how many images are written and ``--dest`` where they go.
"""

import argparse
import os
import struct
import sys
import urllib.request
import zlib

# The COCO validation split. Its images are served as one zip, in which each
# entry is stored uncompressed, so the entries can be read in stream order.
DEFAULT_URL = "http://images.cocodataset.org/zips/val2017.zip"
DEFAULT_DEST = os.path.join("datasets", "distributed_sensing", "coco-val2017")
DEFAULT_LIMIT = 40

# Zip local file header: signature, version, flags, method, time, date, crc,
# compressed size, uncompressed size, name length, extra length.
LOCAL_HEADER = "<4sHHHHHIIIHH"
LOCAL_HEADER_SIZE = struct.calcsize(LOCAL_HEADER)
LOCAL_SIGNATURE = b"PK\x03\x04"
METHOD_STORED = 0
METHOD_DEFLATE = 8
# Set when the entry's sizes follow the data instead of preceding it, which a
# stream reader cannot skip past.
FLAG_DATA_DESCRIPTOR = 0x08
# How much of the response to hold in memory while looking for one entry.
CHUNK_BYTES = 1 << 18
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def stream_entries(url, limit, timeout_s=120.0):
    """Yield ``(name, data)`` for the first ``limit`` image entries of a zip.

    The connection is closed as soon as ``limit`` entries have been read, so the
    rest of the archive is never transferred.
    """
    taken = 0
    buffer = b""
    response = urllib.request.urlopen(url, timeout=timeout_s)
    try:
        while taken < limit:
            while len(buffer) < LOCAL_HEADER_SIZE:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    return
                buffer += chunk
            header = struct.unpack(LOCAL_HEADER, buffer[:LOCAL_HEADER_SIZE])
            signature, flags, method = header[0], header[2], header[3]
            compressed, name_len, extra_len = header[7], header[9], header[10]
            if signature != LOCAL_SIGNATURE:
                return  # past the last entry: the central directory starts here
            if flags & FLAG_DATA_DESCRIPTOR:
                raise RuntimeError(
                    "the archive at {} stores entry sizes after the data, which "
                    "cannot be read in stream order; download it with a zip tool "
                    "and point --dest at the extracted folder instead".format(url)
                )
            needed = LOCAL_HEADER_SIZE + name_len + extra_len + compressed
            while len(buffer) < needed:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    return
                buffer += chunk
            name = buffer[LOCAL_HEADER_SIZE:LOCAL_HEADER_SIZE + name_len].decode("utf-8")
            body = buffer[needed - compressed:needed]
            buffer = buffer[needed:]
            if not name.lower().endswith(IMAGE_SUFFIXES):
                continue
            if method == METHOD_STORED:
                data = body
            elif method == METHOD_DEFLATE:
                data = zlib.decompressobj(-zlib.MAX_WBITS).decompress(body)
            else:
                raise RuntimeError(
                    "entry {} uses compression method {}, which this reader does "
                    "not handle".format(name, method)
                )
            taken += 1
            yield os.path.basename(name), data
    finally:
        response.close()


def fetch(url, dest, limit):
    """Write the first ``limit`` images of the archive at ``url`` into ``dest``."""
    if not os.path.isdir(dest):
        os.makedirs(dest)
    written = 0
    for name, data in stream_entries(url, limit):
        with open(os.path.join(dest, name), "wb") as handle:
            handle.write(data)
        written += 1
    return written


def main(argv=None):
    """Parse the arguments and fetch the frames."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=DEFAULT_URL, help="zip of images to read from")
    parser.add_argument("--dest", default=DEFAULT_DEST, help="folder to write the images into")
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, help="how many images to write"
    )
    args = parser.parse_args(argv)
    written = fetch(args.url, args.dest, args.limit)
    if not written:
        print("no images found in {}".format(args.url), file=sys.stderr)
        return 1
    print("wrote {} image(s) to {}".format(written, args.dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
