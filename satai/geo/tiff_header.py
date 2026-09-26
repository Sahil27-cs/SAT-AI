"""Read a GeoTIFF's georeferencing from its header alone, over HTTP range requests.

Purpose: answer "where is this chip?" without downloading the chip. A TIFF's
image file directory sits at the front of the file, so the first few tens of
kilobytes carry the width, height, pixel scale and tie point -- everything
needed to compute a bounding box. For 68 Sen1Floods11 chips that is ~4 MB of
range requests instead of ~200 MB of rasters.

Pure standard library, deliberately. This runs before the geospatial stack is
installed (it is what tells us *which* footprint to build that stack around),
so it cannot depend on rasterio or GDAL.

Scope: enough of TIFF 6.0 and the GeoTIFF 1.0 spec to extract georeferencing
from the well-formed, GDAL-written files this project consumes. It is not a
general TIFF reader and does not try to be -- unparseable files are reported
as such rather than guessed at.

References: TIFF 6.0 specification; OGC GeoTIFF 1.1 (OGC 19-008r4).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

from satai.errors import ValidationError

__all__ = ["GeoTIFFHeader", "parse_geotiff_header", "union_bbox"]

# TIFF tags we care about.
_TAG_IMAGE_WIDTH = 256
_TAG_IMAGE_LENGTH = 257
_TAG_BITS_PER_SAMPLE = 258
_TAG_COMPRESSION = 259
_TAG_SAMPLES_PER_PIXEL = 277
_TAG_SAMPLE_FORMAT = 339
_TAG_GDAL_NODATA = 42113
_TAG_MODEL_PIXEL_SCALE = 33550
_TAG_MODEL_TIEPOINT = 33922
_TAG_GEO_KEY_DIRECTORY = 34735

# GeoTIFF keys, read out of the GeoKeyDirectory.
_KEY_GEOGRAPHIC_TYPE = 2048  # e.g. 4326
_KEY_PROJECTED_CS_TYPE = 3072  # e.g. 32645

#: TIFF SampleFormat values, as a readable dtype family.
_SAMPLE_FORMATS = {1: "uint", 2: "int", 3: "float", 4: "undefined"}

#: The compression schemes that actually turn up in EO products.
_COMPRESSIONS = {
    1: "none",
    5: "lzw",
    7: "jpeg",
    8: "deflate",
    32773: "packbits",
    34712: "jpeg2000",
    34887: "lerc",
    50000: "zstd",
    50001: "webp",
}

# (size in bytes, struct code) per TIFF field type.
_FIELD_TYPES: dict[int, tuple[int, str]] = {
    1: (1, "B"),  # BYTE
    2: (1, "c"),  # ASCII
    3: (2, "H"),  # SHORT
    4: (4, "I"),  # LONG
    5: (8, "II"),  # RATIONAL
    6: (1, "b"),  # SBYTE
    8: (2, "h"),  # SSHORT
    9: (4, "i"),  # SLONG
    10: (8, "ii"),  # SRATIONAL
    11: (4, "f"),  # FLOAT
    12: (8, "d"),  # DOUBLE
    16: (8, "Q"),  # LONG8 (BigTIFF)
}


@dataclass(frozen=True)
class GeoTIFFHeader:
    """Georeferencing extracted from a TIFF header."""

    width: int
    height: int
    origin_x: float
    origin_y: float
    pixel_size_x: float
    pixel_size_y: float
    epsg: int | None

    bands: int = 1
    bits_per_sample: int | None = None
    sample_format: str | None = None
    compression: str | None = None
    nodata: str | None = None

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """``(min_x, min_y, max_x, max_y)`` in the file's own CRS.

        The tie point gives the *upper-left* corner and pixel_size_y is
        positive in a GeoTIFF's ModelPixelScale, so the raster extends
        downward in y from the origin.
        """
        min_x = self.origin_x
        max_x = self.origin_x + self.width * self.pixel_size_x
        max_y = self.origin_y
        min_y = self.origin_y - self.height * self.pixel_size_y
        return (min(min_x, max_x), min(min_y, max_y), max(min_x, max_x), max(min_y, max_y))

    @property
    def is_geographic(self) -> bool:
        """True when the bbox is already lon/lat and needs no reprojection."""
        return self.epsg == 4326

    @property
    def dtype(self) -> str | None:
        """NumPy-style dtype string, e.g. ``float32``, or None if undetermined."""
        if self.sample_format is None or self.bits_per_sample is None:
            return None
        if self.sample_format == "undefined":
            return None
        prefix = {"uint": "uint", "int": "int", "float": "float"}[self.sample_format]
        return f"{prefix}{self.bits_per_sample}"


def parse_geotiff_header(data: bytes) -> GeoTIFFHeader:
    """Parse georeferencing from the leading bytes of a TIFF file.

    Parameters
    ----------
    data:
        The first N bytes of the file. Must reach far enough to cover the
        first image file directory and the out-of-line values its tags point
        at -- 64 KB is ample for the files this project reads.

    Raises
    ------
    ValidationError
        The bytes are not a TIFF, are truncated before the georeferencing
        tags, or lack the tags needed to place the image.
    """
    if len(data) < 8:
        raise ValidationError("too few bytes to contain a TIFF header")

    byte_order = data[:2]
    if byte_order == b"II":
        endian = "<"
    elif byte_order == b"MM":
        endian = ">"
    else:
        raise ValidationError(f"not a TIFF: byte-order marker is {byte_order!r}")

    magic = struct.unpack(f"{endian}H", data[2:4])[0]
    if magic == 43:
        raise ValidationError("BigTIFF is not supported by this reader")
    if magic != 42:
        raise ValidationError(f"not a TIFF: magic number is {magic}, expected 42")

    ifd_offset = struct.unpack(f"{endian}I", data[4:8])[0]
    tags = _read_ifd(data, ifd_offset, endian)

    try:
        width = int(tags[_TAG_IMAGE_WIDTH][0])
        height = int(tags[_TAG_IMAGE_LENGTH][0])
    except (KeyError, IndexError) as exc:
        raise ValidationError("TIFF header lacks image dimensions") from exc

    scale = tags.get(_TAG_MODEL_PIXEL_SCALE)
    tiepoint = tags.get(_TAG_MODEL_TIEPOINT)
    if not scale or not tiepoint:
        raise ValidationError(
            "not a GeoTIFF, or georeferencing tags fall outside the fetched byte "
            "range: ModelPixelScale/ModelTiepoint missing"
        )
    if len(scale) < 2 or len(tiepoint) < 6:
        raise ValidationError("GeoTIFF georeferencing tags are truncated")

    return GeoTIFFHeader(
        width=width,
        height=height,
        # ModelTiepoint is (i, j, k, x, y, z); the raster point is normally the
        # upper-left pixel, so entries 3 and 4 are its map coordinates.
        origin_x=float(tiepoint[3]),
        origin_y=float(tiepoint[4]),
        pixel_size_x=float(scale[0]),
        pixel_size_y=float(scale[1]),
        epsg=_epsg_from_geokeys(tags.get(_TAG_GEO_KEY_DIRECTORY)),
        bands=int(tags.get(_TAG_SAMPLES_PER_PIXEL, [1])[0]),
        bits_per_sample=_first_int(tags.get(_TAG_BITS_PER_SAMPLE)),
        sample_format=_SAMPLE_FORMATS.get(_first_int(tags.get(_TAG_SAMPLE_FORMAT)) or 1),
        compression=_COMPRESSIONS.get(_first_int(tags.get(_TAG_COMPRESSION)) or 1, "unknown"),
        nodata=_first_str(tags.get(_TAG_GDAL_NODATA)),
    )


def _first_int(values: list[Any] | None) -> int | None:
    try:
        return int(values[0])  # type: ignore[index]
    except (TypeError, IndexError, ValueError):
        return None


def _first_str(values: list[Any] | None) -> str | None:
    if not values:
        return None
    text = str(values[0]).strip().strip("\x00")
    return text or None


def _read_ifd(data: bytes, offset: int, endian: str) -> dict[int, list[Any]]:
    """Read one image file directory into ``{tag: [values]}``."""
    if offset + 2 > len(data):
        raise ValidationError(
            f"IFD at offset {offset} is beyond the fetched byte range "
            f"({len(data)} bytes); fetch more of the file"
        )

    count = struct.unpack(f"{endian}H", data[offset : offset + 2])[0]
    tags: dict[int, list[Any]] = {}

    for i in range(count):
        entry = offset + 2 + i * 12
        if entry + 12 > len(data):
            break  # truncated: return what was readable rather than failing

        tag, field_type, n_values = struct.unpack(f"{endian}HHI", data[entry : entry + 8])
        if field_type not in _FIELD_TYPES:
            continue

        size, code = _FIELD_TYPES[field_type]
        total = size * n_values

        if total <= 4:
            payload = data[entry + 8 : entry + 8 + total]
        else:
            value_offset = struct.unpack(f"{endian}I", data[entry + 8 : entry + 12])[0]
            if value_offset + total > len(data):
                continue  # value lies beyond the fetched range
            payload = data[value_offset : value_offset + total]

        values = _decode(payload, code, n_values, endian, field_type)
        if values is not None:
            tags[tag] = values

    return tags


def _decode(
    payload: bytes, code: str, n_values: int, endian: str, field_type: int
) -> list[Any] | None:
    """Decode a tag payload into a list of Python values."""
    try:
        if field_type == 2:  # ASCII
            return [payload.split(b"\x00")[0].decode("ascii", errors="replace")]
        if field_type in (5, 10):  # RATIONAL: numerator/denominator pairs
            parts = struct.unpack(f"{endian}{2 * n_values}{code[0]}", payload)
            return [
                parts[i] / parts[i + 1] if parts[i + 1] else 0.0 for i in range(0, len(parts), 2)
            ]
        return list(struct.unpack(f"{endian}{n_values}{code}", payload))
    except struct.error:
        return None


def _epsg_from_geokeys(geokeys: list[Any] | None) -> int | None:
    """Pull the CRS code out of a GeoKeyDirectory.

    The directory is a flat array of unsigned shorts: a 4-value header, then
    4 values per key ``(key_id, tiff_tag_location, count, value_or_offset)``.
    Only keys stored inline (``tiff_tag_location == 0``) are read, which covers
    the EPSG codes this project cares about.
    """
    if not geokeys or len(geokeys) < 4:
        return None

    n_keys = int(geokeys[3])
    for i in range(n_keys):
        base = 4 + i * 4
        if base + 4 > len(geokeys):
            break
        key_id, tag_location, _count, value = (int(v) for v in geokeys[base : base + 4])
        if tag_location != 0:
            continue
        if key_id in (_KEY_PROJECTED_CS_TYPE, _KEY_GEOGRAPHIC_TYPE) and 1024 <= value <= 32767:
            return value
    return None


def union_bbox(
    boxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    """Smallest box containing all of ``boxes``."""
    if not boxes:
        raise ValidationError("cannot take the union of zero bounding boxes")
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )
