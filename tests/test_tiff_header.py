"""Tests for the dependency-free GeoTIFF header reader.

The parser is tested against TIFFs constructed byte by byte here rather than
against a real file, because that is the only way to pin down exactly which
byte layouts it accepts -- and because this code has to work before rasterio is
installed, so a fixture file it could be compared against would need the very
library the parser exists to avoid.
"""

from __future__ import annotations

import struct

import pytest

from satai.errors import ValidationError
from satai.geo.tiff_header import GeoTIFFHeader, parse_geotiff_header, union_bbox


def build_tiff(
    *,
    endian: str = "<",
    width: int = 512,
    height: int = 512,
    origin: tuple[float, float] = (85.0, 26.2),
    pixel_scale: tuple[float, float] = (0.0000898, 0.0000898),
    epsg: int | None = 4326,
    magic: int = 42,
    truncate_to: int | None = None,
) -> bytes:
    """Construct a minimal but structurally valid GeoTIFF header."""
    order = b"II" if endian == "<" else b"MM"

    entries: list[tuple[int, int, int, bytes | list[float] | list[int]]] = [
        (256, 4, 1, [width]),  # ImageWidth, LONG
        (257, 4, 1, [height]),  # ImageLength, LONG
        (33550, 12, 3, [pixel_scale[0], pixel_scale[1], 0.0]),  # ModelPixelScale
        (33922, 12, 6, [0.0, 0.0, 0.0, origin[0], origin[1], 0.0]),  # ModelTiepoint
    ]
    if epsg is not None:
        # GeoKeyDirectory: header (1,1,0,n) then one 4-tuple per key.
        key_id = 3072 if epsg >= 32600 else 2048
        entries.append((34735, 3, 8, [1, 1, 0, 1, key_id, 0, 1, epsg]))

    header_size = 8
    ifd_size = 2 + len(entries) * 12 + 4
    value_offset = header_size + ifd_size

    ifd = struct.pack(f"{endian}H", len(entries))
    values = b""

    for tag, field_type, count, payload in entries:
        if field_type == 12:  # DOUBLE
            raw = struct.pack(f"{endian}{count}d", *payload)  # type: ignore[arg-type]
        elif field_type == 4:  # LONG
            raw = struct.pack(f"{endian}{count}I", *payload)  # type: ignore[arg-type]
        elif field_type == 3:  # SHORT
            raw = struct.pack(f"{endian}{count}H", *payload)  # type: ignore[arg-type]
        else:
            raise AssertionError(f"unhandled field type {field_type}")

        if len(raw) <= 4:
            ifd += struct.pack(f"{endian}HHI", tag, field_type, count)
            ifd += raw.ljust(4, b"\x00")
        else:
            ifd += struct.pack(f"{endian}HHII", tag, field_type, count, value_offset + len(values))
            values += raw

    ifd += struct.pack(f"{endian}I", 0)  # no next IFD
    data = order + struct.pack(f"{endian}H", magic) + struct.pack(f"{endian}I", 8) + ifd + values
    return data[:truncate_to] if truncate_to else data


class TestParsing:
    def test_little_endian(self) -> None:
        header = parse_geotiff_header(build_tiff())
        assert header.width == 512
        assert header.height == 512
        assert header.epsg == 4326
        assert header.origin_x == pytest.approx(85.0)
        assert header.origin_y == pytest.approx(26.2)

    def test_big_endian(self) -> None:
        """Both byte orders are legal TIFF and both appear in the wild."""
        header = parse_geotiff_header(build_tiff(endian=">"))
        assert header.width == 512
        assert header.epsg == 4326

    def test_utm_projected_crs(self) -> None:
        header = parse_geotiff_header(
            build_tiff(epsg=32645, origin=(300000.0, 2900000.0), pixel_scale=(10.0, 10.0))
        )
        assert header.epsg == 32645
        assert header.is_geographic is False

    def test_missing_geokeys_yields_unknown_crs(self) -> None:
        header = parse_geotiff_header(build_tiff(epsg=None))
        assert header.epsg is None
        assert header.is_geographic is False


class TestBBox:
    def test_extends_down_and_right_from_the_tiepoint(self) -> None:
        """The tie point is the UPPER-LEFT corner, so y decreases downward.

        Getting this backwards silently mirrors every footprint in the dataset,
        which would place Indian chips in the southern hemisphere without
        raising anything.
        """
        header = GeoTIFFHeader(
            width=100,
            height=200,
            origin_x=85.0,
            origin_y=26.0,
            pixel_size_x=0.001,
            pixel_size_y=0.001,
            epsg=4326,
        )
        min_x, min_y, max_x, max_y = header.bbox
        assert min_x == pytest.approx(85.0)
        assert max_x == pytest.approx(85.1)
        assert max_y == pytest.approx(26.0)
        assert min_y == pytest.approx(25.8)

    def test_bbox_is_ordered(self) -> None:
        min_x, min_y, max_x, max_y = parse_geotiff_header(build_tiff()).bbox
        assert min_x < max_x
        assert min_y < max_y

    def test_realistic_chip_is_about_five_km(self) -> None:
        """512 px at ~0.0000898 deg (~10 m) is roughly 5 km across."""
        header = parse_geotiff_header(build_tiff())
        min_x, _, max_x, _ = header.bbox
        span_km = (max_x - min_x) * 111.32
        assert 4.5 < span_km < 5.5


class TestUnionBBox:
    def test_covers_every_input(self) -> None:
        boxes = [(85.0, 25.0, 85.1, 25.1), (86.0, 26.0, 86.2, 26.2), (84.5, 25.5, 84.6, 25.6)]
        assert union_bbox(boxes) == (84.5, 25.0, 86.2, 26.2)

    def test_single_box_is_itself(self) -> None:
        assert union_bbox([(1.0, 2.0, 3.0, 4.0)]) == (1.0, 2.0, 3.0, 4.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValidationError):
            union_bbox([])


class TestRejection:
    def test_rejects_non_tiff(self) -> None:
        with pytest.raises(ValidationError, match="byte-order marker"):
            parse_geotiff_header(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    def test_rejects_wrong_magic(self) -> None:
        with pytest.raises(ValidationError, match="magic number"):
            parse_geotiff_header(build_tiff(magic=99))

    def test_rejects_bigtiff_explicitly(self) -> None:
        """Better a clear refusal than a wrong answer from a misread layout."""
        with pytest.raises(ValidationError, match="BigTIFF"):
            parse_geotiff_header(build_tiff(magic=43))

    def test_rejects_too_few_bytes(self) -> None:
        with pytest.raises(ValidationError, match="too few bytes"):
            parse_geotiff_header(b"II*\x00")

    def test_truncated_values_are_reported_not_guessed(self) -> None:
        """A range request that stopped short must fail loudly.

        Silently returning a partial answer here would place a chip at the
        wrong coordinates, which is worse than returning nothing.
        """
        full = build_tiff()
        with pytest.raises(ValidationError, match="georeferencing"):
            parse_geotiff_header(full[: len(full) - 40])

    def test_ifd_beyond_range_is_reported(self) -> None:
        data = bytearray(build_tiff())
        data[4:8] = struct.pack("<I", 900_000)  # IFD far past the fetched bytes
        with pytest.raises(ValidationError, match="beyond the fetched byte range"):
            parse_geotiff_header(bytes(data))
