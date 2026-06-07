"""Unit tests for HorizonSurface.from_ts (GOCAD TSurf parser)."""
import tempfile
from pathlib import Path

import pytest
import torch

from horizons.data.mesh import HorizonSurface


TINY_TS_CONTENT = """GOCAD TSurf 1
HEADER {
name: test_surface
}
GOCAD_ORIGINAL_COORDINATE_SYSTEM
NAME from_XYZ
PROJECTION Unknown
DATUM Unknown
AXIS_NAME X Y Z
AXIS_UNIT m m m
ZPOSITIVE Depth
END_ORIGINAL_COORDINATE_SYSTEM
TFACE
PVRTX 1 0.0 0.0 100.0
PVRTX 2 1.0 0.0 110.0
PVRTX 3 0.0 1.0 120.0
PVRTX 4 1.0 1.0 130.0
TRGL 1 2 3
TRGL 2 4 3
END
"""


class TestFromTs:
    def test_parses_minimal_file(self, tmp_path: Path) -> None:
        """A 4-vertex, 2-triangle file parses correctly."""
        ts_path = tmp_path / "tiny.ts"
        ts_path.write_text(TINY_TS_CONTENT)
        surf = HorizonSurface.from_ts(ts_path)

        assert surf.n_vertices == 4
        assert surf.n_faces == 2
        assert surf.V.dtype == torch.float32
        assert surf.F.dtype == torch.int64
        # Triangles should be 0-indexed
        assert surf.F.min() == 0
        assert surf.F.max() == 3

    def test_surface_id_from_filename(self, tmp_path: Path) -> None:
        ts_path = tmp_path / "my_horizon.ts"
        ts_path.write_text(TINY_TS_CONTENT)
        surf = HorizonSurface.from_ts(ts_path)
        assert surf.surface_id == "my_horizon"

    def test_explicit_surface_id_overrides(self, tmp_path: Path) -> None:
        ts_path = tmp_path / "raw_name.ts"
        ts_path.write_text(TINY_TS_CONTENT)
        surf = HorizonSurface.from_ts(
            ts_path,
            surface_id="my_chosen_id",
            reservoir_id="R_test",
        )
        assert surf.surface_id == "my_chosen_id"
        assert surf.reservoir_id == "R_test"

    def test_coordinates_preserved(self, tmp_path: Path) -> None:
        """Vertex coordinates from the file should appear unchanged."""
        ts_path = tmp_path / "coords.ts"
        ts_path.write_text(TINY_TS_CONTENT)
        surf = HorizonSurface.from_ts(ts_path)
        # First vertex should be (0, 0, 100)
        assert torch.allclose(surf.V[0], torch.tensor([0.0, 0.0, 100.0]))
        # Last vertex should be (1, 1, 130)
        assert torch.allclose(surf.V[3], torch.tensor([1.0, 1.0, 130.0]))

    def test_vrtx_keyword_supported(self, tmp_path: Path) -> None:
        """Some GOCAD files use VRTX instead of PVRTX. Both should work."""
        content = TINY_TS_CONTENT.replace("PVRTX", "VRTX")
        ts_path = tmp_path / "vrtx_variant.ts"
        ts_path.write_text(content)
        surf = HorizonSurface.from_ts(ts_path)
        assert surf.n_vertices == 4

    def test_non_contiguous_vertex_ids(self, tmp_path: Path) -> None:
        """GOCAD IDs may be non-contiguous; the loader should remap them."""
        content = """GOCAD TSurf 1
            TFACE
            PVRTX 5 0.0 0.0 100.0
            PVRTX 7 1.0 0.0 110.0
            PVRTX 10 0.0 1.0 120.0
            TRGL 5 7 10
            END
        """
        ts_path = tmp_path / "noncontig.ts"
        ts_path.write_text(content)
        surf = HorizonSurface.from_ts(ts_path)
        assert surf.n_vertices == 3
        # IDs remapped to 0, 1, 2
        assert surf.F.min() == 0
        assert surf.F.max() == 2

    def test_missing_vertex_referenced_by_triangle_raises(
        self, tmp_path: Path
    ) -> None:
        """A triangle referencing an undefined vertex should raise."""
        content = """GOCAD TSurf 1
            TFACE
            PVRTX 1 0 0 0
            PVRTX 2 1 0 0
            TRGL 1 2 999
            END
        """
        ts_path = tmp_path / "bad.ts"
        ts_path.write_text(content)
        with pytest.raises(ValueError, match="missing vertex"):
            HorizonSurface.from_ts(ts_path)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        ts_path = tmp_path / "empty.ts"
        ts_path.write_text("GOCAD TSurf 1\nEND\n")
        with pytest.raises(ValueError, match="No PVRTX"):
            HorizonSurface.from_ts(ts_path)
