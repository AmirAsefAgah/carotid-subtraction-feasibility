"""Small independent fixtures exercise orientation/conflict and scaling safeguards."""

import gzip
import struct
import tempfile
import unittest
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import stage01_audit as audit


class Safeguards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "test.nii.gz"
        data = np.arange(5 * 7 * 9, dtype=np.int16).reshape(5, 7, 9) - 100
        self.image = sitk.GetImageFromArray(data)
        self.image.SetSpacing((0.6, 0.7, 1.3))
        self.image.SetOrigin((-12.5, 37.2, -80.1))
        sitk.WriteImage(self.image, str(self.path))

    def tearDown(self):
        self.temp.cleanup()

    def change_header(self, fmt, at, *values):
        with gzip.open(self.path, "rb") as f:
            data = bytearray(f.read())
        struct.pack_into("<" + fmt, data, at, *values)
        with gzip.open(self.path, "wb") as f:
            f.write(data)

    def test_native_left_right_and_lps_ras_conversion(self):
        hdr = audit.header(self.path)
        geom = audit.geometry(
            hdr, self.path, {"qform_sform_corner_tolerance_mm": 0.001}
        )
        raw = audit.read_raw(self.path, hdr)
        np.testing.assert_array_equal(
            raw.transpose(2, 1, 0), sitk.GetArrayFromImage(self.image)
        )
        point = np.array([[8, 3, 4]])
        actual = audit.transform(np.asarray(geom["affine_ras"]), point)[0]
        expected = np.asarray(self.image.TransformIndexToPhysicalPoint((8, 3, 4))) * [
            -1,
            -1,
            1,
        ]
        np.testing.assert_allclose(actual, expected, atol=1e-5)
        self.assertEqual(geom["axis_codes"], "LPS")

    def test_qform_sform_origin_conflict_is_blocked(self):
        self.change_header("f", 292, 120.0)
        with self.assertRaisesRegex(ValueError, "Conflicting qform/sform"):
            audit.geometry(
                audit.header(self.path),
                self.path,
                {"qform_sform_corner_tolerance_mm": 0.001},
            )

    def test_nontrivial_nifti_scaling_read_once(self):
        self.change_header("2f", 112, 2.0, -1024.0)
        hdr = audit.header(self.path)
        raw = audit.read_raw(self.path, hdr)
        expected = raw.astype(float) * 2.0 - 1024.0
        actual = sitk.GetArrayFromImage(sitk.ReadImage(str(self.path))).transpose(
            2, 1, 0
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertNotEqual(float(expected[0, 0, 0]), float(expected[0, 0, 0]) - 1024.0)

    def test_mirrored_direction_preserved_not_canonicalized(self):
        self.image.SetDirection((-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0))
        sitk.WriteImage(self.image, str(self.path))
        hdr = audit.header(self.path)
        geom = audit.geometry(
            hdr, self.path, {"qform_sform_corner_tolerance_mm": 0.001}
        )
        np.testing.assert_allclose(
            geom["direction_lps"], self.image.GetDirection(), atol=1e-6
        )
        with self.assertRaisesRegex(ValueError, "reviewed orientation"):
            audit.contact_sheet(
                audit.read_raw(self.path, hdr),
                hdr,
                geom,
                "TEST",
                "A",
                Path(self.temp.name),
                {},
            )


if __name__ == "__main__":
    unittest.main()
