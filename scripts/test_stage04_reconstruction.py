"""Fast synthetic safeguards for Stage 4."""

import unittest
import numpy as np
import SimpleITK as sitk
import stage04_reconstruction as stage04


class Stage04Safeguards(unittest.TestCase):
    def setUp(self):
        self.a = sitk.GetImageFromArray(
            np.asarray([[[100.0, 20.0, -10.0]]], np.float32)
        )
        self.n = sitk.GetImageFromArray(np.asarray([[[40.0, 30.0, -5.0]]], np.float32))
        self.v = sitk.GetImageFromArray(np.asarray([[[70.0, 50.0, -20.0]]], np.float32))

    def values(self, name):
        return sitk.GetArrayFromImage(
            stage04.formula_image(name, self.a, self.n, self.v)
        )

    def test_exact_signs_and_unit_weights(self):
        np.testing.assert_array_equal(
            self.values("arterial_enhancement"), [[[60, -10, -5]]]
        )
        np.testing.assert_array_equal(
            self.values("venous_enhancement"), [[[30, 20, -15]]]
        )
        np.testing.assert_array_equal(
            self.values("temporal_difference"), [[[-30, 30, -10]]]
        )
        np.testing.assert_array_equal(self.values("dark_lumen_N"), [[[-20, 40, 0]]])
        np.testing.assert_array_equal(self.values("dark_lumen_V"), [[[40, 80, -30]]])

    def test_temporal_identity(self):
        np.testing.assert_allclose(
            self.values("temporal_difference"),
            self.values("venous_enhancement") - self.values("arterial_enhancement"),
            atol=1e-6,
        )

    def test_geometry_requires_all_fields(self):
        clone = sitk.Image(self.a)
        self.assertTrue(stage04.same_geometry(self.a, clone))
        clone.SetOrigin((0.0, 0.0, 0.01))
        self.assertFalse(stage04.same_geometry(self.a, clone))

    def test_registration_state_is_not_promoted(self):
        self.assertEqual(
            stage04.product_state(("V", "A"), {"N": "review", "V": "review"})[0],
            "provisional_review",
        )
        self.assertEqual(
            stage04.product_state(("A", "N"), {"N": "fail", "V": "pass"})[0],
            "failed_experiment",
        )
        self.assertEqual(
            stage04.product_state(("V", "A"), {"N": "fail", "V": "pass"})[0], "accepted"
        )


if __name__ == "__main__":
    unittest.main()
