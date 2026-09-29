"""Fast synthetic safeguards for Stage 3 geometry, direction, and support."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

import stage03_registration as stage03
from stage03_acceptance import disposition, evidence_id, apply_reviews


class Stage03Safeguards(unittest.TestCase):
    def accepted_pair(self):
        pair = {
            "moving_phase": "N",
            "outputs": {"image": {"path": "image.nii.gz", "sha256": "original"}},
            "support": {"pairwise_support_fraction": 0.97},
            "transform_convention": {"roundtrip_pass": True},
        }
        review = {
            "patient": "P001",
            "moving_phase": "N",
            "decision": "accepted",
            "evidence_id": evidence_id(pair),
        }
        return pair, review

    def test_visual_acceptance_without_any_manual_anatomy(self):
        pair, review = self.accepted_pair()
        self.assertEqual(disposition(pair, review)[0], "pass")
        pair["landmarks"] = {"maximum_final_discrepancy_mm": 11.0}
        self.assertEqual(disposition(pair, review)[0], "pass")
        pair["plaque_identified"] = False
        self.assertEqual(disposition(pair, review)[0], "pass")

    def test_no_review_is_review_not_failure(self):
        pair, _ = self.accepted_pair()
        self.assertEqual(disposition(pair)[0], "review")

    def test_changed_output_invalidates_old_review(self):
        pair, review = self.accepted_pair()
        pair["outputs"]["image"]["sha256"] = "changed"
        self.assertEqual(disposition(pair, review)[0], "review")

    def test_visual_acceptance_cannot_override_gross_failure(self):
        for field, value in [
            ("error", "corrupt"),
            ("gross_technical_failures", ["wrong phase"]),
            ("support", {"pairwise_support_fraction": 0.5}),
            ("transform_convention", {"roundtrip_pass": False}),
        ]:
            pair, review = self.accepted_pair()
            pair[field] = value
            self.assertEqual(disposition(pair, review)[0], "fail")

    def test_apply_preserves_qc_and_unmeasured_accuracy(self):
        pair, review = self.accepted_pair()
        pair["qc"] = {"edge_proxy": 0.2}
        summary = {"patients": [{"patient": "P001", "pairs": [pair]}]}
        apply_reviews(summary, {"pairs": [review]})
        self.assertEqual(pair["qc"], {"edge_proxy": 0.2})
        self.assertIsNone(pair["quantitative_landmark_registration_accuracy_mm"])
        self.assertEqual(pair["reviewer_visual_qc"], "accepted")

    def test_cohort_acceptance_is_idempotent_and_failure_blocks_patient(self):
        import copy

        patients, reviews = [], []
        for pid in ("P001", "P002", "P003", "P004", "P005"):
            pairs = []
            for phase in ("N", "V"):
                pair, review = self.accepted_pair()
                pair["moving_phase"] = phase
                review.update(patient=pid, moving_phase=phase)
                pairs.append(pair)
                reviews.append(review)
            patients.append({"patient": pid, "pairs": pairs})
        summary = {"patients": patients}
        apply_reviews(summary, {"pairs": reviews})
        snapshot = copy.deepcopy(summary)
        apply_reviews(summary, {"pairs": reviews})
        self.assertEqual(summary, snapshot)
        self.assertEqual(summary["status"], "COMPLETE")
        self.assertTrue(all(p["state"] == "pass" for p in patients))
        patients[0]["pairs"][0]["gross_technical_failures"] = ["severe artifact"]
        apply_reviews(summary, {"pairs": reviews})
        self.assertEqual(patients[0]["state"], "fail")
        self.assertEqual(summary["status"], "COMPLETE_WITH_FAILURES")

    def image(self, size=(20, 18, 16)):
        image = sitk.Image(size, sitk.sitkInt16)
        image.SetSpacing((0.6, 0.7, 1.1))
        image.SetOrigin((10.0, -20.0, 30.0))
        return image

    def test_crop_origin_is_physical_index(self):
        image = self.image()
        mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
        mask.CopyInformation(image)
        array = sitk.GetArrayFromImage(mask)
        array[3:10, 4:12, 5:15] = 1
        mask = sitk.GetImageFromArray(array)
        mask.CopyInformation(image)
        cropped, record = stage03.crop_to_mask_bbox(image, mask)
        self.assertTrue(record["origin_verified_from_native_A_index"])
        self.assertTrue(
            np.allclose(
                cropped.GetOrigin(), image.TransformIndexToPhysicalPoint((5, 4, 3))
            )
        )

    def test_support_excludes_every_boundary_face(self):
        image = self.image()
        support = sitk.GetArrayFromImage(stage03.conservative_native_support(image))
        self.assertFalse(support[0].any())
        self.assertFalse(support[-1].any())
        self.assertFalse(support[:, 0].any())
        self.assertFalse(support[:, -1].any())
        self.assertFalse(support[:, :, 0].any())
        self.assertFalse(support[:, :, -1].any())
        self.assertTrue(support[1:-1, 1:-1, 1:-1].all())

    def test_fixed_to_moving_transform_direction(self):
        transform = sitk.Euler3DTransform()
        transform.SetTranslation((2.0, -3.0, 4.0))
        fixed = {"C1": (1.0, 2.0, 3.0), "T1": (4.0, 5.0, 6.0)}
        moving = {
            name: transform.TransformPoint(point) for name, point in fixed.items()
        }
        landmarks, maximum = stage03.landmark_discrepancies(transform, fixed, moving)
        self.assertLess(maximum, 1e-10)
        check = stage03.direction_verification(transform, fixed, moving)
        self.assertTrue(check["roundtrip_pass"])
        self.assertLess(check["mean_correct_direction_landmark_error_mm"], 1e-10)
        self.assertGreater(check["mean_deliberately_wrong_direction_error_mm"], 1.0)

    def test_written_geometry_roundtrip(self):
        image = sitk.Cast(self.image(), sitk.sitkFloat32)
        with tempfile.TemporaryDirectory() as directory:
            record = stage03.write_image(image, Path(directory) / "image.nii.gz")
        self.assertEqual(record["size_xyz"], list(image.GetSize()))
        self.assertTrue(np.allclose(record["origin_lps_mm"], image.GetOrigin()))

    def test_matching_plane_extraction(self):
        array = np.arange(16 * 18 * 20, dtype=np.int16).reshape(16, 18, 20)
        image = sitk.GetImageFromArray(array)
        self.assertTrue(
            np.array_equal(stage03.extract_matching_plane(image, 2, 3), array[3])
        )
        self.assertTrue(
            np.array_equal(stage03.extract_matching_plane(image, 1, 4), array[:, 4, :])
        )
        self.assertTrue(
            np.array_equal(stage03.extract_matching_plane(image, 0, 5), array[:, :, 5])
        )

    def test_flattened_composite_serializes_and_preserves_mapping(self):
        inner = sitk.CompositeTransform(3)
        inner.AddTransform(sitk.TranslationTransform(3, (1.0, 2.0, 3.0)))
        outer = sitk.CompositeTransform(3)
        outer.AddTransform(inner)
        point = (4.0, 5.0, 6.0)
        expected = outer.TransformPoint(point)
        outer.FlattenTransform()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transform.hdf"
            sitk.WriteTransform(outer, str(path))
            reread = sitk.ReadTransform(str(path))
        self.assertTrue(np.allclose(reread.TransformPoint(point), expected))


if __name__ == "__main__":
    unittest.main()
