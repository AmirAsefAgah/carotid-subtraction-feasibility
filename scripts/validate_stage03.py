"""Validate registration geometry, transform direction, and support."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from stage03_acceptance import disposition, artifacts
from stage04_reconstruction import sha256


from study_paths import ROOT

PATIENTS = ("P001", "P002", "P003", "P004", "P005")


def same_geometry(a: sitk.Image, b: sitk.Image, atol: float = 1e-6) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol)
    )


def foreground(image: sitk.Image) -> int:
    stats = sitk.StatisticsImageFilter()
    stats.Execute(sitk.Cast(image > 0, sitk.sitkUInt8))
    return int(round(stats.GetSum()))


def subset_violations(child: sitk.Image, parent: sitk.Image) -> int:
    violation = sitk.And(
        sitk.Cast(child > 0, sitk.sitkUInt8), sitk.Cast(parent == 0, sitk.sitkUInt8)
    )
    return foreground(violation)


def main() -> int:
    summary = json.loads((ROOT / "stage_03.json").read_text(encoding="utf-8"))
    review_path = ROOT / "config" / "stage03_reviewer_visual_qc.json"
    reviews = json.loads(review_path.read_text()) if review_path.exists() else {}
    review_lookup = {
        (r["patient"], r["moving_phase"]): r for r in reviews.get("pairs", [])
    }
    errors: list[str] = []
    checks: list[dict] = []
    by_patient = {p["patient"]: p for p in summary.get("patients", [])}
    if set(by_patient) != set(PATIENTS):
        errors.append(f"Expected five patients, found {sorted(by_patient)}")

    for patient in PATIENTS:
        if patient not in by_patient:
            continue
        record = by_patient[patient]
        pairs = {p.get("moving_phase"): p for p in record.get("pairs", [])}
        if set(pairs) != {"N", "V"}:
            errors.append(f"{patient}: expected N and V pairs, found {sorted(pairs)}")
            continue
        common_record = record.get("common_support") or {}
        common_outputs = common_record.get("outputs", {})
        required_common = {
            "A_original_hu_crop",
            "A_body_crop",
            "requested_roi_crop",
            "common_three_phase_support",
        }
        if set(common_outputs) != required_common:
            errors.append(f"{patient}: incomplete common outputs")
            continue
        a_crop = sitk.ReadImage(common_outputs["A_original_hu_crop"]["path"])
        a_body = sitk.ReadImage(common_outputs["A_body_crop"]["path"])
        roi = sitk.ReadImage(common_outputs["requested_roi_crop"]["path"])
        common = sitk.ReadImage(common_outputs["common_three_phase_support"]["path"])
        if a_crop.GetPixelID() != sitk.sitkFloat32:
            errors.append(f"{patient}: A crop is not float32")
        for name, image in (("A body", a_body), ("ROI", roi), ("common", common)):
            if not same_geometry(a_crop, image):
                errors.append(f"{patient}: {name} geometry differs from A crop")
        native_a = sitk.ReadImage(pairs["N"]["inputs"]["fixed_native_A"])
        crop = common_record["crop"]
        expected_origin = native_a.TransformIndexToPhysicalPoint(
            tuple(crop["index_xyz"])
        )
        if not np.allclose(a_crop.GetOrigin(), expected_origin, atol=1e-6):
            errors.append(
                f"{patient}: cropped origin is not native-A crop-index origin"
            )

        pair_masks = {}
        pair_check = {"patient": patient, "pairs": {}}
        for phase in ("N", "V"):
            pair = pairs[phase]
            expected_state, _ = disposition(pair, review_lookup.get((patient, phase)))
            if pair["state"] != expected_state:
                errors.append(
                    f"{patient} {phase}->A: acceptance state inconsistent with reviewer evidence and technical safeguards"
                )
            if pair["state"] == "pass":
                for artifact in [
                    pair["transforms"]["final"],
                    *artifacts(pair["outputs"]),
                ]:
                    if sha256(Path(artifact["path"])) != artifact["sha256"]:
                        errors.append(
                            f"{patient} {phase}->A: accepted artifact fingerprint changed: {artifact['path']}"
                        )
            if pair.get("quantitative_landmark_registration_accuracy_mm") is not None:
                errors.append(
                    f"{patient} {phase}->A: manual accuracy claimed without manual validation"
                )
            registered = sitk.ReadImage(
                pair["outputs"]["registered_original_hu"]["path"]
            )
            valid = sitk.ReadImage(pair["outputs"]["valid_sampling_mask"]["path"])
            support = sitk.ReadImage(pair["outputs"]["pairwise_support_mask"]["path"])
            pair_masks[phase] = support
            if registered.GetPixelID() != sitk.sitkFloat32:
                errors.append(f"{patient} {phase}->A: registered image is not float32")
            for name, image in (
                ("registered", registered),
                ("valid", valid),
                ("pair support", support),
            ):
                if not same_geometry(a_crop, image):
                    errors.append(
                        f"{patient} {phase}->A: {name} geometry differs from A crop"
                    )
            violations = subset_violations(support, valid)
            if violations:
                errors.append(
                    f"{patient} {phase}->A: pair support has {violations} voxels outside native support"
                )
            transform = sitk.ReadTransform(pair["transforms"]["final"]["path"])
            if len(transform.GetParameters()) != 6:
                errors.append(
                    f"{patient} {phase}->A: final transform is not 6-parameter rigid"
                )
            point = a_crop.TransformContinuousIndexToPhysicalPoint(
                tuple((np.asarray(a_crop.GetSize()) - 1) / 2)
            )
            roundtrip = np.linalg.norm(
                np.asarray(
                    transform.GetInverse().TransformPoint(
                        transform.TransformPoint(point)
                    )
                )
                - point
            )
            if roundtrip >= 1e-5 or not pair["transform_convention"].get(
                "roundtrip_pass"
            ):
                errors.append(
                    f"{patient} {phase}->A: transform direction roundtrip failed"
                )
            for key, figure in pair["qc"]["figures"].items():
                if key.endswith("sha256"):
                    continue
                if isinstance(figure, str) and not Path(figure).exists():
                    errors.append(f"{patient} {phase}->A: missing QC figure {figure}")
            pair_check["pairs"][phase] = {
                "state": pair["state"],
                "selection": pair["registration"]["final_selection"],
                "support_fraction_recorded": pair["support"][
                    "pairwise_support_fraction"
                ],
                "support_voxels_recomputed": foreground(support),
                "support_outside_valid_voxels": violations,
                "transform_roundtrip_error_mm": float(roundtrip),
                "automated_maximum_C1_T1_centroid_discrepancy_mm": pair.get(
                    "landmarks", {}
                ).get("maximum_final_discrepancy_mm"),
                "reviewer_visual_qc": pair.get("reviewer_visual_qc", "pending"),
                "manual_landmarks_required": False,
                "manual_landmark_accuracy_mm": None,
            }

        n_violation = subset_violations(common, pair_masks["N"])
        v_violation = subset_violations(common, pair_masks["V"])
        body_violation = subset_violations(common, a_body)
        roi_violation = subset_violations(common, roi)
        if any((n_violation, v_violation, body_violation, roi_violation)):
            errors.append(
                f"{patient}: common support is not a subset of every required mask"
            )
        pair_check["common_support"] = {
            "fraction_recorded": common_record["common_three_phase_support_fraction"],
            "voxels_recomputed": foreground(common),
            "subset_violations": {
                "N_pair": n_violation,
                "V_pair": v_violation,
                "A_body": body_violation,
                "requested_roi": roi_violation,
            },
        }
        checks.append(pair_check)

    report = {
        "stage": 3,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "scope": "artifact integrity, geometry, mask containment, rigid transform dimensionality/direction and consistency with recorded reviewer visual acceptance; no manual landmark accuracy validation",
        "errors": errors,
        "patients": checks,
    }
    path = ROOT / "logs" / "stage03_validation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"status": report["status"], "errors": errors, "report": str(path)},
            indent=2,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
