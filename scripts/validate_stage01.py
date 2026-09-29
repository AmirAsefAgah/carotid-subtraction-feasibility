"""Validate saved Stage 1 geometry and source-integrity records."""

import csv
import hashlib
import json
from pathlib import Path
from PIL import Image

out = Path(__file__).resolve().parents[1]
stage = json.loads((out / "stage_01.json").read_text())
with (out / "manifest.csv").open(encoding="utf-8-sig") as f:
    manifest = list(csv.DictReader(f))
expected = {(f"P{i:03}", p) for i in range(1, 6) for p in ["N", "A", "V"]}
assert (
    len(manifest) == 15 and {(r["patient"], r["phase"]) for r in manifest} == expected
)
assert stage["accounted_inputs"] == 15 and stage["readable_inputs"] == 15
assert stage["contact_sheets_created"] == 15
assert stage["originals_unchanged"] is True
assert stage["settings"]["resample_inputs"] is False
assert stage["settings"]["normalize_intensities"] is False
assert stage["settings"]["apply_dicom_rescale_to_nifti"] is False
assert not list(out.rglob("*.nii*")), "Unexpected generated NIfTI in Stage 1 output"
for r in manifest:
    audit = json.loads(
        (out / "patients" / r["patient"] / f"{r['phase']}_audit.json").read_text()
    )
    assert audit["intensity"]["nonfinite_count"] == 0
    assert audit["intensity"]["simpleitk_sample_max_error"] == 0
    assert audit["geometry"]["axis_codes"] == "LPS"
    assert audit["geometry"]["qform_sform_max_corner_error_mm"] == 0
    assert audit["dicom_verification"]["max_sampled_hu_error"] == 0
    assert audit["dicom_verification"]["sampled_voxels"] == 1280
    assert audit["dicom_verification"]["dicom_intercept_must_not_be_reapplied"]
    assert r["injection_relative_delay_seconds"] == "unknown"
    assert r["kvp"] == "120" and r["kernel"] == "SB"
    assert (
        float(r["slice_thickness_mm"]) == 1
        and float(r["dicom_spacing_between_slices_mm"]) == 0.5
    )
    assert r["phase_identity_evidence"] not in ["", "[]"]
    assert len(audit["contact_sheet"]["slices"]) == 21
    with Image.open(r["contact_sheet"]) as im:
        im.verify()
    reference = json.loads(
        (out / "patients" / r["patient"] / "reference_grid.json").read_text()
    )
    if r["phase"] == "A":
        assert reference["affine_ras"] == audit["geometry"]["affine_ras"]
        assert reference["shape"] == audit["nifti_header"]["shape"]
review_path = out / "visual_review.json"
if review_path.exists():
    review = json.loads(review_path.read_text())
    assert {(r["patient"], r["phase"]) for r in review["images"]} == expected
    for row in review["images"]:
        assert (
            hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() == row["sha256"]
        )
    if review.get("human_review", {}).get("status") == "completed":
        human = review["human_review"]
        eligible = "eligible_for_registration_trial"
        flagged = {"P003/N", "P005/N", "P005/A", "P005/V"}
        assert {(r["patient"], r["phase"]) for r in human["inputs"]} == expected
        assert human["registration_input_status"] == eligible
        assert stage["review_status"]["human_review"] == "completed"
        assert stage["geometry_review_inputs"] == [] and stage["blocked_inputs"] == []
        assert set(stage["reviewed_geometry_inputs"]) == flagged
        assert stage["handoff"]["ready_for_stage_02"] is True
        assert stage["handoff"]["remaining_stage_01_blockers"] == []
        assert stage["handoff"]["coverage_policy"] == human["coverage_policy"]
        assert (
            stage["handoff"]["registration_qc_watch_items"]
            == human["registration_qc_watch_items"]
        )
        assert human["P003_V_selection"]["replace_with_later_acquisition"] is False
        observed_flags = set()
        for row in manifest:
            audit = json.loads(
                (
                    out / "patients" / row["patient"] / f"{row['phase']}_audit.json"
                ).read_text()
            )
            assert (
                row["registration_input_status"]
                == audit["registration_input_status"]
                == eligible
            )
            assert (
                row["human_review_status"]
                == audit["human_review"]["status"]
                == "completed"
            )
            assert audit["contact_sheet"]["human_review_status"] == "completed"
            assert (
                not audit["errors"]
                and not audit["dicom_verification"]["geometry_block"]
            )
            if audit["dicom_verification"]["geometry_review"]:
                observed_flags.add(row["patient"] + "/" + row["phase"])
                assert (
                    audit["human_review"]["geometry_accepted_for_registration_trial"]
                    is True
                )
            if (row["patient"], row["phase"]) == ("P003", "V"):
                assert row["source_path"] == human["P003_V_selection"]["source_path"]
                assert row["series_uid"] == human["P003_V_selection"]["series_uid"]
                assert row["series_number"] == "7"
        assert observed_flags == flagged, (
            "Raw measured geometry flags must be preserved"
        )
        assert all(r["human_review_status"] == "completed" for r in review["images"])
        with (out / "tables/stage01_patient_summary.csv").open(
            encoding="utf-8-sig"
        ) as f:
            patients = list(csv.DictReader(f))
        assert len(patients) == 5
        for row in patients:
            saved = next(r for r in stage["patients"] if r["patient"] == row["patient"])
            assert row["human_review"] == saved["human_review"] == "completed"
            for key in ["N_to_A_status", "V_to_A_status"]:
                assert row[key] == saved[key] == eligible
            for key in [
                "unregistered_common_z_span_mm",
                "A_inferior_z_loss_mm",
                "A_superior_z_loss_mm",
                "N_to_A_start_seconds",
                "A_to_V_start_seconds",
            ]:
                assert float(row[key]) == saved[key]
        print(
            "PASS: investigator review completed for 15 inputs; all 10 pairs eligible; P003/V deliberate; four measured geometry flags retained with human acceptance; coverage/QC handoff consistent."
        )
print(
    "PASS: 15 manifest rows, 15 complete audits, 15 valid contact sheets, 5 native A grids; geometry, HU, metadata, orientation, scaling and Stage 1 scope verified."
)
