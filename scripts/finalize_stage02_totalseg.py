"""Validate C1/T1 masks and finalize the physical neck crop."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


from study_paths import ROOT
from study_paths import SOURCE_ROOT

PATIENTS = ("P001", "P002", "P003", "P004", "P005")
PHASES = ("A", "N", "V")
LABELS = ("vertebrae_C1", "vertebrae_T1")
LONGITUDINAL_MARGIN_MM = 15.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mask_record(path: Path, source: nib.Nifti1Image) -> tuple[dict, np.ndarray]:
    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.uint8) > 0
    geometry_match = image.shape == source.shape and np.allclose(
        image.affine, source.affine, atol=1e-5
    )
    if not geometry_match:
        raise RuntimeError(f"Mask geometry does not match source: {path}")
    locations = np.argwhere(data)
    if not len(locations):
        raise RuntimeError(f"TotalSegmentator produced an empty mask: {path}")
    low = locations.min(axis=0).astype(float) - 0.5
    high = locations.max(axis=0).astype(float) + 0.5
    corners = np.asarray(
        [
            [x, y, z, 1.0]
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ]
    )
    ras = (source.affine @ corners.T).T[:, :3]
    record = {
        "path": str(path),
        "sha256": sha256(path),
        "voxel_count": int(data.sum()),
        "volume_mm3": float(data.sum() * np.prod(source.header.get_zooms()[:3])),
        "geometry_matches_source": True,
        "bbox_index_inclusive": {
            "start_xyz": locations.min(axis=0).tolist(),
            "end_xyz": locations.max(axis=0).tolist(),
        },
        "ras_edge_bounds_mm": {
            "min_xyz": ras.min(axis=0).tolist(),
            "max_xyz": ras.max(axis=0).tolist(),
        },
    }
    return record, data


def save_qc(
    patient: str, phase: str, source: nib.Nifti1Image, c1: np.ndarray, t1: np.ndarray
) -> Path:
    union = c1 | t1
    center = np.rint(np.argwhere(union).mean(axis=0)).astype(int)
    x, y, _ = center
    c1_z = int(np.rint(np.argwhere(c1)[:, 2].mean()))
    t1_z = int(np.rint(np.argwhere(t1)[:, 2].mean()))
    panels = [
        (
            np.asarray(source.dataobj[:, :, c1_z]).T,
            c1[:, :, c1_z].T,
            t1[:, :, c1_z].T,
            "Axial C1",
        ),
        (
            np.asarray(source.dataobj[:, :, t1_z]).T,
            c1[:, :, t1_z].T,
            t1[:, :, t1_z].T,
            "Axial T1",
        ),
        (
            np.asarray(source.dataobj[:, y, :]).T,
            c1[:, y, :].T,
            t1[:, y, :].T,
            "Coronal",
        ),
        (
            np.asarray(source.dataobj[x, :, :]).T,
            c1[x, :, :].T,
            t1[x, :, :].T,
            "Sagittal",
        ),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(17, 5), constrained_layout=True)
    for axis, (ct, c1_view, t1_view, title) in zip(axes, panels):
        axis.imshow(
            ct,
            cmap="gray",
            vmin=-150,
            vmax=600,
            origin="lower",
            interpolation="nearest",
        )
        if c1_view.any():
            axis.contour(c1_view, levels=[0.5], colors=["lime"], linewidths=0.8)
        if t1_view.any():
            axis.contour(t1_view, levels=[0.5], colors=["cyan"], linewidths=0.8)
        axis.set_title(f"{title}: C1 lime, T1 cyan")
        axis.set_axis_off()
    figure.suptitle(f"{patient} {phase} — TotalSegmentator C1/T1 machine QC")
    output = ROOT / "figures" / f"{patient}_{phase}_stage02_C1_T1_qc.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def make_longitudinal_roi(
    source: nib.Nifti1Image, inferior: float, superior: float
) -> np.ndarray:
    nx, ny, nz = source.shape
    affine = source.affine
    xx = np.arange(nx, dtype=np.float32)[:, None]
    yy = np.arange(ny, dtype=np.float32)[None, :]
    roi = np.zeros(source.shape, dtype=np.uint8)
    for z in range(nz):
        ras_z = affine[2, 0] * xx + affine[2, 1] * yy + affine[2, 2] * z + affine[2, 3]
        roi[:, :, z] = ((ras_z >= inferior) & (ras_z <= superior)).astype(np.uint8)
    return roi


def main() -> None:
    stage_path = ROOT / "stage_02.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    patients_by_id = {item["patient"]: item for item in stage["patients"]}
    summary_rows = []

    for patient in PATIENTS:
        patient_record = patients_by_id[patient]
        phases_by_id = {item["phase"]: item for item in patient_record["phases"]}
        phase_results = {}
        for phase in PHASES:
            source_path = SOURCE_ROOT / patient / "nifti" / f"{phase}.nii.gz"
            source = nib.load(str(source_path))
            output_dir = (
                ROOT / "patients" / patient / "stage_02" / "totalsegmentator" / phase
            )
            report_path = output_dir / "run_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            labels, arrays = {}, {}
            for label in LABELS:
                labels[label], arrays[label] = mask_record(
                    output_dir / f"{label}.nii.gz", source
                )
            qc_path = save_qc(
                patient, phase, source, arrays["vertebrae_C1"], arrays["vertebrae_T1"]
            )
            result = {
                "status": "COMPLETE_NONEMPTY_NATIVE_GRID_MASKS",
                "software": {
                    "name": "TotalSegmentator",
                    "version": report["totalsegmentator_version"],
                },
                "task": report["task"],
                "roi_subset": report["roi_subset"],
                "device": report["device"],
                "runtime_seconds": report["runtime_seconds"],
                "run_report": str(report_path),
                "labels": labels,
                "machine_qc_preview": str(qc_path),
            }
            phases_by_id[phase]["totalsegmentator_C1_T1"] = result
            phases_by_id[phase]["coverage_flags"]["C1_T1_masks_available"] = True
            phase_results[phase] = result
            summary_rows.append(
                {
                    "patient": patient,
                    "phase": phase,
                    "c1_voxels": labels["vertebrae_C1"]["voxel_count"],
                    "t1_voxels": labels["vertebrae_T1"]["voxel_count"],
                    "runtime_seconds": report["runtime_seconds"],
                    "geometry_match": True,
                    "qc_preview": str(qc_path),
                }
            )

        arterial = phase_results["A"]["labels"]
        c1_superior = arterial["vertebrae_C1"]["ras_edge_bounds_mm"]["max_xyz"][2]
        t1_inferior = arterial["vertebrae_T1"]["ras_edge_bounds_mm"]["min_xyz"][2]
        length = c1_superior - t1_inferior
        if length <= 0:
            raise RuntimeError(
                f"Invalid C1/T1 physical ordering for {patient}: L={length}"
            )
        requested_inferior = t1_inferior - LONGITUDINAL_MARGIN_MM
        requested_superior = c1_superior + LONGITUDINAL_MARGIN_MM
        arterial_source = nib.load(str(SOURCE_ROOT / patient / "nifti" / "A.nii.gz"))
        roi_data = make_longitudinal_roi(
            arterial_source, requested_inferior, requested_superior
        )
        roi_path = (
            ROOT
            / "patients"
            / patient
            / "stage_02"
            / f"{patient}_A_C1_T1_plus15mm_longitudinal_roi.nii.gz"
        )
        nib.save(
            nib.Nifti1Image(roi_data, arterial_source.affine, arterial_source.header),
            str(roi_path),
        )
        covered = np.argwhere(roi_data)
        patient_record["roi"].update(
            {
                "definition": "superior edge C1 to inferior edge T1; extend 15 mm superiorly and inferiorly",
                "status": "AUTOMATIC_C1_T1_LABELS_AVAILABLE_ROI_DEFINED",
                "c1_superior_edge_ras_mm": c1_superior,
                "t1_inferior_edge_ras_mm": t1_inferior,
                "L_mm": length,
                "extended_superior_bound_ras_mm": requested_superior,
                "extended_inferior_bound_ras_mm": requested_inferior,
                "margin_mm_each_end": LONGITUDINAL_MARGIN_MM,
                "longitudinal_roi_mask": str(roi_path),
                "longitudinal_roi_mask_sha256": sha256(roi_path),
                "longitudinal_roi_index_bounds_inclusive": {
                    "start_xyz": covered.min(axis=0).tolist(),
                    "end_xyz": covered.max(axis=0).tolist(),
                },
                "source": "TotalSegmentator C1 and T1 masks on native arterial CT",
                "coverage_flags": {
                    "requested_longitudinal_roi_available": True,
                    "C1_automatic_label_available": True,
                    "T1_automatic_label_available": True,
                    "carotid_cut_check": "PENDING_HUMAN_REVIEW",
                },
            }
        )
        patient_record["status"] = "STAGE02_MACHINE_COMPLETE_PENDING_HUMAN_REVIEW"
        patient_record["human_review_status"] = (
            "PENDING_C1_T1_LABEL_AND_CAROTID_COVERAGE_REVIEW"
        )
        patient_record["pending_review_items"] = [
            "Human reviewer: confirm C1 and T1 labels on the generated overlays.",
            "Human reviewer: confirm body-envelope contours exclude table/pads and preserve both carotid bifurcations.",
            "Human reviewer: confirm the requested longitudinal ROI does not cut the carotid target region.",
        ]
        per_patient_path = (
            ROOT / "patients" / patient / "stage_02" / f"{patient}_stage02_roi.json"
        )
        per_patient_path.write_text(
            json.dumps(patient_record, indent=2), encoding="utf-8"
        )

    stage.setdefault("settings", {}).update(
        {
            "longitudinal_margin_method": "fixed_physical_distance",
            "longitudinal_margin_mm_each_end": LONGITUDINAL_MARGIN_MM,
        }
    )
    stage.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "STAGE02_MACHINE_COMPLETE_PENDING_HUMAN_REVIEW",
            "human_review_status": "PENDING_C1_T1_LABEL_BODY_CONTOUR_AND_CAROTID_COVERAGE_REVIEW",
            "machine_review_status": "COMPLETE_30_NONEMPTY_C1_T1_MASKS_AND_15_C1_T1_QC_PREVIEWS",
            "patients": [patients_by_id[patient] for patient in PATIENTS],
            "totalsegmentator_check": {
                "installed": True,
                "version": "2.18.0",
                "task": "total",
                "roi_subset": list(LABELS),
                "device": "cpu",
                "all_native_grid_masks_nonempty": True,
            },
            "manual_landmark_request": "RESOLVED_BY_AUTOMATIC_C1_T1_SEGMENTATION",
            "pending_review_items": [
                "Human review of 15 C1/T1 overlays and body masks is pending.",
                "Human confirmation that each arterial C1–T1 plus 15 mm ROI preserves the carotid target region is pending.",
            ],
        }
    )
    stage_path.write_text(json.dumps(stage, indent=2), encoding="utf-8")
    table_path = ROOT / "tables" / "stage02_totalsegmentator_summary.csv"
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    request_path = ROOT / "tables" / "stage02_manual_landmarks_request.md"
    request_path.write_text(
        "# Stage 2 manual landmark request — resolved\n\n"
        "The former request for manual C1/T1 coordinates was resolved by running "
        "TotalSegmentator 2.18.0 on all A, N, and V scans. Both native-grid C1 "
        "and T1 masks are nonempty for all 15 scans. The arterial masks were used "
        "to calculate the requested C1–T1 physical span with a fixed 15 mm extension at each end.\n\n"
        "Human visual review of the labels, body contours, and carotid coverage remains "
        "pending; see `stage_02.json` and `stage02_totalsegmentator_summary.csv`.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": stage["status"],
                "validated_masks": len(summary_rows) * 2,
                "qc_previews": len(summary_rows),
                "summary": str(table_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
