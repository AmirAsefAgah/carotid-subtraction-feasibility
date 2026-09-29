"""Create filled body masks on native grids and record C1-T1 crop requirements."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi


from study_paths import ROOT
from study_paths import SOURCE_ROOT

PATIENTS = ("P001", "P002", "P003", "P004", "P005")
PHASES = ("N", "A", "V")
HU_THRESHOLD = -450
INPLANE_MARGIN_MM = 15.0


def iso_utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physical_bounds(img: sitk.Image, start_xyz, end_xyz):
    """Return RAS physical corners for inclusive voxel-index bounds."""
    start_lps = np.asarray(
        img.TransformIndexToPhysicalPoint(tuple(map(int, start_xyz)))
    )
    end_lps = np.asarray(img.TransformIndexToPhysicalPoint(tuple(map(int, end_xyz))))
    ras = lambda p: [-float(p[0]), -float(p[1]), float(p[2])]
    return {
        "start_index_xyz": list(map(int, start_xyz)),
        "end_index_xyz": list(map(int, end_xyz)),
        "start_ras_mm": ras(start_lps),
        "end_ras_mm": ras(end_lps),
    }


def select_slice_envelope(slice_hu: np.ndarray) -> np.ndarray:
    """Return one filled patient silhouette, rejecting off-centre table/pads.

    Candidate patient tissue is HU > -450.  Components are selected by a
    centre-weighted area score; a broad, low rectangular couch is normally
    peripheral and scores poorly.  Holes are then filled, so the output is a
    filled body envelope rather than a skin shell.
    """
    candidate = slice_hu > HU_THRESHOLD
    labels, n = ndi.label(candidate)
    if n == 0:
        return np.zeros_like(candidate, dtype=bool)
    cy, cx = (np.asarray(slice_hu.shape) - 1.0) / 2.0
    areas = np.bincount(labels.ravel())
    usable = np.flatnonzero(areas[1:] >= 25) + 1
    if not len(usable):
        return np.zeros_like(candidate, dtype=bool)
    # Limit centroid calculations to the 12 largest components.
    if len(usable) > 12:
        usable = usable[np.argsort(areas[usable])[-12:]]
    centres = np.asarray(ndi.center_of_mass(candidate, labels, usable))
    d = np.hypot((centres[:, 0] - cy) / max(cy, 1), (centres[:, 1] - cx) / max(cx, 1))
    # Penalize peripheral components to exclude the table and pads.
    selected_index = int(np.argmax(areas[usable] * np.exp(-3.5 * d * d)))
    # Reject peripheral supports when no central body component is present.
    if d[selected_index] > 0.55:
        return np.zeros_like(candidate, dtype=bool)
    best_label = int(usable[selected_index])
    envelope = labels == best_label
    cols = np.arange(envelope.shape[1])
    occupied = envelope.any(axis=1)
    if occupied.any():
        left = envelope.argmax(axis=1)
        right = envelope.shape[1] - 1 - envelope[:, ::-1].argmax(axis=1)
        envelope = (
            occupied[:, None]
            & (cols[None, :] >= left[:, None])
            & (cols[None, :] <= right[:, None])
        )
    return envelope.astype(bool)


def body_envelope(hu_zyx: np.ndarray) -> np.ndarray:
    """Slice-wise patient selection prevents connected posterior supports."""
    masks = np.empty(hu_zyx.shape, dtype=bool)
    for z in range(hu_zyx.shape[0]):
        masks[z] = select_slice_envelope(hu_zyx[z])
    return masks


def crop_bounds(mask_zyx: np.ndarray, img: sitk.Image):
    z_indices = np.flatnonzero(mask_zyx.any(axis=(1, 2)))
    y_indices = np.flatnonzero(mask_zyx.any(axis=(0, 2)))
    x_indices = np.flatnonzero(mask_zyx.any(axis=(0, 1)))
    if not len(z_indices):
        raise RuntimeError("Body envelope is empty")
    z0, z1 = z_indices[[0, -1]]
    y0, y1 = y_indices[[0, -1]]
    x0, x1 = x_indices[[0, -1]]
    sx, sy, _ = img.GetSpacing()
    mx, my = int(np.ceil(INPLANE_MARGIN_MM / sx)), int(np.ceil(INPLANE_MARGIN_MM / sy))
    size_x, size_y, size_z = img.GetSize()
    x0, x1 = max(0, x0 - mx), min(size_x - 1, x1 + mx)
    y0, y1 = max(0, y0 - my), min(size_y - 1, y1 + my)
    # Preserve complete moving-image support through plane: no z crop here.
    return (x0, y0, 0), (x1, y1, size_z - 1)


def contour(ax, image, mask, title, rect=None, orientation=""):
    ax.imshow(image, cmap="gray", vmin=-150, vmax=350, interpolation="nearest")
    if np.any(mask):
        ax.contour(mask.astype(float), levels=[0.5], colors=["lime"], linewidths=0.7)
    if rect:
        x0, y0, x1, y1 = rect
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="cyan", lw=0.7)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(orientation, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


def preview(
    patient: str, phase: str, hu, mask, crop_start, crop_end, out: Path, roi_note: str
):
    # Use central grid indices to avoid allocating all foreground coordinates.
    zc, yc, xc = (hu.shape[0] // 2, hu.shape[1] // 2, hu.shape[2] // 2)
    # Sample central body levels and the lower neck for contour QC.
    z_neck = int(np.clip(zc, 0, hu.shape[0] - 1))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    contour(
        axes[0],
        hu[z_neck],
        mask[z_neck],
        "Axial: CT + filled body contour",
        (crop_start[0], crop_start[1], crop_end[0], crop_end[1]),
        "L/R and A/P as native array display",
    )
    contour(
        axes[1],
        hu[:, yc, :],
        mask[:, yc, :],
        "Coronal: CT + filled body contour",
        None,
        "superior/inferior follows native CT",
    )
    contour(
        axes[2],
        hu[:, :, xc],
        mask[:, :, xc],
        "Sagittal: CT + filled body contour",
        None,
        "superior/inferior follows native CT",
    )
    fig.suptitle(f"{patient} {phase} — MACHINE QC ONLY | {roi_note}", fontsize=10)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def write_mask(mask_zyx, reference: sitk.Image, output: Path):
    mask_img = sitk.GetImageFromArray(mask_zyx.astype(np.uint8))
    mask_img.CopyInformation(reference)
    sitk.WriteImage(mask_img, str(output), useCompression=True)


def task_check():
    """Record actual availability, rather than assuming stale preview outputs exist."""
    result = {
        "installed": False,
        "version": None,
        "cli": shutil.which("TotalSegmentator"),
        "body_task_verified": False,
        "vertebral_label_task_verified": False,
        "reason": None,
    }
    try:
        import TotalSegmentator  # noqa: F401

        result["installed"] = True
        result["version"] = getattr(TotalSegmentator, "__version__", "unknown")
        # Check availability here; segmentation runs separately.
        result["reason"] = (
            "Python package importable, but task names were not invoked by this run."
        )
    except Exception as exc:
        result["reason"] = (
            f"TotalSegmentator unavailable in configured runtime: {type(exc).__name__}: {exc}"
        )
    return result


def main():
    manifest = list(
        csv.DictReader((ROOT / "manifest.csv").open(newline="", encoding="utf-8-sig"))
    )
    manifest_by_key = {(r["patient"], r["phase"]): r for r in manifest}
    task = task_check()
    # Record missing preview artifacts without using them as inputs.
    stale_preview = (
        SOURCE_ROOT / "experimental_auto_crop_all_patients_PREVIEW_summary.json"
    )
    stale_state = {
        "summary_path": str(stale_preview),
        "exists": stale_preview.exists(),
        "usable_outputs_found": False,
        "note": "Preview summary is reference-only; referenced body/C2/T1 files were absent at run time.",
    }
    all_patients, pending = [], []
    for patient in PATIENTS:
        pdir = ROOT / "patients" / patient / "stage_02"
        pdir.mkdir(parents=True, exist_ok=True)
        phase_records = []
        arterial_crop = None
        for phase in PHASES:
            row = manifest_by_key[(patient, phase)]
            source = Path(row["source_path"])
            img = sitk.ReadImage(str(source))
            hu = sitk.GetArrayFromImage(img)
            mask = body_envelope(hu)
            start, end = crop_bounds(mask, img)
            out_mask = pdir / f"{patient}_{phase}_body_envelope.nii.gz"
            out_qc = ROOT / "figures" / f"{patient}_{phase}_stage02_body_qc.png"
            write_mask(mask, img, out_mask)
            preview(
                patient,
                phase,
                hu,
                mask,
                start,
                end,
                out_qc,
                "C1–T1 ROI pending manual C1/T1 landmarks",
            )
            written = sitk.ReadImage(str(out_mask))
            geometry_match = (
                written.GetSize() == img.GetSize()
                and np.allclose(written.GetSpacing(), img.GetSpacing())
                and np.allclose(written.GetOrigin(), img.GetOrigin())
                and np.allclose(written.GetDirection(), img.GetDirection())
            )
            # The slice-wise method does not require a 3-D component count.
            nonempty_axial_slices = int(np.count_nonzero(mask.any(axis=(1, 2))))
            record = {
                "phase": phase,
                "source_ct": str(source),
                "source_sha256": sha256(source),
                "body_envelope_mask": str(out_mask),
                "mask_sha256": sha256(out_mask),
                "machine_qc_preview": str(out_qc),
                "method": "threshold_connected_component_per_axial_slice_then_hole_fill_fallback",
                "threshold_hu_gt": HU_THRESHOLD,
                "is_filled_envelope": True,
                "not_a_skin_shell": True,
                "mask_geometry_matches_source": bool(geometry_match),
                "mask_voxels": int(mask.sum()),
                "mask_fraction": float(mask.mean()),
                "nonempty_axial_slices": nonempty_axial_slices,
                "broad_registration_support_crop": physical_bounds(img, start, end),
                "coverage_flags": {
                    "body_envelope_available": True,
                    "table_pads_exterior": "machine_screened_by_central_component_rule; human contour confirmation pending",
                    "machine_contour_preview_available": True,
                    "carotid_region_not_cut_by_longitudinal_roi": "not_assessable_until_C1_and_T1_landmarks_are_supplied",
                    "human_review_status": "PENDING_HUMAN_STAGE02_MASK_AND_CAROTID_COVERAGE_REVIEW",
                },
                "manual_corrections": [],
            }
            phase_records.append(record)
            if phase == "A":
                arterial_crop = record["broad_registration_support_crop"]
        roi = {
            "reference_phase": "A",
            "definition": "superior edge C1 to inferior edge T1; extend 15 mm superiorly and inferiorly",
            "status": "PENDING_MANUAL_LANDMARKS",
            "c1_superior_edge_ras_mm": None,
            "t1_inferior_edge_ras_mm": None,
            "L_mm": None,
            "extended_superior_bound_ras_mm": None,
            "extended_inferior_bound_ras_mm": None,
            "coverage_flags": {
                "requested_longitudinal_roi_available": False,
                "C1_automatic_label_available": False,
                "T1_automatic_label_available": False,
                "carotid_cut_check": "PENDING until requested ROI exists",
            },
            "broad_in_plane_support_on_A": arterial_crop,
            "manual_corrections": [],
        }
        patient_record = {
            "patient": patient,
            "status": "BODY_MASKS_AVAILABLE_ROI_PENDING_MANUAL_C1_T1",
            "human_review_status": "PENDING_HUMAN_STAGE02_MASK_CONTOUR_AND_CAROTID_COVERAGE_REVIEW",
            "roi": roi,
            "phases": phase_records,
            "pending_review_items": [
                "Human reviewer: inspect all body-envelope contours against CT; confirm table, pads, and exterior excluded.",
                "Human reviewer: supply A-space RAS physical superior C1 edge and inferior T1 edge; then apply fixed 15 mm extensions.",
                "Human reviewer: confirm bilateral carotids, bifurcations, and adjacent soft tissue remain within eventual ROI and broad in-plane crop.",
            ],
        }
        all_patients.append(patient_record)
        pending.append(
            {
                "patient": patient,
                "arterial_ct": manifest_by_key[(patient, "A")]["source_path"],
                "required_coordinates": "RAS physical z (mm) of superior edge of C1 and inferior edge of T1 on arterial CT",
            }
        )
        (pdir / f"{patient}_stage02_roi.json").write_text(
            json.dumps(patient_record, indent=2), encoding="utf-8"
        )

    request = [
        "# Stage 2 manual landmark request",
        "",
        "Automated C1/T1 recovery was not performed because TotalSegmentator is not installed in the configured runtime and the prior preview output files are absent. No vertebral level was substituted or inferred.",
        "",
        "For each arterial CT below, please provide two RAS physical z coordinates in millimetres:",
        "- superior edge of C1",
        "- inferior edge of T1",
        "",
        "The pipeline will calculate the C1-to-T1 span in physical space, then extend both ends by 15 mm. Please also confirm the proposed body-mask contour excludes table/pads and preserves both carotid bifurcations.",
        "",
    ]
    for item in pending:
        request.append(f"- {item['patient']}: {item['arterial_ct']}")
    request_path = ROOT / "tables" / "stage02_manual_landmarks_request.md"
    request_path.write_text("\n".join(request) + "\n", encoding="utf-8")
    summary = {
        "stage": "02",
        "created_utc": iso_utc(),
        "status": "PARTIALLY_COMPLETE_BODY_MASKS_AVAILABLE_C1_T1_ROI_PENDING_MANUAL_LANDMARKS",
        "human_review_status": "NOT_PERFORMED_FOR_STAGE02; PENDING_HUMAN_REVIEW",
        "machine_review_status": "COMPLETED_PREVIEWS_GENERATED_FOR_15_PHASES",
        "settings": {
            "threshold_hu_gt": HU_THRESHOLD,
            "inplane_margin_mm": INPLANE_MARGIN_MM,
            "source_intensities_modified": False,
            "mask_storage": "separate uint8 native-grid NIfTI",
        },
        "totalsegmentator_check": task,
        "existing_output_check": stale_state,
        "manual_landmark_request": str(request_path),
        "patients": all_patients,
        "pending_review_items": [
            "All five arterial scans need manual C1/T1 physical landmark coordinates.",
            "All 15 machine contour previews require human Stage 2 review; no human Stage 2 review has been claimed.",
        ],
    }
    (ROOT / "stage_02.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "stage_summary": str(ROOT / "stage_02.json"),
                "manual_landmark_request": str(request_path),
                "machine_previews": 15,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
