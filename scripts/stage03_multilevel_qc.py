"""Generate multi-level bilateral carotid-region QC from completed Stage 3 outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

import stage03_registration as s3


ROOT = s3.ROOT
OFFSETS_MM = (-12.0, -6.0, 0.0, 6.0, 12.0)


def edge_correlation(fixed: np.ndarray, moving: np.ndarray) -> float:
    fixed = np.clip(fixed.astype(np.float32), -150, 300)
    moving = np.clip(moving.astype(np.float32), -150, 300)
    fg = np.hypot(ndi.sobel(fixed, 0), ndi.sobel(fixed, 1)).ravel()
    mg = np.hypot(ndi.sobel(moving, 0), ndi.sobel(moving, 1)).ravel()
    return (
        float(np.corrcoef(fg, mg)[0, 1]) if np.std(fg) and np.std(mg) else float("nan")
    )


def main() -> int:
    summary_path = ROOT / "stage_03.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = []
    for patient in summary["patients"]:
        a_path = patient["common_support"]["outputs"]["A_original_hu_crop"]["path"]
        fixed = sitk.ReadImage(a_path)
        spacing_z = float(fixed.GetSpacing()[2])
        for pair in patient["pairs"]:
            moving = sitk.ReadImage(pair["outputs"]["registered_original_hu"]["path"])
            phase = pair["moving_phase"]
            location = pair["qc"]["location"]
            base_z = int(location["target_level_index_z_in_crop"])
            target_index = fixed.TransformPhysicalPointToContinuousIndex(
                tuple(location["target_level_lps_mm"])
            )
            vertebra_xy = (float(target_index[1]), float(target_index[0]))
            fig, axes = plt.subplots(
                len(OFFSETS_MM), 4, figsize=(12, 15), constrained_layout=True
            )
            level_records = []
            planes = []
            for offset_mm in OFFSETS_MM:
                z = int(
                    np.clip(
                        round(base_z + offset_mm / spacing_z), 0, fixed.GetSize()[2] - 1
                    )
                )
                fa = s3.extract_matching_plane(fixed, 2, z)
                ma = s3.extract_matching_plane(moving, 2, z)
                centres, method = s3.find_carotid_centres(fa, vertebra_xy)
                planes.append(
                    {
                        "offset": offset_mm,
                        "z": z,
                        "fixed": fa,
                        "moving": ma,
                        "centres": centres,
                        "method": method,
                    }
                )
            # Track outward from the central level to avoid anchoring on unrelated bone.
            for indices in ((3, 4), (1, 0)):
                previous = planes[2]["centres"]
                for index in indices:
                    current = planes[index]["centres"]
                    planes[index]["centres"] = [
                        candidate
                        if np.hypot(candidate[0] - prior[0], candidate[1] - prior[1])
                        <= 28
                        else prior
                        for candidate, prior in zip(current, previous)
                    ]
                    previous = planes[index]["centres"]
            for row, plane in enumerate(planes):
                offset_mm = plane["offset"]
                z = plane["z"]
                fa = plane["fixed"]
                ma = plane["moving"]
                centres = plane["centres"]
                method = plane["method"]
                physical_z = float(fixed.TransformIndexToPhysicalPoint((0, 0, z))[2])
                level = {
                    "offset_mm": offset_mm,
                    "index_z": z,
                    "physical_lps_z_mm": physical_z,
                    "sides": [],
                }
                for side_index, (cy, cx) in enumerate(centres):
                    radius = 42
                    y0, y1 = max(0, cy - radius), min(fa.shape[0], cy + radius)
                    x0, x1 = max(0, cx - radius), min(fa.shape[1], cx + radius)
                    fp = fa[y0:y1, x0:x1]
                    mp = ma[y0:y1, x0:x1]
                    col = side_index * 2
                    axes[row, col].imshow(
                        fp, cmap="gray", vmin=-150, vmax=350, interpolation="nearest"
                    )
                    axes[row, col].set_title(
                        f"{offset_mm:+.0f} mm | {'image-left' if side_index == 0 else 'image-right'} A"
                    )
                    axes[row, col + 1].imshow(
                        s3.overlay(fp, mp), interpolation="nearest"
                    )
                    correlation = edge_correlation(fp, mp)
                    axes[row, col + 1].set_title(
                        f"selected rigid overlay | edge r={correlation:.2f}"
                    )
                    level["sides"].append(
                        {
                            "display_side": "image-left"
                            if side_index == 0
                            else "image-right",
                            "candidate_center_index_xy": [cx, cy],
                            "edge_gradient_correlation": correlation,
                        }
                    )
                for axis in axes[row]:
                    axis.set_xticks([])
                    axis.set_yticks([])
                level["candidate_method"] = method
                level_records.append(level)
            fig.suptitle(
                f"{patient['patient']} {phase}->A | MULTI-LEVEL CANDIDATE QC | not confirmed bifurcations",
                fontsize=13,
            )
            output = (
                ROOT
                / "figures"
                / f"{patient['patient']}_{phase}_to_A_multilevel_carotid_qc.png"
            )
            fig.savefig(output, dpi=150)
            plt.close(fig)
            records.append(
                {
                    "patient": patient["patient"],
                    "pair": f"{phase}->A",
                    "figure": str(output),
                    "figure_sha256": s3.sha256(output),
                    "levels": level_records,
                    "interpretation": "Candidate-region screen across five levels; expert confirmation is required before any landmark is treated as a bifurcation, calcium, or vessel-wall correspondence.",
                }
            )
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "offsets_mm": list(OFFSETS_MM),
        "pairs": records,
        "anatomical_acceptance": "not established",
    }
    output_json = ROOT / "stage03_multilevel_qc.json"
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary["multilevel_candidate_qc"] = {
        "path": str(output_json),
        "pair_count": len(records),
        "status": "AVAILABLE_PENDING_EXPERT_BIFURCATION_CONFIRMATION",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
