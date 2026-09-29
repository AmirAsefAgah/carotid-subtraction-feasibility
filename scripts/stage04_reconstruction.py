"""Stage 4: signed HU subtraction/reconstruction products and presentation slices."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


from study_paths import ROOT

PATIENTS = ("P001", "P002", "P003", "P004", "P005")
FORMULAS = {
    "arterial_enhancement": {
        "formula": "A - N",
        "phases": ("A", "N"),
        "mask": "A_N_pairwise",
    },
    "venous_enhancement": {
        "formula": "V - N",
        "phases": ("V", "N"),
        "mask": "NAV_common",
    },
    "temporal_difference": {
        "formula": "V - A",
        "phases": ("V", "A"),
        "mask": "A_V_pairwise",
    },
    "dark_lumen_N": {
        "formula": "2*N - A",
        "phases": ("N", "A"),
        "mask": "A_N_pairwise",
    },
    "dark_lumen_V": {
        "formula": "2*V - A",
        "phases": ("V", "A"),
        "mask": "A_V_pairwise",
    },
}
DIFFERENCES = ("arterial_enhancement", "venous_enhancement", "temporal_difference")
DISPLAY = {
    "source_window": {
        "width_hu": 650.0,
        "level_hu": 100.0,
        "range_hu": [-225.0, 425.0],
    },
    "composite_window": {
        "width_hu": 650.0,
        "level_hu": 30.0,
        "range_hu": [-295.0, 355.0],
        "basis": "paper-derived starting reference; fixed cohort-wide",
    },
    "signed_difference_window": {
        "range_hu": [-300.0, 300.0],
        "cmap": "RdBu_r",
        "zero_centred": True,
        "basis": "fixed cohort-wide; quantitative values remain unclipped",
    },
    "invalid_exterior": "black, from common validity independently of windowing",
    "interpolation": "nearest",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_geometry(a: sitk.Image, b: sitk.Image, atol: float = 1e-6) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol, rtol=0)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol, rtol=0)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol, rtol=0)
    )


def geometry(image: sitk.Image) -> dict:
    return {
        "size_xyz": list(map(int, image.GetSize())),
        "spacing_mm": list(map(float, image.GetSpacing())),
        "origin_lps_mm": list(map(float, image.GetOrigin())),
        "direction": list(map(float, image.GetDirection())),
    }


def write_image(image: sitk.Image, path: Path) -> dict:
    sitk.WriteImage(image, str(path), useCompression=True)
    reread = sitk.ReadImage(str(path))
    return {
        "path": str(path),
        "sha256": sha256(path),
        "pixel_type": reread.GetPixelIDTypeAsString(),
        **geometry(reread),
    }


def foreground(mask: sitk.Image) -> int:
    stats = sitk.StatisticsImageFilter()
    stats.Execute(sitk.Cast(mask > 0, sitk.sitkUInt8))
    return int(round(stats.GetSum()))


def product_state(
    dependencies: tuple[str, ...], pair_states: dict[str, str]
) -> tuple[str, str]:
    states = [pair_states[p] for p in dependencies if p != "A"]
    if "fail" in states:
        return (
            "failed_experiment",
            "At least one required Stage 3 registration is failed.",
        )
    if all(state == "pass" for state in states):
        return "accepted", "Every required Stage 3 registration is accepted."
    return (
        "provisional_review",
        "Generated for review; at least one required registration remains unaccepted.",
    )


def formula_image(name: str, a: sitk.Image, n: sitk.Image, v: sitk.Image) -> sitk.Image:
    if name == "arterial_enhancement":
        return a - n
    if name == "venous_enhancement":
        return v - n
    if name == "temporal_difference":
        return v - a
    if name == "dark_lumen_N":
        return 2.0 * n - a
    if name == "dark_lumen_V":
        return 2.0 * v - a
    raise KeyError(name)


def masked_statistics(image: sitk.Image, mask: sitk.Image) -> dict:
    values = sitk.LabelStatisticsImageFilter()
    values.Execute(image, sitk.Cast(mask > 0, sitk.sitkUInt8))
    if not values.HasLabel(1):
        raise RuntimeError("Empty validity mask")
    return {
        "count": int(values.GetCount(1)),
        "minimum_hu": float(values.GetMinimum(1)),
        "maximum_hu": float(values.GetMaximum(1)),
        "mean_hu": float(values.GetMean(1)),
        "standard_deviation_hu": float(values.GetSigma(1)),
        "negative_voxels": foreground(sitk.And(image < 0, mask > 0)),
        "positive_voxels": foreground(sitk.And(image > 0, mask > 0)),
    }


def extract(image: sitk.Image, axis: int, index: int) -> np.ndarray:
    size, start = list(image.GetSize()), [0, 0, 0]
    size[axis], start[axis] = 0, int(index)
    return sitk.GetArrayFromImage(sitk.Extract(image, size, start))


def masked_plane(
    image: sitk.Image, mask: sitk.Image, axis: int, index: int
) -> np.ma.MaskedArray:
    return np.ma.array(
        extract(image, axis, index), mask=extract(mask, axis, index) == 0
    )


def plot_panel(ax, array, title: str, kind: str, aspect: float) -> None:
    cmap = matplotlib.colormaps["RdBu_r" if kind == "difference" else "gray"].copy()
    cmap.set_bad("black")
    if kind == "difference":
        lo, hi = DISPLAY["signed_difference_window"]["range_hu"]
    elif kind == "composite":
        lo, hi = DISPLAY["composite_window"]["range_hu"]
    else:
        lo, hi = DISPLAY["source_window"]["range_hu"]
    ax.imshow(
        array, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest", aspect=aspect
    )
    ax.set_title(title, fontsize=8, color="white")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.01, 0.5, "R", color="white", transform=ax.transAxes, va="center", fontsize=7
    )
    ax.text(
        0.99,
        0.5,
        "L",
        color="white",
        transform=ax.transAxes,
        va="center",
        ha="right",
        fontsize=7,
    )


def save_full_view(
    patient, axis, index, a, n, v, products, common, out_path, stage3_state
) -> dict:
    plane = "axial" if axis == 2 else "coronal"
    spacing = a.GetSpacing()
    aspect = spacing[1] / spacing[0] if axis == 2 else spacing[2] / spacing[0]
    panels = [
        ("A", a, "source"),
        ("N registered", n, "source"),
        ("V registered", v, "source"),
    ]
    panels += [
        (
            f"{name}\n{FORMULAS[name]['formula']}",
            image,
            "difference" if name in DIFFERENCES else "composite",
        )
        for name, image in products.items()
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 8), constrained_layout=True)
    for ax, (title, image, kind) in zip(axes.flat, panels):
        plot_panel(ax, masked_plane(image, common, axis, index), title, kind, aspect)
    probe = (
        (a.GetSize()[0] // 2, a.GetSize()[1] // 2, index)
        if axis == 2
        else (a.GetSize()[0] // 2, index, a.GetSize()[2] // 2)
    )
    physical = a.TransformIndexToPhysicalPoint(probe)[axis]
    fig.suptitle(
        f"{patient} matched {plane} | index {index} | LPS {physical:.1f} mm | Stage 3: {stage3_state.upper()}\nPrimary comparison: common N/A/V neck-body validity; black is invalid, not measured 0 HU",
        fontsize=10,
        color="white",
    )
    fig.savefig(out_path, dpi=160, facecolor="black")
    plt.close(fig)
    return {
        "path": str(out_path),
        "sha256": sha256(out_path),
        "axis": plane,
        "index": int(index),
        "physical_lps_mm": float(physical),
    }


def crop2d(array, cx, cy, radius=48):
    return array[
        max(0, cy - radius) : min(array.shape[0], cy + radius),
        max(0, cx - radius) : min(array.shape[1], cx + radius),
    ]


def save_closeups(
    patient, z, centres, products, common, spacing, out_path, stage3_state
) -> dict:
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
    for row, (cx, cy) in enumerate(centres):
        side = (
            "image-left / patient-right" if row == 0 else "image-right / patient-left"
        )
        for col, (name, image) in enumerate(products.items()):
            kind = "difference" if name in DIFFERENCES else "composite"
            plot_panel(
                axes[row, col],
                crop2d(masked_plane(image, common, 2, z), int(cx), int(cy)),
                f"{side}\n{name} ({FORMULAS[name]['formula']})",
                kind,
                spacing[1] / spacing[0],
            )
    fig.suptitle(
        f"{patient} bilateral CTA-derived candidate close-ups | axial {z} | Stage 3: {stage3_state.upper()}\nNot expert-confirmed bifurcations; common validity; slices, not MIPs",
        fontsize=10,
        color="white",
    )
    fig.savefig(out_path, dpi=180, facecolor="black")
    plt.close(fig)
    return {
        "path": str(out_path),
        "sha256": sha256(out_path),
        "axis": "axial",
        "index": int(z),
        "centres_index_xy": centres,
        "anatomical_status": "CTA intensity candidates; expert bifurcation confirmation pending",
    }


def read_block(path: Path, z0: int, depth: int) -> np.ndarray:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    reader.SetExtractIndex([0, 0, z0])
    reader.SetExtractSize([reader.GetSize()[0], reader.GetSize()[1], depth])
    return sitk.GetArrayFromImage(reader.Execute()).astype(np.float32, copy=False)


def streamed_statistics(path: Path, mask_path: Path) -> dict:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    size_z = reader.GetSize()[2]
    count = negative = positive = 0
    total = total2 = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    for z0 in range(0, size_z, 16):
        depth = min(16, size_z - z0)
        values = read_block(path, z0, depth)
        valid = read_block(mask_path, z0, depth) > 0
        sample = values[valid].astype(np.float64)
        if not sample.size:
            continue
        count += sample.size
        total += float(sample.sum())
        total2 += float(np.square(sample).sum())
        minimum = min(minimum, float(sample.min()))
        maximum = max(maximum, float(sample.max()))
        negative += int((sample < 0).sum())
        positive += int((sample > 0).sum())
    if not count:
        raise RuntimeError("Empty validity mask")
    mean = total / count
    variance = (
        max(0.0, (total2 - count * mean * mean) / (count - 1)) if count > 1 else 0.0
    )
    return {
        "count": int(count),
        "minimum_hu": minimum,
        "maximum_hu": maximum,
        "mean_hu": mean,
        "standard_deviation_hu": float(np.sqrt(variance)),
        "negative_voxels": negative,
        "positive_voxels": positive,
    }


def generate_one(patient: str, name: str) -> None:
    stage3 = ROOT / "patients" / patient / "stage_03"
    out = ROOT / "patients" / patient / "stage_04"
    out.mkdir(parents=True, exist_ok=True)
    spec = FORMULAS[name]
    loaded = {
        phase: sitk.Cast(
            sitk.ReadImage(
                str(
                    stage3
                    / f"{patient}_{'A_native_crop' if phase == 'A' else phase + '_registered_to_A'}.nii.gz"
                )
            ),
            sitk.sitkFloat32,
        )
        for phase in spec["phases"]
    }
    reference = next(iter(loaded.values()))
    if not all(same_geometry(reference, image) for image in loaded.values()):
        raise RuntimeError(f"{patient} {name}: source geometry mismatch")
    mask_file = {
        "A_N_pairwise": f"{patient}_A_N_pairwise_support.nii.gz",
        "A_V_pairwise": f"{patient}_A_V_pairwise_support.nii.gz",
        "NAV_common": f"{patient}_NAV_common_support.nii.gz",
    }[spec["mask"]]
    mask = sitk.Cast(sitk.ReadImage(str(stage3 / mask_file)) > 0, sitk.sitkUInt8)
    if not same_geometry(reference, mask):
        raise RuntimeError(f"{patient} {name}: mask geometry mismatch")
    a = loaded.get("A")
    n = loaded.get("N")
    v = loaded.get("V")
    if name == "arterial_enhancement":
        raw = a - n
    elif name == "venous_enhancement":
        raw = v - n
    elif name == "temporal_difference":
        raw = v - a
    elif name == "dark_lumen_N":
        raw = 2.0 * n - a
    else:
        raw = 2.0 * v - a
    sitk.WriteImage(
        sitk.Mask(sitk.Cast(raw, sitk.sitkFloat32), mask, 0.0),
        str(out / f"{patient}_{name}.nii.gz"),
        useCompression=True,
    )


def streamed_formula_qc(paths: dict[str, Path], mask_path: Path) -> dict:
    checks = {name: 0.0 for name in FORMULAS}
    identity_max = 0.0
    atol = 0.001
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(mask_path))
    reader.ReadImageInformation()
    size_z = reader.GetSize()[2]
    for z0 in range(0, size_z, 16):
        depth = min(16, size_z - z0)
        valid = read_block(mask_path, z0, depth) > 0
        a, n, v = (read_block(paths[x], z0, depth) for x in ("A", "N", "V"))
        outputs = {name: read_block(paths[name], z0, depth) for name in FORMULAS}
        expected = {
            "arterial_enhancement": a - n,
            "venous_enhancement": v - n,
            "temporal_difference": v - a,
            "dark_lumen_N": np.float32(2) * n - a,
            "dark_lumen_V": np.float32(2) * v - a,
        }
        if valid.any():
            for name in FORMULAS:
                checks[name] = max(
                    checks[name],
                    float(np.max(np.abs(outputs[name][valid] - expected[name][valid]))),
                )
            identity = np.abs(
                outputs["temporal_difference"]
                - (outputs["venous_enhancement"] - outputs["arterial_enhancement"])
            )
            identity_max = max(identity_max, float(identity[valid].max()))
    result = {
        name: {"maximum_absolute_error_hu_on_common_mask": error, "pass": error <= atol}
        for name, error in checks.items()
    }
    result["identity_V_minus_A_equals_V_minus_N_minus_A_minus_N"] = {
        "maximum_absolute_error_hu_on_common_mask": identity_max,
        "pass": identity_max <= atol,
    }
    return {
        "tolerance": {
            "absolute_hu": atol,
            "relative": 2e-6,
            "basis": "float32 arithmetic",
        },
        "checks": result,
        "all_pass": all(x["pass"] for x in result.values()),
    }


def save_figures_from_paths(patient, paths, common_path, candidate, reference, state):
    level = candidate["levels"][len(candidate["levels"]) // 2]
    z = int(level["index_z"])
    centres = [s["candidate_center_index_xy"] for s in level["sides"]]
    y = int(round(np.mean([c[1] for c in centres])))
    common = sitk.ReadImage(str(common_path))
    axial = {}
    coronal = {}
    close = {}
    labels = ("A", "N", "V", *FORMULAS.keys())
    for label in labels:
        image = sitk.ReadImage(str(paths[label]))
        axial[label] = masked_plane(image, common, 2, z)
        coronal[label] = masked_plane(image, common, 1, y)
        close[label] = axial[label]
        del image
        gc.collect()
    spacing = reference.GetSpacing()
    figs = ROOT / "figures"

    def full(axis_arrays, axis, index, out_path):
        plane = "axial" if axis == 2 else "coronal"
        aspect = spacing[1] / spacing[0] if axis == 2 else spacing[2] / spacing[0]
        panels = [
            ("A", axis_arrays["A"], "source"),
            ("N registered", axis_arrays["N"], "source"),
            ("V registered", axis_arrays["V"], "source"),
        ] + [
            (
                f"{name}\n{FORMULAS[name]['formula']}",
                axis_arrays[name],
                "difference" if name in DIFFERENCES else "composite",
            )
            for name in FORMULAS
        ]
        fig, axes = plt.subplots(2, 4, figsize=(14, 8), constrained_layout=True)
        for ax, (title, array, kind) in zip(axes.flat, panels):
            plot_panel(ax, array, title, kind, aspect)
        probe = (
            (reference.GetSize()[0] // 2, reference.GetSize()[1] // 2, index)
            if axis == 2
            else (reference.GetSize()[0] // 2, index, reference.GetSize()[2] // 2)
        )
        physical = reference.TransformIndexToPhysicalPoint(probe)[axis]
        fig.suptitle(
            f"{patient} matched {plane} | index {index} | LPS {physical:.1f} mm | Stage 3: {state.upper()}\nPrimary comparison: common N/A/V neck-body validity; black is invalid, not measured 0 HU",
            fontsize=10,
            color="white",
        )
        fig.savefig(out_path, dpi=160, facecolor="black")
        plt.close(fig)
        return {
            "path": str(out_path),
            "sha256": sha256(out_path),
            "axis": plane,
            "index": index,
            "physical_lps_mm": float(physical),
        }

    axial_record = full(axial, 2, z, figs / f"{patient}_stage04_axial_products.png")
    coronal_record = full(
        coronal, 1, y, figs / f"{patient}_stage04_coronal_products.png"
    )
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
    for row, (cx, cy) in enumerate(centres):
        side = (
            "image-left / patient-right" if row == 0 else "image-right / patient-left"
        )
        for col, name in enumerate(FORMULAS):
            plot_panel(
                axes[row, col],
                crop2d(close[name], int(cx), int(cy)),
                f"{side}\n{name} ({FORMULAS[name]['formula']})",
                "difference" if name in DIFFERENCES else "composite",
                spacing[1] / spacing[0],
            )
    out = figs / f"{patient}_stage04_bilateral_closeups.png"
    fig.suptitle(
        f"{patient} bilateral CTA-derived candidate close-ups | axial {z} | Stage 3: {state.upper()}\nNot expert-confirmed bifurcations; common validity; slices, not MIPs",
        fontsize=10,
        color="white",
    )
    fig.savefig(out, dpi=180, facecolor="black")
    plt.close(fig)
    close_record = {
        "path": str(out),
        "sha256": sha256(out),
        "axis": "axial",
        "index": z,
        "centres_index_xy": centres,
        "anatomical_status": "CTA intensity candidates; expert bifurcation confirmation pending",
    }
    return {
        "axial_full_neck": axial_record,
        "coronal_full_neck": coronal_record,
        "bilateral_closeups": close_record,
    }


def run_patient(record: dict, candidate: dict) -> dict:
    patient = record["patient"]
    stage3_dir = ROOT / "patients" / patient / "stage_03"
    out_dir = ROOT / "patients" / patient / "stage_04"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = {p["moving_phase"]: p for p in record["pairs"]}
    pair_states = {p: pair["state"] for p, pair in pairs.items()}
    paths = {
        "A": stage3_dir / f"{patient}_A_native_crop.nii.gz",
        "N": stage3_dir / f"{patient}_N_registered_to_A.nii.gz",
        "V": stage3_dir / f"{patient}_V_registered_to_A.nii.gz",
    }

    def hdr(path):
        r = sitk.ImageFileReader()
        r.SetFileName(str(path))
        r.ReadImageInformation()
        return r

    a_header = hdr(paths["A"])
    n_header = hdr(paths["N"])
    v_header = hdr(paths["V"])
    mask_sources = {
        "A_N_pairwise": stage3_dir / f"{patient}_A_N_pairwise_support.nii.gz",
        "A_V_pairwise": stage3_dir / f"{patient}_A_V_pairwise_support.nii.gz",
        "NAV_common": stage3_dir / f"{patient}_NAV_common_support.nii.gz",
    }
    geometry_checks = {
        "N_registered": same_geometry(a_header, n_header),
        "V_registered": same_geometry(a_header, v_header),
        **{k: same_geometry(a_header, hdr(p)) for k, p in mask_sources.items()},
    }
    if not all(geometry_checks.values()):
        raise RuntimeError(
            f"{patient}: geometry mismatch before arithmetic: {geometry_checks}"
        )
    mask_outputs = {}
    for name, source in mask_sources.items():
        mask = sitk.Cast(sitk.ReadImage(str(source)) > 0, sitk.sitkUInt8)
        mask_outputs[name] = write_image(
            mask, out_dir / f"{patient}_{name}_validity.nii.gz"
        )
        del mask
        gc.collect()
    product_records = {}
    for name, spec in FORMULAS.items():
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--patient",
                patient,
                "--emit-product",
                name,
            ],
            check=True,
        )
        state, basis = product_state(spec["phases"], pair_states)
        output_path = out_dir / f"{patient}_{name}.nii.gz"
        paths[name] = output_path
        valid_path = Path(mask_outputs[spec["mask"]]["path"])
        stats = streamed_statistics(output_path, valid_path)
        product_records[name] = {
            "formula": spec["formula"],
            "fixed_unit_weights": True,
            "required_phases": list(spec["phases"]),
            "validity_mask": spec["mask"],
            "validity_mask_path": mask_outputs[spec["mask"]]["path"],
            "outside_validity_fill": 0.0,
            "outside_fill_interpretation": "masked compatibility fill; excluded from every statistic and display",
            "coverage_voxels": stats["count"],
            "state": state,
            "state_basis": basis,
            "quantitative_image": {
                "path": str(output_path),
                "sha256": sha256(output_path),
                "pixel_type": sitk.GetPixelIDValueAsString(
                    hdr(output_path).GetPixelID()
                ),
                **geometry(hdr(output_path)),
            },
            "statistics_on_own_validity_mask": stats,
            "statistics_on_primary_common_mask": streamed_statistics(
                output_path, Path(mask_outputs["NAV_common"]["path"])
            ),
            "negative_values_preserved": stats["negative_voxels"] > 0,
        }
    qc = streamed_formula_qc(paths, Path(mask_outputs["NAV_common"]["path"]))
    if not qc["all_pass"]:
        raise RuntimeError(f"{patient}: arithmetic QC failed")
    common = streamed_statistics(paths["A"], Path(mask_outputs["NAV_common"]["path"]))[
        "count"
    ]
    coverage = {
        "primary_comparison_mask": "NAV_common",
        "primary_common_voxels": common,
        "A_N_pairwise_voxels": streamed_statistics(
            paths["A"], Path(mask_outputs["A_N_pairwise"]["path"])
        )["count"],
        "A_V_pairwise_voxels": streamed_statistics(
            paths["A"], Path(mask_outputs["A_V_pairwise"]["path"])
        )["count"],
    }
    coverage["A_N_extra_beyond_common_voxels"] = (
        coverage["A_N_pairwise_voxels"] - common
    )
    coverage["A_V_extra_beyond_common_voxels"] = (
        coverage["A_V_pairwise_voxels"] - common
    )
    coverage["interpretation"] = (
        "Matched five-product comparisons use NAV_common. Quantitative products retain maximal formula-valid pairwise support, including V-A/2V-A where N is unnecessary."
    )
    a = sitk.ReadImage(str(paths["A"]))
    figures = save_figures_from_paths(
        patient,
        paths,
        Path(mask_outputs["NAV_common"]["path"]),
        candidate,
        a,
        record["state"],
    )
    state = (
        "failed_experiment"
        if "fail" in pair_states.values()
        else (
            "accepted"
            if all(s == "pass" for s in pair_states.values())
            else "provisional_review"
        )
    )
    return {
        "patient": patient,
        "state": state,
        "stage03_registration_states": pair_states,
        "geometry_verified_before_arithmetic": True,
        "geometry_checks": geometry_checks,
        "reference_geometry": geometry(a),
        "coverage": coverage,
        "masks": mask_outputs,
        "products": product_records,
        "arithmetic_qc": qc,
        "figures": figures,
        "bone_handling": {
            "primary_images_anatomically_intact": True,
            "ordinary_differences": "Static bone should cancel when registration/acquisition are compatible; residual edges remain QC evidence.",
            "composites": "2N-A and 2V-A retain the anatomical base and therefore retain bone.",
            "optional_bone_suppressed_display_generated": False,
            "reason": "No reviewed anatomical bone mask with carotid-vicinity plaque/lumen preservation evidence was available; no global high-HU deletion was used.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", choices=PATIENTS)
    parser.add_argument("--emit-product", choices=FORMULAS)
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    if args.emit_product:
        if not args.patient:
            parser.error("--emit-product requires --patient")
        generate_one(args.patient, args.emit_product)
        return 0
    stage3_path = ROOT / "stage_03.json"
    stage3 = json.loads(stage3_path.read_text(encoding="utf-8"))
    multi = json.loads(
        (ROOT / "stage03_multilevel_qc.json").read_text(encoding="utf-8")
    )
    candidates = {p["patient"]: p for p in multi["pairs"] if p["pair"] == "N->A"}
    selected = [p for p in stage3["patients"] if args.patient in (None, p["patient"])]
    if args.figures_only:
        summary_path = ROOT / "stage_04.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        by_patient = {p["patient"]: p for p in summary["patients"]}
        for source in selected:
            patient = source["patient"]
            stage3_dir = ROOT / "patients" / patient / "stage_03"
            out_dir = ROOT / "patients" / patient / "stage_04"
            paths = {
                "A": stage3_dir / f"{patient}_A_native_crop.nii.gz",
                "N": stage3_dir / f"{patient}_N_registered_to_A.nii.gz",
                "V": stage3_dir / f"{patient}_V_registered_to_A.nii.gz",
                **{name: out_dir / f"{patient}_{name}.nii.gz" for name in FORMULAS},
            }
            reference = sitk.ReadImage(str(paths["A"]))
            common = Path(by_patient[patient]["masks"]["NAV_common"]["path"])
            by_patient[patient]["figures"] = save_figures_from_paths(
                patient, paths, common, candidates[patient], reference, source["state"]
            )
        summary["patients"] = [by_patient[p] for p in PATIENTS]
        summary["figures_regenerated_at_utc"] = datetime.now(timezone.utc).isoformat()
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return 0
    results = [run_patient(p, candidates[p["patient"]]) for p in selected]
    summary_path = ROOT / "stage_04.json"
    if args.patient and summary_path.exists():
        merged = {
            p["patient"]: p
            for p in json.loads(summary_path.read_text(encoding="utf-8")).get(
                "patients", []
            )
        }
        merged.update({p["patient"]: p for p in results})
        results = [merged[p] for p in PATIENTS if p in merged]
    summary = {
        "schema_version": 1,
        "stage": 4,
        "title": "Subtraction and reconstruction",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage03_summary": {
            "path": str(stage3_path),
            "sha256": sha256(stage3_path),
            "status": stage3["status"],
        },
        "status": (
            "COMPLETE_WITH_FAILED_EXPERIMENTS"
            if any(p["state"] == "failed_experiment" for p in results)
            else "COMPLETE_REGISTRATION_ACCEPTED"
            if all(p["state"] == "accepted" for p in results)
            else "COMPLETE_PROVISIONAL_PENDING_STAGE03_VISUAL_REVIEW"
        ),
        "acceptance_policy": "Stage 4 never promotes a Stage 3 review/fail registration. Review products are provisional; failed registrations are labelled failed_experiment.",
        "quantitative_contract": {
            "units": "HU-derived signed values; not calibrated iodine concentration maps",
            "pixel_type": "float32",
            "arithmetic": "exact listed formulas with fixed unit weights; no absolute value, zero clamp, histogram matching, or intensity normalization",
            "outside_validity": "0 is compatibility fill only and excluded from statistics/presentation by masks",
            "primary_comparison": "common three-phase valid neck/body region",
            "pairwise_retention": "maximal formula-valid support retained per product",
        },
        "method_labels": {
            "dark_lumen_N": "Deng subtraction-plus-addition construction (2N-A) on an arterial reference grid.",
            "dark_lumen_V": "Paper-inspired venous substitution (2V-A) for the 150-second delayed method; not an exact reproduction of proprietary BBCT.",
        },
        "formulas": {n: s["formula"] for n, s in FORMULAS.items()},
        "display_settings": DISPLAY,
        "patients": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "patients": [(p["patient"], p["state"]) for p in results],
                "summary": str(summary_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
