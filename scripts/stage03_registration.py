"""Stage 3: rigid N->A and V->A registration with conservative support masks.

SimpleITK resampling transforms map points in the fixed/output domain to the
moving/input domain.  Every transform written here follows that convention:
``A fixed point -> moving phase point``.  The original moving HU data are read
again and sampled once, directly onto the final cropped native-A grid.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from stage03_acceptance import apply_reviews

# Use one thread for repeatable floating-point reductions.
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)

from study_paths import ROOT

PATIENTS = ("P001", "P002", "P003", "P004", "P005")
MOVING_PHASES = ("N", "V")
SEED = 20260913
CLIP_HU = (-200.0, 800.0)
OPTIMIZATION_SPACING_MM = 1.0
RIGID_SHRINK = (4, 2, 1)
RIGID_SMOOTH_MM = (2.0, 1.0, 0.0)
RIGID_ITERATIONS = 160
RIGID_SAMPLING = 0.012
LOCAL_ITERATIONS = 90
LOCAL_SAMPLING = 0.025
LANDMARK_RESCUE_MM = 1.5
LANDMARK_FAIL_MM = 10.0
SUPPORT_FAIL_FRACTION = 0.80
DEFAULT_FILL_HU = -1024.0


def iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_geometry(a: sitk.Image, b: sitk.Image, atol: float = 1e-6) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol)
    )


def center_of_label(mask: sitk.Image) -> tuple[float, float, float]:
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk.Cast(mask > 0, sitk.sitkUInt8))
    if not stats.HasLabel(1):
        raise RuntimeError("Empty landmark label")
    return tuple(float(v) for v in stats.GetCentroid(1))


def crop_to_mask_bbox(
    reference: sitk.Image, mask: sitk.Image
) -> tuple[sitk.Image, dict]:
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk.Cast(mask > 0, sitk.sitkUInt8))
    if not stats.HasLabel(1):
        raise RuntimeError("Requested Stage 2 ROI is empty")
    bbox = stats.GetBoundingBox(1)  # x,y,z,size_x,size_y,size_z
    index = [int(v) for v in bbox[:3]]
    size = [int(v) for v in bbox[3:]]
    cropped = sitk.RegionOfInterest(reference, size, index)
    expected_origin = reference.TransformIndexToPhysicalPoint(tuple(index))
    if not np.allclose(cropped.GetOrigin(), expected_origin, atol=1e-6):
        raise RuntimeError("Crop origin did not update to the selected physical index")
    return cropped, {
        "index_xyz": index,
        "size_xyz": size,
        "origin_lps_mm": list(map(float, cropped.GetOrigin())),
        "spacing_mm": list(map(float, cropped.GetSpacing())),
        "direction": list(map(float, cropped.GetDirection())),
        "origin_verified_from_native_A_index": True,
    }


def resample_isotropic(
    image: sitk.Image, spacing_mm: float, interpolator: int
) -> sitk.Image:
    old_size = np.asarray(image.GetSize(), dtype=float)
    old_spacing = np.asarray(image.GetSpacing(), dtype=float)
    spacing = np.full(3, float(spacing_mm))
    size = np.maximum(1, np.round(old_size * old_spacing / spacing)).astype(int)
    return sitk.Resample(
        image,
        [int(v) for v in size],
        sitk.Transform(3, sitk.sitkIdentity),
        interpolator,
        image.GetOrigin(),
        tuple(spacing),
        image.GetDirection(),
        0,
        image.GetPixelID(),
    )


def initial_transform(
    fixed_landmarks: dict, moving_landmarks: dict
) -> sitk.Euler3DTransform:
    fixed_points = np.asarray(list(fixed_landmarks.values()), dtype=float)
    moving_points = np.asarray(list(moving_landmarks.values()), dtype=float)
    fixed_center = fixed_points.mean(axis=0)
    translation = moving_points.mean(axis=0) - fixed_center
    tx = sitk.Euler3DTransform()
    tx.SetCenter(tuple(fixed_center))
    tx.SetTranslation(tuple(translation))
    tx.SetComputeZYX(False)
    return tx


def make_registration(
    seed: int,
    iterations: int,
    sampling: float,
    shrink: tuple[int, ...],
    smooth: tuple[float, ...],
) -> sitk.ImageRegistrationMethod:
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(float(sampling), int(seed))
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=int(iterations),
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=15,
        estimateLearningRate=method.EachIteration,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel(list(shrink))
    method.SetSmoothingSigmasPerLevel(list(smooth))
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return method


def optimize_rigid(
    fixed: sitk.Image,
    moving: sitk.Image,
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    initial: sitk.Transform,
    seed: int,
    local: bool = False,
) -> tuple[sitk.Transform, dict]:
    if local:
        method = make_registration(
            seed, LOCAL_ITERATIONS, LOCAL_SAMPLING, (2, 1), (1.0, 0.0)
        )
    else:
        method = make_registration(
            seed, RIGID_ITERATIONS, RIGID_SAMPLING, RIGID_SHRINK, RIGID_SMOOTH_MM
        )
    method.SetMetricFixedMask(sitk.Cast(fixed_mask > 0, sitk.sitkUInt8))
    method.SetMetricMovingMask(sitk.Cast(moving_mask > 0, sitk.sitkUInt8))
    method.SetInitialTransform(initial, inPlace=False)
    history: list[dict] = []

    def record_iteration():
        history.append(
            {
                "iteration": int(method.GetOptimizerIteration()),
                "metric": float(method.GetMetricValue()),
                "level": int(method.GetCurrentLevel()),
            }
        )

    method.AddCommand(sitk.sitkIterationEvent, record_iteration)
    started = time.perf_counter()
    result = method.Execute(fixed, moving)
    # Flatten nested transforms for ITK HDF5 serialization.
    result = sitk.CompositeTransform(result)
    result.FlattenTransform()
    return result, {
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "final_metric": float(method.GetMetricValue()),
        "optimizer_iteration": int(method.GetOptimizerIteration()),
        "stop_condition": method.GetOptimizerStopConditionDescription(),
        "metric_valid_points": int(method.GetMetricNumberOfValidPoints()),
        "history": history,
    }


def landmark_discrepancies(
    transform_fixed_to_moving: sitk.Transform, fixed: dict, moving: dict
) -> tuple[list[dict], float]:
    inverse = transform_fixed_to_moving.GetInverse()
    records = []
    for name in sorted(fixed):
        moved_in_fixed = np.asarray(inverse.TransformPoint(moving[name]), dtype=float)
        fixed_point = np.asarray(fixed[name], dtype=float)
        error = float(np.linalg.norm(moved_in_fixed - fixed_point))
        records.append(
            {
                "landmark": name,
                "source": "TotalSegmentator label centroid",
                "fixed_A_lps_mm": list(map(float, fixed_point)),
                "moving_native_lps_mm": list(map(float, moving[name])),
                "moving_mapped_to_A_lps_mm": list(map(float, moved_in_fixed)),
                "discrepancy_mm": error,
            }
        )
    return records, max(r["discrepancy_mm"] for r in records)


def direction_verification(
    transform: sitk.Transform, fixed: dict, moving: dict
) -> dict:
    correct = []
    wrong = []
    inverse = transform.GetInverse()
    for name in sorted(fixed):
        correct.append(
            np.linalg.norm(
                np.asarray(inverse.TransformPoint(moving[name])) - fixed[name]
            )
        )
        # Reverse-direction evaluation checks the transform convention.
        wrong.append(
            np.linalg.norm(
                np.asarray(transform.TransformPoint(moving[name])) - fixed[name]
            )
        )
    points = list(fixed.values()) + list(moving.values())
    roundtrip = []
    for point in points:
        roundtrip.append(
            np.linalg.norm(
                np.asarray(inverse.TransformPoint(transform.TransformPoint(point)))
                - point
            )
        )
    return {
        "stored_convention": "fixed A physical point -> moving native physical point (SimpleITK resampling convention)",
        "moving_to_fixed_for_landmarks": "inverse(stored transform)",
        "mean_correct_direction_landmark_error_mm": float(np.mean(correct)),
        "mean_deliberately_wrong_direction_error_mm": float(np.mean(wrong)),
        "max_transform_inverse_roundtrip_error_mm": float(np.max(roundtrip)),
        "roundtrip_tolerance_mm": 1e-5,
        "roundtrip_pass": bool(np.max(roundtrip) < 1e-5),
    }


def conservative_native_support(image: sitk.Image) -> sitk.Image:
    """One-voxel erosion guarantees distance from every native grid boundary."""
    support = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    support.CopyInformation(image)
    support = support + 1
    return sitk.BinaryErode(support, [1, 1, 1], sitk.sitkBall, 0, 1, False)


def write_image(image: sitk.Image, path: Path) -> dict:
    sitk.WriteImage(image, str(path), useCompression=True)
    reread = sitk.ReadImage(str(path))
    return {
        "path": str(path),
        "sha256": sha256(path),
        "size_xyz": list(map(int, reread.GetSize())),
        "spacing_mm": list(map(float, reread.GetSpacing())),
        "origin_lps_mm": list(map(float, reread.GetOrigin())),
        "direction": list(map(float, reread.GetDirection())),
    }


def transform_summary(transform: sitk.Transform) -> dict:
    return {
        "name": transform.GetName(),
        "dimension": int(transform.GetDimension()),
        "parameters": list(map(float, transform.GetParameters())),
        "fixed_parameters": list(map(float, transform.GetFixedParameters())),
        "number_of_subtransforms": int(transform.GetNumberOfTransforms())
        if hasattr(transform, "GetNumberOfTransforms")
        else 1,
    }


def normalize_window(
    array: np.ndarray, lo: float = -150, hi: float = 350
) -> np.ndarray:
    return np.clip((array.astype(np.float32) - lo) / (hi - lo), 0, 1)


def overlay(fixed: np.ndarray, moving: np.ndarray) -> np.ndarray:
    f = normalize_window(fixed)
    m = normalize_window(moving)
    return np.stack((f, m, m), axis=-1)


def checkerboard(fixed: np.ndarray, moving: np.ndarray, tile: int = 24) -> np.ndarray:
    yy, xx = np.indices(fixed.shape)
    choose = ((yy // tile + xx // tile) % 2) == 0
    return np.where(choose, fixed, moving)


def extract_matching_plane(image: sitk.Image, axis: int, index: int) -> np.ndarray:
    """Extract one 2-D plane without materializing the full 3-D image as NumPy."""
    size = list(image.GetSize())
    start = [0, 0, 0]
    start[axis] = int(index)
    size[axis] = 0
    return sitk.GetArrayFromImage(sitk.Extract(image, size, start))


def label_index_in_crop(label: sitk.Image, crop: sitk.Image) -> np.ndarray:
    point = center_of_label(label)
    return np.asarray(crop.TransformPhysicalPointToContinuousIndex(point), dtype=float)


def find_carotid_centres(
    axial: np.ndarray, vertebra_xy: tuple[float, float]
) -> tuple[list[tuple[int, int]], str]:
    """Find bilateral contrast-enhanced vessel candidates for QC crops only."""
    vy, vx = vertebra_xy
    yy, xx = np.indices(axial.shape)
    candidate = (
        (axial > 140)
        & (axial < 650)
        & (yy > vy - 125)
        & (yy < vy + 20)
        & (np.abs(xx - vx) > 18)
        & (np.abs(xx - vx) < 115)
    )
    labels, count = ndi.label(candidate)
    objects = ndi.find_objects(labels)
    options: list[tuple[float, int, int, int]] = []
    for number, obj in enumerate(objects, start=1):
        if obj is None:
            continue
        coords = np.argwhere(labels == number)
        area = len(coords)
        if not 6 <= area <= 700:
            continue
        cy, cx = coords.mean(axis=0)
        compact = area / max(
            (obj[0].stop - obj[0].start) * (obj[1].stop - obj[1].start), 1
        )
        dx = abs(cx - vx)
        dy = cy - vy
        # Favor compact anterolateral candidates to reduce selection of bright bone.
        if not (25 <= dx <= 100 and -105 <= dy <= -18 and compact >= 0.25):
            continue
        score = (
            float(axial[labels == number].mean())
            + 140 * compact
            - 1.7 * abs(dy + 55)
            - 1.2 * abs(dx - 55)
        )
        options.append((score, int(round(cy)), int(round(cx)), area))
    centres = []
    for side in (-1, 1):
        candidates = [o for o in options if math.copysign(1, o[2] - vx) == side]
        if candidates:
            best = max(candidates)
            centres.append((best[1], best[2]))
        else:
            centres.append((int(round(vy - 45)), int(round(vx + side * 55))))
    method = "CTA intensity/component candidates; fallback anatomical offsets used where no candidate passed"
    return centres, method


def save_qc(
    patient: str,
    phase: str,
    fixed_img: sitk.Image,
    before_img: sitk.Image,
    after_img: sitk.Image,
    fixed_landmarks: dict,
    output_dir: Path,
) -> tuple[dict, dict]:
    c1 = np.asarray(
        fixed_img.TransformPhysicalPointToContinuousIndex(fixed_landmarks["C1"]),
        dtype=float,
    )
    t1 = np.asarray(
        fixed_img.TransformPhysicalPointToContinuousIndex(fixed_landmarks["T1"]),
        dtype=float,
    )
    size_x, size_y, size_z = fixed_img.GetSize()
    target_z = int(np.clip(round(t1[2] + 0.62 * (c1[2] - t1[2])), 0, size_z - 1))
    target_y = int(np.clip(round(t1[1] + 0.62 * (c1[1] - t1[1])), 0, size_y - 1))
    target_x = int(np.clip(round(t1[0] + 0.62 * (c1[0] - t1[0])), 0, size_x - 1))
    views = [
        tuple(
            extract_matching_plane(image, 2, target_z)
            for image in (fixed_img, before_img, after_img)
        )
        + ("Axial target-level proxy",),
        tuple(
            extract_matching_plane(image, 1, target_y)
            for image in (fixed_img, before_img, after_img)
        )
        + ("Coronal matching physical plane",),
        tuple(
            extract_matching_plane(image, 0, target_x)
            for image in (fixed_img, before_img, after_img)
        )
        + ("Sagittal matching physical plane",),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(13, 13), constrained_layout=True)
    for row, (f, b, a, title) in enumerate(views):
        axes[row, 0].imshow(overlay(f, b), interpolation="nearest")
        axes[row, 0].set_title(f"{title}: geometry-only\nA=red, {phase}=cyan")
        axes[row, 1].imshow(overlay(f, a), interpolation="nearest")
        axes[row, 1].set_title(f"Selected final rigid\nA=red, {phase}=cyan")
        axes[row, 2].imshow(
            checkerboard(f, a),
            cmap="gray",
            vmin=-150,
            vmax=350,
            interpolation="nearest",
        )
        axes[row, 2].set_title("After checkerboard")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        f"{patient} {phase}->A | MACHINE QC | geometry-only is not registration | target level is a proxy",
        fontsize=12,
    )
    overview = output_dir / f"{patient}_{phase}_to_A_registration_qc.png"
    fig.savefig(overview, dpi=150)
    plt.close(fig)

    fixed_axial = views[0][0]
    before_axial = views[0][1]
    after_axial = views[0][2]
    vertebra_xy = (float(target_y), float(target_x))
    centres, centre_method = find_carotid_centres(fixed_axial, vertebra_xy)
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    closeup_records = []
    for row, ((cy, cx), side) in enumerate(zip(centres, ("image-left", "image-right"))):
        radius = 55
        y0, y1 = max(0, cy - radius), min(size_y, cy + radius)
        x0, x1 = max(0, cx - radius), min(size_x, cx + radius)
        f = fixed_axial[y0:y1, x0:x1]
        b = before_axial[y0:y1, x0:x1]
        a = after_axial[y0:y1, x0:x1]
        axes[row, 0].imshow(
            f, cmap="gray", vmin=-150, vmax=350, interpolation="nearest"
        )
        axes[row, 0].set_title(f"{side}: native A")
        axes[row, 1].imshow(overlay(f, b), interpolation="nearest")
        axes[row, 1].set_title("geometry-only overlay")
        axes[row, 2].imshow(overlay(f, a), interpolation="nearest")
        axes[row, 2].set_title("selected final overlay")
        axes[row, 3].imshow(
            checkerboard(f, a, tile=12), cmap="gray", vmin=-150, vmax=350
        )
        axes[row, 3].set_title("after checkerboard")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
        closeup_records.append(
            {
                "side_in_display": side,
                "center_index_xy": [cx, cy],
                "radius_voxels": radius,
            }
        )
    fig.suptitle(
        f"{patient} {phase}->A | BILATERAL CAROTID-REGION QC (candidate centres, not confirmed bifurcations)",
        fontsize=11,
    )
    closeup = output_dir / f"{patient}_{phase}_to_A_carotid_closeups.png"
    fig.savefig(closeup, dpi=170)
    plt.close(fig)
    return (
        {
            "overview": str(overview),
            "overview_sha256": sha256(overview),
            "carotid_closeups": str(closeup),
            "carotid_closeups_sha256": sha256(closeup),
        },
        {
            "target_level_index_z_in_crop": target_z,
            "target_level_lps_mm": list(
                map(
                    float,
                    fixed_img.TransformIndexToPhysicalPoint(
                        (target_x, target_y, target_z)
                    ),
                )
            ),
            "target_level_definition": "62% from T1 centroid toward C1 centroid; approximate carotid target band only",
            "confirmed_bifurcation_landmark": False,
            "carotid_candidate_method": centre_method,
            "closeups": closeup_records,
        },
    )


def local_patch_gradient_qc(
    fixed: sitk.Image, moving: sitk.Image, qc_location: dict
) -> dict:
    z = int(qc_location["target_level_index_z_in_crop"])
    f = extract_matching_plane(fixed, 2, z).astype(np.float32)
    m = extract_matching_plane(moving, 2, z).astype(np.float32)
    values = []
    for record in qc_location["closeups"]:
        cx, cy = record["center_index_xy"]
        r = 35
        y0, y1 = max(0, cy - r), min(f.shape[0], cy + r)
        x0, x1 = max(0, cx - r), min(f.shape[1], cx + r)
        fp = np.clip(f[y0:y1, x0:x1], -150, 300)
        mp = np.clip(m[y0:y1, x0:x1], -150, 300)
        fg = np.hypot(ndi.sobel(fp, 0), ndi.sobel(fp, 1)).ravel()
        mg = np.hypot(ndi.sobel(mp, 0), ndi.sobel(mp, 1)).ravel()
        corr = (
            float(np.corrcoef(fg, mg)[0, 1])
            if np.std(fg) and np.std(mg)
            else float("nan")
        )
        values.append(
            {
                "side_in_display": record["side_in_display"],
                "gradient_magnitude_correlation": corr,
            }
        )
    return {
        "values": values,
        "interpretation": "Automated local edge proxy only; phase contrast differs and this is not a landmark discrepancy or wall-tolerance measurement.",
    }


def local_refinement_mask(mask: sitk.Image, fixed_landmarks: dict) -> sitk.Image:
    arr = sitk.GetArrayFromImage(mask > 0).astype(np.uint8)
    image = mask
    c1_z = image.TransformPhysicalPointToContinuousIndex(fixed_landmarks["C1"])[2]
    t1_z = image.TransformPhysicalPointToContinuousIndex(fixed_landmarks["T1"])[2]
    low, high = sorted((c1_z, t1_z))
    z0 = int(max(0, math.floor(low + 0.22 * (high - low))))
    z1 = int(min(arr.shape[0] - 1, math.ceil(low + 0.82 * (high - low))))
    band = np.zeros_like(arr)
    band[z0 : z1 + 1] = arr[z0 : z1 + 1]
    result = sitk.GetImageFromArray(band)
    result.CopyInformation(mask)
    return result


def run_pair(patient: str, phase: str, manifest_row: dict, output_dir: Path) -> dict:
    started_at = iso_utc()
    start = time.perf_counter()
    p2 = ROOT / "patients" / patient / "stage_02"
    fixed_path = Path(manifest_row[(patient, "A")]["source_path"])
    moving_path = Path(manifest_row[(patient, phase)]["source_path"])
    fixed_native = sitk.ReadImage(str(fixed_path))
    moving_native = sitk.ReadImage(str(moving_path))
    native_voxel_trigger = float(max(fixed_native.GetSpacing()))
    fixed_body = sitk.ReadImage(str(p2 / f"{patient}_A_body_envelope.nii.gz"))
    moving_body = sitk.ReadImage(str(p2 / f"{patient}_{phase}_body_envelope.nii.gz"))
    roi = sitk.ReadImage(
        str(p2 / f"{patient}_A_C1_T1_plus15mm_longitudinal_roi.nii.gz")
    )
    if not same_geometry(fixed_native, fixed_body) or not same_geometry(
        fixed_native, roi
    ):
        raise RuntimeError("Stage 2 fixed masks do not match native A geometry")
    if not same_geometry(moving_native, moving_body):
        raise RuntimeError("Stage 2 moving body mask does not match moving geometry")

    fixed_labels = {
        name: sitk.ReadImage(
            str(p2 / "totalsegmentator" / "A" / f"vertebrae_{name}.nii.gz")
        )
        for name in ("C1", "T1")
    }
    moving_labels = {
        name: sitk.ReadImage(
            str(p2 / "totalsegmentator" / phase / f"vertebrae_{name}.nii.gz")
        )
        for name in ("C1", "T1")
    }
    fixed_landmarks = {
        name: center_of_label(label) for name, label in fixed_labels.items()
    }
    moving_landmarks = {
        name: center_of_label(label) for name, label in moving_labels.items()
    }
    # Retain centroids and reload labels one at a time to limit memory use.
    del fixed_labels, moving_labels
    gc.collect()
    initial = initial_transform(fixed_landmarks, moving_landmarks)
    initial_landmarks, initial_max = landmark_discrepancies(
        initial, fixed_landmarks, moving_landmarks
    )

    registration_fixed_mask_native = sitk.Cast(
        (fixed_body > 0) & (roi > 0), sitk.sitkUInt8
    )
    fixed_roi, crop = crop_to_mask_bbox(fixed_native, roi)
    fixed_mask_crop = sitk.RegionOfInterest(
        registration_fixed_mask_native, crop["size_xyz"], crop["index_xyz"]
    )
    # Optimization copies only: clip, cast, and use a 1 mm physical grid.
    fixed_opt_native = sitk.Cast(
        sitk.Clamp(fixed_roi, sitk.sitkInt16, CLIP_HU[0], CLIP_HU[1]), sitk.sitkFloat32
    )
    moving_opt_native = sitk.Cast(
        sitk.Clamp(moving_native, sitk.sitkInt16, CLIP_HU[0], CLIP_HU[1]),
        sitk.sitkFloat32,
    )
    fixed_opt = resample_isotropic(
        fixed_opt_native, OPTIMIZATION_SPACING_MM, sitk.sitkLinear
    )
    moving_opt = resample_isotropic(
        moving_opt_native, OPTIMIZATION_SPACING_MM, sitk.sitkLinear
    )
    fixed_mask_opt = sitk.Resample(
        fixed_mask_crop,
        fixed_opt,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    moving_mask_opt = sitk.Resample(
        moving_body,
        moving_opt,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )

    rigid, rigid_run = optimize_rigid(
        fixed_opt,
        moving_opt,
        fixed_mask_opt,
        moving_mask_opt,
        initial,
        SEED + (0 if phase == "N" else 1),
    )
    rigid_landmarks, rigid_max = landmark_discrepancies(
        rigid, fixed_landmarks, moving_landmarks
    )
    rescue_actions = []
    final_transform = rigid
    final_run = rigid_run
    final_landmarks = rigid_landmarks
    final_max = rigid_max
    if rigid_max > LANDMARK_RESCUE_MM:
        local_native = local_refinement_mask(fixed_mask_crop, fixed_landmarks)
        local_opt_mask = sitk.Resample(
            local_native,
            fixed_opt,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
        local, local_run = optimize_rigid(
            fixed_opt,
            moving_opt,
            local_opt_mask,
            moving_mask_opt,
            rigid,
            SEED + 100 + (0 if phase == "N" else 1),
            local=True,
        )
        local_landmarks, local_max = landmark_discrepancies(
            local, fixed_landmarks, moving_landmarks
        )
        # Accept the rigid rescue only if the label-centroid residual does not increase.
        accepted = local_max <= rigid_max + 0.05
        rescue_actions.append(
            {
                "action": "local_rigid_refinement",
                "trigger": f"C1/T1 centroid maximum residual {rigid_max:.3f} mm > {LANDMARK_RESCUE_MM:.3f} mm",
                "target_band": "22%-82% of the T1-to-C1 centroid span within fixed body/ROI",
                "result": local_run,
                "candidate_max_landmark_discrepancy_mm": local_max,
                "accepted": accepted,
                "acceptance_rule": "accept only if maximum independent C1/T1 centroid residual is not worse by >0.05 mm",
            }
        )
        if accepted:
            final_transform, final_run = local, local_run
            final_landmarks, final_max = local_landmarks, local_max

    final_selection = (
        "optimized_full_rigid" if final_transform is rigid else "optimized_local_rigid"
    )
    # Reject metric improvements that worsen C1/T1 alignment by over one native A voxel.
    if final_max > initial_max + native_voxel_trigger:
        rescue_actions.append(
            {
                "action": "reject_optimized_rigid_candidate",
                "trigger": (
                    f"Selected optimized candidate worsened maximum C1/T1 centroid discrepancy "
                    f"from {initial_max:.3f} mm to {final_max:.3f} mm, exceeding the "
                    f"one-native-voxel allowance ({native_voxel_trigger:.3f} mm)."
                ),
                "result": "landmark_initialized_rigid_retained",
                "note": "Similarity score alone is not accepted as anatomical evidence.",
            }
        )
        final_transform = initial
        final_landmarks = initial_landmarks
        final_max = initial_max
        final_selection = (
            "landmark_initialized_rigid_after_optimized_candidate_rejection"
        )
        final_run = {
            "type": "landmark_initialization_no_optimizer",
            "reason": "optimized candidate rejected by external centroid safety check",
        }

    del (
        fixed_opt_native,
        moving_opt_native,
        fixed_opt,
        moving_opt,
        fixed_mask_opt,
        moving_mask_opt,
    )
    gc.collect()

    initial_path = output_dir / f"{patient}_{phase}_to_A_initial.tfm"
    rigid_path = output_dir / f"{patient}_{phase}_to_A_rigid.hdf"
    final_path = output_dir / f"{patient}_{phase}_to_A_final.hdf"
    sitk.WriteTransform(initial, str(initial_path))
    sitk.WriteTransform(rigid, str(rigid_path))
    sitk.WriteTransform(final_transform, str(final_path))

    # Resample original moving HU data once for the quantitative result.
    registered = sitk.Resample(
        moving_native,
        fixed_roi,
        final_transform,
        sitk.sitkLinear,
        DEFAULT_FILL_HU,
        sitk.sitkFloat32,
    )
    registered_output = write_image(
        registered, output_dir / f"{patient}_{phase}_registered_to_A.nii.gz"
    )
    # Retain the fixed image pixel type to avoid a full-volume float32 copy.
    fixed_crop = fixed_roi

    support_native = conservative_native_support(moving_native)
    valid = sitk.Resample(
        support_native,
        fixed_roi,
        final_transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    moving_body_registered = sitk.Resample(
        moving_body,
        fixed_roi,
        final_transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    fixed_body_crop = sitk.RegionOfInterest(
        fixed_body, crop["size_xyz"], crop["index_xyz"]
    )
    roi_crop = sitk.RegionOfInterest(roi, crop["size_xyz"], crop["index_xyz"])
    pair_support = sitk.Cast(
        (valid > 0)
        & (moving_body_registered > 0)
        & (fixed_body_crop > 0)
        & (roi_crop > 0),
        sitk.sitkUInt8,
    )

    target_voxels = int(sitk.GetArrayViewFromImage(fixed_mask_crop).sum())
    pair_voxels = int(sitk.GetArrayViewFromImage(pair_support).sum())
    support_fraction = float(pair_voxels / target_voxels) if target_voxels else 0.0

    valid_output = write_image(
        valid, output_dir / f"{patient}_{phase}_valid_sampling_mask.nii.gz"
    )
    body_output = write_image(
        moving_body_registered,
        output_dir / f"{patient}_{phase}_body_registered_to_A_nn.nii.gz",
    )
    pair_output = write_image(
        pair_support, output_dir / f"{patient}_A_{phase}_pairwise_support.nii.gz"
    )

    # Release native volumes before compression and plotting.
    del (
        support_native,
        moving_body,
        fixed_native,
        fixed_body,
        roi,
        valid,
        moving_body_registered,
        fixed_body_crop,
        roi_crop,
        pair_support,
        fixed_mask_crop,
    )
    gc.collect()
    identity = sitk.Transform(3, sitk.sitkIdentity)
    before_qc = sitk.Resample(
        moving_native,
        fixed_roi,
        identity,
        sitk.sitkLinear,
        DEFAULT_FILL_HU,
        sitk.sitkFloat32,
    )
    del moving_native
    gc.collect()

    outputs = {
        "registered_original_hu": registered_output,
        "valid_sampling_mask": valid_output,
        "moving_body_registered_nn": body_output,
        "pairwise_support_mask": pair_output,
    }
    registered_labels = {}
    for name in ("C1", "T1"):
        label = sitk.ReadImage(
            str(p2 / "totalsegmentator" / phase / f"vertebrae_{name}.nii.gz")
        )
        resampled = sitk.Resample(
            label,
            fixed_roi,
            final_transform,
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
        registered_labels[name] = write_image(
            resampled,
            output_dir / f"{patient}_{phase}_{name}_registered_to_A_nn.nii.gz",
        )
        del label, resampled
        gc.collect()
    outputs["registered_categorical_labels"] = registered_labels

    qc_paths, qc_location = save_qc(
        patient,
        phase,
        fixed_crop,
        before_qc,
        registered,
        fixed_landmarks,
        ROOT / "figures",
    )
    local_qc = local_patch_gradient_qc(fixed_crop, registered, qc_location)
    direction = direction_verification(
        final_transform, fixed_landmarks, moving_landmarks
    )
    voxel_trigger = native_voxel_trigger
    review_triggers = []
    if final_max > voxel_trigger:
        review_triggers.append(
            f"Maximum C1/T1 label-centroid discrepancy {final_max:.3f} mm exceeds approximately one native A voxel ({voxel_trigger:.3f} mm)."
        )
    review_triggers.append(
        "Reviewer visual QC determines alignment acceptance; manual coordinates, segmentation and plaque identification are optional."
    )
    if final_max > LANDMARK_FAIL_MM:
        review_triggers.append(
            "Large automated C1/T1 centroid discrepancy: supplementary review alert, not manual landmark accuracy or automatic rejection."
        )
    if support_fraction < SUPPORT_FAIL_FRACTION or not direction["roundtrip_pass"]:
        state = "fail"
    else:
        state = "review"
    return {
        "patient": patient,
        "pair": f"{phase}->A",
        "fixed_phase": "A",
        "moving_phase": phase,
        "started_at_utc": started_at,
        "completed_at_utc": iso_utc(),
        "runtime_seconds": round(time.perf_counter() - start, 3),
        "state": state,
        "state_basis": "Awaiting reviewer visual QC; manual landmarks and segmentation are optional. Gross technical safeguards remain applicable.",
        "inputs": {
            "fixed_native_A": str(fixed_path),
            "fixed_sha256": sha256(fixed_path),
            "moving_native": str(moving_path),
            "moving_sha256": sha256(moving_path),
            "stage02_fixed_body": str(p2 / f"{patient}_A_body_envelope.nii.gz"),
            "stage02_moving_body": str(p2 / f"{patient}_{phase}_body_envelope.nii.gz"),
            "stage02_longitudinal_roi": str(
                p2 / f"{patient}_A_C1_T1_plus15mm_longitudinal_roi.nii.gz"
            ),
        },
        "transform_convention": direction,
        "transforms": {
            "initial": {
                "path": str(initial_path),
                "sha256": sha256(initial_path),
                **transform_summary(initial),
            },
            "full_rigid": {
                "path": str(rigid_path),
                "sha256": sha256(rigid_path),
                **transform_summary(rigid),
            },
            "final": {
                "path": str(final_path),
                "sha256": sha256(final_path),
                **transform_summary(final_transform),
            },
            "final_has_scale_shear_or_deformation": False,
        },
        "registration": {
            "method": "multiresolution 6-DOF Euler3D rigid",
            "fixed_reference": "native arterial CT physical space",
            "metric": "Mattes mutual information",
            "histogram_bins": 50,
            "fixed_metric_mask": "A body envelope intersect requested C1-T1+15mm ROI",
            "moving_metric_mask": "moving native body envelope",
            "optimization_images_only": {
                "hu_clip": list(CLIP_HU),
                "isotropic_spacing_mm": OPTIMIZATION_SPACING_MM,
                "note": "Clipped/downsampled copies are used only for optimization; altered body exterior is excluded by masks and is not anatomical evidence.",
            },
            "sampling": {
                "strategy": "RANDOM",
                "percentage": RIGID_SAMPLING,
                "seed": SEED + (0 if phase == "N" else 1),
            },
            "pyramid": {
                "shrink_factors": list(RIGID_SHRINK),
                "smoothing_sigmas_mm": list(RIGID_SMOOTH_MM),
            },
            "optimizer": "gradient descent with physical-shift scales",
            "full_rigid_run": rigid_run,
            "final_selected_run": final_run,
            "final_selection": final_selection,
            "optimized_candidate_rejection_rule": "retain landmark-initialized rigid if optimized maximum C1/T1 centroid discrepancy worsens by more than one native A voxel",
        },
        "resampling": {
            "quantitative_source": "original unmodified moving HU image",
            "quantitative_resampling_count": 1,
            "target": "cropped native A grid",
            "interpolator": "linear",
            "outside_fill_hu": DEFAULT_FILL_HU,
            "outside_fill_is_valid_data": False,
            "categorical_interpolator": "nearest-neighbour",
            "geometry_only_before": "separate identity physical resampling used only in QC figure; not treated as registration or quantitative output",
        },
        "crop": crop,
        "landmarks": {
            "measurement_type": "automated C1/T1 label-centroid discrepancy; not manual landmark-registration accuracy",
            "initial": initial_landmarks,
            "after_full_rigid": rigid_landmarks,
            "final": final_landmarks,
            "maximum_final_discrepancy_mm": final_max,
            "pilot_review_trigger_mm": voxel_trigger,
            "trigger_interpretation": "approximately one native voxel; pilot review trigger, not validated vessel-wall tolerance",
            "calcium_correspondence": "not automatically identifiable with sufficient confidence",
            "soft_tissue_correspondence": "not automatically identifiable with sufficient confidence",
        },
        "support": {
            "validity_definition": "nearest-neighbour resampling of original moving support eroded one native voxel on every face",
            "partial_linear_interpolation_near_boundary_excluded": True,
            "target_neck_body_voxels": target_voxels,
            "pairwise_support_voxels": pair_voxels,
            "pairwise_support_fraction": support_fraction,
        },
        "qc": {
            "figures": qc_paths,
            "location": qc_location,
            "local_edge_proxy": local_qc,
            "observations": [
                "Matching physical planes compare geometry-only identity resampling against optimized rigid registration.",
                "Full views include vertebral/bony and visible soft-tissue boundaries; bilateral close-ups target carotid-region candidates.",
                "Good bone alignment alone is not accepted as evidence of carotid wall alignment.",
                "Reviewer visual acceptance does not require identifiable plaque or manual anatomical coordinates.",
            ],
            "review_triggers": review_triggers,
        },
        "rescue_actions": rescue_actions,
        "outputs": outputs,
    }


def combine_patient_outputs(
    patient: str, pair_records: list[dict], output_dir: Path
) -> dict:
    fixed_path = Path(pair_records[0]["inputs"]["fixed_native_A"])
    p2 = ROOT / "patients" / patient / "stage_02"
    crop = pair_records[0]["crop"]

    # Build support masks sequentially to limit memory use.
    fixed_native = sitk.ReadImage(str(fixed_path))
    fixed_crop = sitk.RegionOfInterest(
        fixed_native, crop["size_xyz"], crop["index_xyz"]
    )
    fixed_crop_float = sitk.Cast(fixed_crop, sitk.sitkFloat32)
    a_output = write_image(
        fixed_crop_float, output_dir / f"{patient}_A_native_crop.nii.gz"
    )
    del fixed_native, fixed_crop, fixed_crop_float
    gc.collect()

    fixed_body = sitk.ReadImage(str(p2 / f"{patient}_A_body_envelope.nii.gz"))
    fixed_body_crop = sitk.RegionOfInterest(
        fixed_body, crop["size_xyz"], crop["index_xyz"]
    )
    fixed_body_crop = sitk.Cast(fixed_body_crop > 0, sitk.sitkUInt8)
    body_output = write_image(
        fixed_body_crop, output_dir / f"{patient}_A_body_crop.nii.gz"
    )
    del fixed_body
    gc.collect()

    roi = sitk.ReadImage(
        str(p2 / f"{patient}_A_C1_T1_plus15mm_longitudinal_roi.nii.gz")
    )
    roi_crop = sitk.RegionOfInterest(roi, crop["size_xyz"], crop["index_xyz"])
    roi_crop = sitk.Cast(roi_crop > 0, sitk.sitkUInt8)
    roi_output = write_image(
        roi_crop, output_dir / f"{patient}_requested_roi_crop.nii.gz"
    )
    del roi
    gc.collect()

    n_support = sitk.ReadImage(
        pair_records[0]["outputs"]["pairwise_support_mask"]["path"]
    )
    v_support = sitk.ReadImage(
        pair_records[1]["outputs"]["pairwise_support_mask"]["path"]
    )
    # Verify matching grids before harmonizing mask metadata after NIfTI rounding.
    for name, image in (
        ("V support", v_support),
        ("A body", fixed_body_crop),
        ("ROI", roi_crop),
    ):
        if image.GetSize() != n_support.GetSize() or not (
            np.allclose(image.GetSpacing(), n_support.GetSpacing(), atol=1e-5)
            and np.allclose(image.GetOrigin(), n_support.GetOrigin(), atol=1e-5)
            and np.allclose(image.GetDirection(), n_support.GetDirection(), atol=1e-6)
        ):
            raise RuntimeError(f"{patient}: {name} is not on the cropped A grid")
        image.SetSpacing(n_support.GetSpacing())
        image.SetOrigin(n_support.GetOrigin())
        image.SetDirection(n_support.GetDirection())
    target = sitk.And(fixed_body_crop, roi_crop)
    common = sitk.And(sitk.And(n_support, v_support), target)
    target_voxels = int(sitk.GetArrayViewFromImage(target).sum())
    common_voxels = int(sitk.GetArrayViewFromImage(common).sum())
    common_output = write_image(
        common, output_dir / f"{patient}_NAV_common_support.nii.gz"
    )
    outputs = {
        "A_original_hu_crop": a_output,
        "A_body_crop": body_output,
        "requested_roi_crop": roi_output,
        "common_three_phase_support": common_output,
    }
    return {
        "crop": crop,
        "target_neck_body_voxels": target_voxels,
        "common_three_phase_support_voxels": common_voxels,
        "common_three_phase_support_fraction": float(common_voxels / target_voxels)
        if target_voxels
        else 0.0,
        "outputs": outputs,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patients", nargs="+", choices=PATIENTS, default=list(PATIENTS)
    )
    parser.add_argument(
        "--phases", nargs="+", choices=MOVING_PHASES, default=list(MOVING_PHASES)
    )
    parser.add_argument("--combine-only", action="store_true")
    parser.add_argument("--config", default=str(ROOT / "config" / "study.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if int(config.get("random_seed", -1)) != SEED:
        raise RuntimeError(f"Configured random seed must remain {SEED}")
    stage01 = json.loads((ROOT / "stage_01.json").read_text(encoding="utf-8"))
    stage02 = json.loads((ROOT / "stage_02.json").read_text(encoding="utf-8"))
    manifest = list(
        csv.DictReader((ROOT / "manifest.csv").open(newline="", encoding="utf-8-sig"))
    )
    rows = {(row["patient"], row["phase"]): row for row in manifest}
    summary_path = ROOT / "stage_03.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {
            "schema_version": 1,
            "stage": 3,
            "title": "Registration and validity masks",
            "started_at_utc": iso_utc(),
            "roadmap": "METHODS.md",
            "stage01_status_consumed": stage01.get("status"),
            "stage02_status_consumed": stage02.get("status"),
            "patients": [],
        }
    existing = {p["patient"]: p for p in summary.get("patients", [])}
    for patient in args.patients:
        print(f"[{iso_utc()}] Starting {patient}", flush=True)
        output_dir = ROOT / "patients" / patient / "stage_03"
        output_dir.mkdir(parents=True, exist_ok=True)
        prior_pairs = {
            p.get("moving_phase"): p
            for p in existing.get(patient, {}).get("pairs", [])
            if p.get("moving_phase") in MOVING_PHASES
        }
        phases_to_run = () if args.combine_only else args.phases
        for phase in phases_to_run:
            print(f"[{iso_utc()}] {patient} {phase}->A", flush=True)
            try:
                prior_pairs[phase] = run_pair(patient, phase, rows, output_dir)
            except Exception as exc:
                traceback.print_exc()
                prior_pairs[phase] = {
                    "patient": patient,
                    "pair": f"{phase}->A",
                    "fixed_phase": "A",
                    "moving_phase": phase,
                    "state": "fail",
                    "state_basis": "Registration pipeline exception; other pairs/patients continue.",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "completed_at_utc": iso_utc(),
                }
        pair_records = [prior_pairs[p] for p in MOVING_PHASES if p in prior_pairs]
        successful = [p for p in pair_records if "outputs" in p]
        combined = None
        if {p["moving_phase"] for p in successful} == set(MOVING_PHASES):
            try:
                combined = combine_patient_outputs(patient, successful, output_dir)
            except Exception as exc:
                combined = {"status": "fail", "error": f"{type(exc).__name__}: {exc}"}
        states = [p["state"] for p in pair_records]
        patient_state = (
            "partial"
            if len(pair_records) != len(MOVING_PHASES)
            else "fail"
            if "fail" in states
            else "review"
            if "review" in states
            else "pass"
        )
        existing[patient] = {
            "patient": patient,
            "state": patient_state,
            "pairs": pair_records,
            "common_support": combined,
            "human_review_status": "PENDING_VISUAL_REVIEW",
            "completed_at_utc": iso_utc(),
        }
        summary["patients"] = [existing[p] for p in PATIENTS if p in existing]
        summary["completed_at_utc"] = iso_utc()
        summary["status"] = "IN_PROGRESS"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[{iso_utc()}] Completed {patient}: {patient_state}", flush=True)
    completed_ids = {p["patient"] for p in summary["patients"]}
    all_states = [
        pair["state"] for patient in summary["patients"] for pair in patient["pairs"]
    ]
    summary["status"] = (
        "COMPLETE_WITH_FAILURES"
        if completed_ids == set(PATIENTS) and any(s == "fail" for s in all_states)
        else "COMPLETE_PENDING_ANATOMICAL_REVIEW"
        if completed_ids == set(PATIENTS) and any(s == "review" for s in all_states)
        else "COMPLETE"
        if completed_ids == set(PATIENTS)
        else "PARTIAL"
    )
    summary["completion_interpretation"] = (
        "All requested registrations were attempted and outputs/QC were produced where technically possible. "
        "A review state is not anatomical acceptance; human carotid-level review remains pending."
    )
    summary["default_settings_unchanged_across_patients"] = True
    summary["deformable_refinement"] = {
        "attempted": False,
        "reason": "This study uses rigid registration only; local alignment requires review.",
    }
    summary["software"] = {
        "python": sys.version,
        "SimpleITK": sitk.Version_VersionString(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "SimpleITK_global_default_threads": sitk.ProcessObject.GetGlobalDefaultNumberOfThreads(),
    }
    summary["completed_at_utc"] = iso_utc()
    review_path = ROOT / "config" / "stage03_reviewer_visual_qc.json"
    apply_reviews(
        summary, json.loads(review_path.read_text()) if review_path.exists() else {}
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if not any(s == "fail" for s in all_states) else 2


if __name__ == "__main__":
    raise SystemExit(main())
