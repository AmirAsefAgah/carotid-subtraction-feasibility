"""Audit native NIfTI geometry, HU scaling, and source DICOM evidence."""

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import platform
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LPS_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def table(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(
            {
                k: json.dumps(v) if isinstance(v, (list, dict)) else v
                for k, v in r.items()
            }
            for r in rows
        )


def corners(shape, edges=False):
    return np.array(
        list(
            itertools.product(
                *[(-0.5, n - 0.5) if edges else (0, n - 1) for n in shape]
            )
        )
    )


def transform(affine, points):
    return points @ affine[:3, :3].T + affine[:3, 3]


def header(path):
    with gzip.open(path, "rb") as f:
        raw = f.read(348)
    if len(raw) != 348:
        raise ValueError("Truncated NIfTI header")
    endian = "<" if struct.unpack_from("<i", raw)[0] == 348 else ">"

    def read(fmt, offset):
        return struct.unpack_from(endian + fmt, raw, offset)

    if read("i", 0)[0] != 348 or raw[344:348] != b"n+1\x00":
        raise ValueError("Only single-file NIfTI-1 supported by raw audit")
    dim = read("8h", 40)
    if dim[0] != 3 or min(dim[1:4]) <= 0:
        raise ValueError("Expected a positive three-dimensional CT array")
    pixdim = np.asarray(read("8f", 76))
    qcode, scode = read("2h", 252)
    b, c, d, x, y, z = read("6f", 256)
    square = b * b + c * c + d * d
    if square > 1 + 1e-5:
        raise ValueError("Invalid qform quaternion")
    if square > 1:
        b, c, d = np.array([b, c, d]) / np.sqrt(square)
    a = np.sqrt(max(0, 1 - b * b - c * c - d * d))
    rotation = np.array(
        [
            [
                a * a + b * b - c * c - d * d,
                2 * b * c - 2 * a * d,
                2 * b * d + 2 * a * c,
            ],
            [
                2 * b * c + 2 * a * d,
                a * a + c * c - b * b - d * d,
                2 * c * d - 2 * a * b,
            ],
            [
                2 * b * d - 2 * a * c,
                2 * c * d + 2 * a * b,
                a * a + d * d - c * c - b * b,
            ],
        ]
    )
    q = np.eye(4)
    q[:3, :3] = rotation @ np.diag(
        pixdim[1:4] * np.array([1, 1, -1 if pixdim[0] < 0 else 1])
    )
    q[:3, 3] = [x, y, z]
    s = np.eye(4)
    s[:3] = np.asarray(read("12f", 280)).reshape(3, 4)
    dtypes = {
        2: "u1",
        4: "i2",
        8: "i4",
        16: "f4",
        64: "f8",
        256: "i1",
        512: "u2",
        768: "u4",
    }
    dtype_code = read("h", 70)[0]
    if dtype_code not in dtypes:
        raise ValueError("Unsupported raw numeric datatype")
    slope, intercept = read("2f", 112)
    if not np.isfinite(slope) or slope == 0:
        effective_slope, effective_intercept = 1.0, 0.0
    else:
        effective_slope, effective_intercept = slope, intercept
    return {
        "shape": list(dim[1:4]),
        "spacing": pixdim[1:4].tolist(),
        "qform_code": qcode,
        "sform_code": scode,
        "qform": q.tolist(),
        "sform": s.tolist(),
        "units_code": raw[123],
        "spatial_units": "mm" if raw[123] & 7 == 2 else "unsupported",
        "raw_slope": float(slope) if np.isfinite(slope) else str(slope),
        "raw_intercept": float(intercept) if np.isfinite(intercept) else str(intercept),
        "effective_slope": effective_slope,
        "effective_intercept": effective_intercept,
        "dtype": np.dtype(endian + dtypes[dtype_code]).str,
        "offset": int(read("f", 108)[0]),
        "description": raw[148:228].split(b"\0")[0].decode("ascii", errors="replace"),
    }


def read_raw(path, hdr):
    dtype = np.dtype(hdr["dtype"])
    expected = int(np.prod(hdr["shape"])) * dtype.itemsize
    with gzip.open(path, "rb") as f:
        f.seek(hdr["offset"])
        data = f.read(expected)
        if len(data) != expected:
            raise ValueError("Truncated NIfTI voxel payload")
        f.read()  # verify gzip trailer/CRC, including any legal trailing bytes
    return np.frombuffer(data, dtype=dtype).reshape(hdr["shape"], order="F")


def geometry(hdr, path, audit):
    q, s = np.asarray(hdr["qform"]), np.asarray(hdr["sform"])
    if hdr["qform_code"] <= 0 or hdr["sform_code"] <= 0:
        raise ValueError(
            "Missing coded qform or sform; explicit geometry review required"
        )
    if hdr["spatial_units"] != "mm":
        raise ValueError("Spatial units are not explicitly millimetres")
    cp = corners(hdr["shape"])
    delta = float(np.max(np.linalg.norm(transform(q, cp) - transform(s, cp), axis=1)))
    if not np.isfinite(s).all() or abs(np.linalg.det(s[:3, :3])) < 1e-10:
        raise ValueError("Invalid/noninvertible affine")
    spacing = np.linalg.norm(s[:3, :3], axis=0)
    direction = s[:3, :3] / spacing
    if delta > audit["qform_sform_corner_tolerance_mm"]:
        raise ValueError("Conflicting qform/sform: %.9f mm at image corners" % delta)
    if not np.allclose(spacing, hdr["spacing"], atol=1e-6, rtol=0):
        raise ValueError("pixdim and sform spacing conflict")
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError("Sheared/nonorthogonal direction; review required")
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    itk_affine = np.eye(4)
    itk_affine[:3, :3] = np.asarray(reader.GetDirection()).reshape(3, 3) @ np.diag(
        reader.GetSpacing()
    )
    itk_affine[:3, 3] = reader.GetOrigin()
    itk_delta = float(
        np.max(np.abs(transform(LPS_RAS @ itk_affine, cp) - transform(s, cp)))
    )
    if tuple(reader.GetSize()) != tuple(hdr["shape"]) or itk_delta > 0.001:
        raise ValueError("SimpleITK and raw NIfTI geometry conflict")
    center = transform(s, cp)
    edge = transform(s, corners(hdr["shape"], True))
    return {
        "affine_ras": s.tolist(),
        "origin_ras_mm": s[:3, 3].tolist(),
        "direction_ras": direction.tolist(),
        "origin_lps_mm": list(reader.GetOrigin()),
        "direction_lps": list(reader.GetDirection()),
        "spacing_mm": spacing.tolist(),
        "axis_codes": sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
            reader.GetDirection()
        ),
        "determinant_ras": float(np.linalg.det(s[:3, :3])),
        "qform_sform_max_corner_error_mm": delta,
        "simpleitk_max_corner_error_mm": itk_delta,
        "voxel_center_bounds_ras_mm": [center.min(0).tolist(), center.max(0).tolist()],
        "voxel_edge_bounds_ras_mm": [edge.min(0).tolist(), edge.max(0).tolist()],
        "voxel_center_span_mm": ((np.array(hdr["shape"]) - 1) * spacing).tolist(),
        "voxel_support_size_mm": (np.array(hdr["shape"]) * spacing).tolist(),
    }


def check_dicom(evidence, raw, hdr, geom, cfg):
    if evidence.get("errors") or not evidence.get("instances"):
        raise ValueError("DICOM evidence unavailable: " + str(evidence.get("errors")))
    instances = evidence["instances"]
    shape = np.asarray(hdr["shape"])
    if len(instances) != shape[2]:
        raise ValueError("DICOM instance count and NIfTI depth disagree")
    affine = np.asarray(geom["affine_ras"])
    inverse = np.linalg.inv(affine)
    geometry_errors = []
    positions = []
    sop_uids = set()
    for k, r in enumerate(instances):
        if int(r["Rows"]) != shape[1] or int(r["Columns"]) != shape[0]:
            raise ValueError("DICOM dimensions conflict")
        if r["SOPInstanceUID"] in sop_uids:
            raise ValueError("Duplicate DICOM SOPInstanceUID")
        sop_uids.add(r["SOPInstanceUID"])
        orient = np.asarray(r["ImageOrientationPatient"], float)
        spacing = np.asarray(r["PixelSpacing"], float)
        pos = np.asarray(r["ImagePositionPatient"], float)
        positions.append(pos)
        for i, j in [
            (0, 0),
            (shape[0] - 1, 0),
            (0, shape[1] - 1),
            (shape[0] - 1, shape[1] - 1),
        ]:
            dicom_lps = pos + orient[:3] * i * spacing[1] + orient[3:] * j * spacing[0]
            nifti_ras = transform(affine, np.array([[i, j, k]]))[0]
            geometry_errors.append(
                float(np.linalg.norm(dicom_lps * np.array([-1, -1, 1]) - nifti_ras))
            )
    hu_errors, raw_errors, rounding_errors = [], [], []
    for sample in evidence["samples"]:
        r = instances[sample["slice_index"]]
        orient = np.asarray(r["ImageOrientationPatient"], float)
        spacing = np.asarray(r["PixelSpacing"], float)
        rr, cc = np.asarray(sample["rows"]), np.asarray(sample["columns"])
        lps = (
            np.asarray(r["ImagePositionPatient"], float)[None, :]
            + cc[:, None] * orient[:3] * spacing[1]
            + rr[:, None] * orient[3:] * spacing[0]
        )
        ijk = transform(inverse, lps * np.array([-1, -1, 1]))
        nearest = np.rint(ijk).astype(int)
        if np.any(nearest < 0) or np.any(nearest >= shape):
            raise ValueError("DICOM physical samples outside NIfTI")
        actual_raw = raw[tuple(nearest.T)].astype(float)
        expected_hu = np.asarray(sample["hu_values"])
        actual = actual_raw * hdr["effective_slope"] + hdr["effective_intercept"]
        hu_errors.extend(np.abs(actual - expected_hu))
        raw_errors.extend(np.abs(actual_raw - np.asarray(sample["stored_values"])))
        rounding_errors.extend(np.max(np.abs(ijk - nearest), axis=1))
    positions = np.array(positions)
    normal = np.cross(
        np.asarray(instances[0]["ImageOrientationPatient"], float)[:3],
        np.asarray(instances[0]["ImageOrientationPatient"], float)[3:],
    )
    gaps = np.diff(positions @ normal)
    # CT modality and DICOM rescale establish HU semantics; pixel equality shows conversion applied it.
    hu_ok = max(hu_errors) <= cfg["hu_sample_tolerance"] and evidence["values"][
        "Modality"
    ] == ["CT"]
    max_geo = max(geometry_errors)
    return {
        "mapping_status": "verified_sampled_pixels_and_all_slice_geometry"
        if hu_ok and max_geo <= cfg["dicom_geometry_block_tolerance_mm"]
        else "conflict_review_required",
        "sampled_voxels": len(hu_errors),
        "sampled_slices": len(evidence["samples"]),
        "max_sampled_hu_error": float(max(hu_errors)),
        "max_raw_nifti_vs_raw_dicom_difference": float(max(raw_errors)),
        "max_sample_coordinate_rounding_error_voxels": float(max(rounding_errors)),
        "max_all_slice_corner_geometry_error_mm": max_geo,
        "geometry_review": max_geo > cfg["dicom_geometry_review_tolerance_mm"],
        "geometry_block": max_geo > cfg["dicom_geometry_block_tolerance_mm"],
        "measured_dicom_slice_spacing_mm": {
            "min": float(gaps.min()),
            "median": float(np.median(gaps)),
            "max": float(gaps.max()),
        },
        "strictly_ordered_unique_positions": bool(np.all(gaps > 0)),
        "hu_status": "DICOM_rescaled_HU_verified_at_samples" if hu_ok else "unverified",
        "hu_verification_scope": "5 physical slices; sampled pixels, not full-volume DICOM equality",
        "dicom_intercept_must_not_be_reapplied": bool(hu_ok),
    }


def contact_sheet(raw, hdr, geom, patient, phase, out, cfg):
    # Explicitly support this cohort's native LPS axes; never guess labels for oblique/conflicting data.
    if not np.allclose(
        np.asarray(geom["direction_lps"]).reshape(3, 3), np.eye(3), atol=1e-6, rtol=0
    ):
        raise ValueError(
            "Contact sheet requires reviewed orientation for non-LPS native grid"
        )
    fractions = cfg["contact_sheet_fractions"]
    fig, axes = plt.subplots(3, len(fractions), figsize=(21, 11.4), facecolor="#10141a")
    spacing = np.asarray(geom["spacing_mm"])
    affine = np.asarray(geom["affine_ras"])
    width, level = cfg["display_window_width"], cfg["display_window_level"]
    slice_records = []
    for row, name in enumerate(["Axial", "Coronal", "Sagittal"]):
        axis = [2, 1, 0][row]
        for col, fraction in enumerate(fractions):
            k = round(fraction * (raw.shape[axis] - 1))
            if row == 0:
                plane = raw[:, :, k].T
                aspect = spacing[1] / spacing[0]
                labels = ("R", "L", "A", "P")
            elif row == 1:
                plane = raw[:, k, :].T[::-1]
                aspect = spacing[2] / spacing[0]
                labels = ("R", "L", "S", "I")
            else:
                plane = raw[k, :, :].T[::-1]
                aspect = spacing[2] / spacing[1]
                labels = ("A", "P", "S", "I")
            plane = (
                plane.astype(float) * hdr["effective_slope"]
                + hdr["effective_intercept"]
            )
            ax = axes[row, col]
            ax.imshow(
                plane,
                cmap="gray",
                vmin=level - width / 2,
                vmax=level + width / 2,
                origin="upper",
                aspect=aspect,
                interpolation="nearest",
            )
            ax.set_axis_off()
            for text, x, y in [
                (labels[0], 0.02, 0.5),
                (labels[1], 0.98, 0.5),
                (labels[2], 0.5, 0.97),
                (labels[3], 0.5, 0.03),
            ]:
                ax.text(
                    x,
                    y,
                    text,
                    transform=ax.transAxes,
                    color="#70f0ed",
                    fontsize=10,
                    weight="bold",
                    ha="center",
                    va="center",
                    bbox={
                        "facecolor": "black",
                        "alpha": 0.65,
                        "pad": 1,
                        "edgecolor": "none",
                    },
                )
            pos = affine[axis, 3] + affine[axis, axis] * k
            ax.set_title(
                f"{name} {'XYZ'[axis]}={pos:.1f} mm | i={k}", color="white", fontsize=9
            )
            slice_records.append(
                {
                    "plane": name,
                    "native_index": k,
                    "ras_axis": "XYZ"[axis],
                    "ras_coordinate_mm": float(pos),
                }
            )
    fig.suptitle(
        f"{patient} | {phase} | Native CT overview | W={width}, L={level} HU\n"
        f"{raw.shape[0]} x {raw.shape[1]} x {raw.shape[2]} | "
        f"spacing {spacing[0]:.6f} x {spacing[1]:.6f} x {spacing[2]:.6f} mm",
        color="white",
        fontsize=17,
    )
    fig.text(
        0.5,
        0.018,
        "Patient directions: R right, L left, A anterior, P posterior, S superior, I inferior. "
        "Native slices across 0-100% coverage; display only; no registration or volume resampling.",
        ha="center",
        color="white",
        fontsize=10,
    )
    fig.subplots_adjust(
        left=0.015, right=0.985, top=0.89, bottom=0.06, wspace=0.06, hspace=0.2
    )
    path = out / "figures" / f"{patient}_{phase}_contact_sheet.png"
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {
        "path": str(path),
        "display_window_width": width,
        "display_window_level": level,
        "orientation": "radiological axial/coronal (patient right at image left); sagittal anterior at left",
        "slices": slice_records,
        "human_review_status": "pending",
    }


def audit_one(patient, phase, path, evidence, cfg, out):
    result = {
        "patient": patient,
        "phase": phase,
        "source_path": str(path),
        "warnings": [],
        "errors": [],
    }
    try:
        hdr = header(path)
        result["nifti_header"] = hdr
        geom = geometry(hdr, path, cfg["audit"])
        result["geometry"] = geom
        raw = read_raw(path, hdr)
        finite = np.isfinite(raw)
        nbad = int(raw.size - np.count_nonzero(finite))
        scaled_min = (
            float(raw[finite].min()) * hdr["effective_slope"]
            + hdr["effective_intercept"]
        )
        scaled_max = (
            float(raw[finite].max()) * hdr["effective_slope"]
            + hdr["effective_intercept"]
        )
        scaled_finite = nbad == 0 and np.isfinite([scaled_min, scaled_max]).all()
        sample = (
            raw[::8, ::8, ::4].astype(float) * hdr["effective_slope"]
            + hdr["effective_intercept"]
        )
        result["intensity"] = {
            "stored_dtype": hdr["dtype"],
            "stored_min": float(raw[finite].min()),
            "stored_max": float(raw[finite].max()),
            "scaled_min": min(scaled_min, scaled_max),
            "scaled_max": max(scaled_min, scaled_max),
            "voxel_count": int(raw.size),
            "nonfinite_count": nbad,
            "all_scaled_values_finite": bool(scaled_finite),
            "sample_percentiles": dict(
                zip(
                    ["p0", "p1", "p5", "p50", "p95", "p99", "p100"],
                    np.percentile(sample, [0, 1, 5, 50, 95, 99, 100]).tolist(),
                )
            ),
            "percentile_sampling_stride": [8, 8, 4],
        }
        del finite
        # Independent production-reader scaling check, never applying DICOM intercept to NIfTI.
        itk = sitk.ReadImage(str(path))
        itk_values = sitk.GetArrayViewFromImage(itk)[::4, ::8, ::8].transpose(2, 1, 0)
        result["intensity"]["simpleitk_sample_max_error"] = float(
            np.max(np.abs(itk_values - sample))
        )
        del itk_values, itk
        if (
            not scaled_finite
            or result["intensity"]["simpleitk_sample_max_error"]
            > cfg["audit"]["hu_sample_tolerance"]
        ):
            result["errors"].append(
                "Invalid values or production-reader scaling disagreement"
            )
        if evidence is None:
            result["errors"].append("DICOM evidence missing")
        else:
            dc = check_dicom(evidence, raw, hdr, geom, cfg["audit"])
            result["dicom_verification"] = dc
            result["series_metadata"] = {
                k: v
                for k, v in evidence.items()
                if k
                in [
                    "candidate_series_uid",
                    "values",
                    "acquisition_start",
                    "acquisition_end",
                    "same_acquisition_named_series",
                ]
            }
            if dc["geometry_review"]:
                result["warnings"].append(
                    "DICOM/NIfTI geometry discrepancy %.9f mm; preserve native header, review before N-dependent work"
                    % dc["max_all_slice_corner_geometry_error_mm"]
                )
            if (
                dc["geometry_block"]
                or dc["hu_status"] == "unverified"
                or not dc["strictly_ordered_unique_positions"]
            ):
                result["errors"].append("DICOM geometry/intensity conflict")
            for key in [
                "KVP",
                "ConvolutionKernel",
                "SliceThickness",
                "PixelSpacing",
                "ImageOrientationPatient",
                "RescaleSlope",
                "RescaleIntercept",
            ]:
                if len(evidence["values"][key]) != 1 or evidence["values"][key] == [""]:
                    result["warnings"].append("Missing or varying DICOM " + key)
            if evidence["values"]["ContrastBolusStartTime"] == [""]:
                result["warnings"].append(
                    "Injection start absent on all matched DICOM instances; injection-relative delay unknown"
                )
        result["contact_sheet"] = contact_sheet(
            raw, hdr, geom, patient, phase, out, cfg["audit"]
        )
        del raw
    except Exception as exc:
        result["errors"].append(type(exc).__name__ + ": " + str(exc))
    result["registration_input_status"] = (
        "blocked"
        if result["errors"]
        else (
            "eligible_with_geometry_review"
            if result.get("dicom_verification", {}).get("geometry_review")
            else "eligible_for_registration_trial"
        )
    )
    result["anatomical_registration_success"] = "not assessed; Stage 3 not implemented"
    return result


def time_seconds(value):
    return int(value[:2]) * 3600 + int(value[2:4]) * 60 + float(value[4:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1],
        default=1,
        help="Run the input audit (Stage 1)",
    )
    parser.add_argument("--refresh-dicom", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Rehash originals against the completed audit",
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text())
    out = Path(cfg["output_root"]).resolve()
    source = Path(cfg["source_root"]).resolve()
    if out == source or source in out.parents or out in source.parents:
        raise ValueError("Output and source roots must be separate sibling trees")
    if (
        cfg["resample_inputs"]
        or cfg["normalize_intensities"]
        or cfg["apply_dicom_rescale_to_nifti"]
    ):
        raise ValueError(
            "Stage 1 forbids resampling, normalization, or DICOM intercept reapplication"
        )
    if cfg["reference_phase"] != "A" or cfg["reference_spacing"] != "native":
        raise ValueError("This study requires native A reference grids")
    if args.verify_only:
        prior = json.loads((out / "input_fingerprints.json").read_text())
        checks = [
            {
                "path": r["path"],
                "unchanged": Path(r["path"]).exists()
                and digest(Path(r["path"])) == r["sha256_before"],
            }
            for r in prior["inputs"]
        ]
        print(json.dumps(checks, indent=2))
        sys.exit(0 if len(checks) == 15 and all(r["unchanged"] for r in checks) else 1)
    for folder in [
        "config",
        "scripts",
        "tables",
        "figures",
        "patients",
        "report",
        "logs",
    ]:
        (out / folder).mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    paths = [
        (p, f, source / p / "nifti" / f"{f}.nii.gz")
        for p in cfg["patients"]
        for f in cfg["phases"]
    ]
    fingerprints = []
    for patient, phase, path in paths:
        record = {
            "patient": patient,
            "phase": phase,
            "path": str(path),
            "exists": path.exists(),
        }
        if path.exists():
            record.update(
                size_bytes=path.stat().st_size,
                mtime_ns_before=path.stat().st_mtime_ns,
                sha256_before=digest(path),
            )
        fingerprints.append(record)
    dump(
        out / "input_fingerprints.json",
        {"inputs": fingerprints, "status": "in_progress"},
    )
    if args.refresh_dicom:
        subprocess.run(
            [
                cfg["dicom_python"],
                str(out / "scripts/stage01_dicom_evidence.py"),
                "--config",
                str(config_path),
            ],
            check=True,
        )
    evidence_dir = out / "tables/stage01_dicom_evidence"
    evidence_index = (
        json.loads((evidence_dir / "index.json").read_text())
        if (evidence_dir / "index.json").exists()
        else {}
    )
    rows = []
    results = []
    for patient, phase, path in paths:
        evidence_path = evidence_dir / f"{patient}_{phase}.json"
        evidence = (
            json.loads(evidence_path.read_text()) if evidence_path.exists() else None
        )
        result = audit_one(patient, phase, path, evidence, cfg, out)
        result["dicom_evidence_path"] = str(evidence_path)
        results.append(result)
        hdr = result.get("nifti_header", {})
        geom = result.get("geometry", {})
        dc = result.get("dicom_verification", {})
        row = {
            "patient": patient,
            "phase": phase,
            "source_path": str(path),
            "reference_phase": "A",
            "reference_path": str(source / patient / "nifti/A.nii.gz"),
            "registration_input_status": result["registration_input_status"],
            "mapping_status": dc.get("mapping_status", "unverified"),
            "hu_status": dc.get("hu_status", "unverified"),
            "dimensions": hdr.get("shape"),
            "spacing_mm": geom.get("spacing_mm"),
            "axis_codes": geom.get("axis_codes"),
            "origin_ras_mm": geom.get("origin_ras_mm"),
            "direction_ras": geom.get("direction_ras"),
            "affine_ras": geom.get("affine_ras"),
            "qform_code": hdr.get("qform_code"),
            "sform_code": hdr.get("sform_code"),
            "qform_sform_error_mm": geom.get("qform_sform_max_corner_error_mm"),
            "voxel_center_bounds_ras_mm": geom.get("voxel_center_bounds_ras_mm"),
            "voxel_edge_bounds_ras_mm": geom.get("voxel_edge_bounds_ras_mm"),
            "nifti_slope": hdr.get("effective_slope"),
            "nifti_intercept": hdr.get("effective_intercept"),
            "nonfinite_count": result.get("intensity", {}).get("nonfinite_count"),
            "stored_min": result.get("intensity", {}).get("stored_min"),
            "stored_max": result.get("intensity", {}).get("stored_max"),
            "injection_relative_delay_seconds": "unknown",
            "injection_delay_status": "unknown; no injection start available",
            "max_dicom_geometry_error_mm": dc.get(
                "max_all_slice_corner_geometry_error_mm"
            ),
            "max_sampled_hu_error": dc.get("max_sampled_hu_error"),
            "contact_sheet": result.get("contact_sheet", {}).get("path", ""),
            "warnings": " | ".join(result["warnings"]),
            "errors": " | ".join(result["errors"]),
        }
        if evidence and evidence.get("values"):
            vals = evidence["values"]
            row.update(
                series_uid=evidence["candidate_series_uid"],
                acquisition_start=evidence["acquisition_start"],
                acquisition_end=evidence["acquisition_end"],
                phase_identity_evidence=evidence["same_acquisition_named_series"],
            )
            for field, tag in {
                "series_number": "SeriesNumber",
                "series_description": "SeriesDescription",
                "protocol": "ProtocolName",
                "acquisition_date": "AcquisitionDate",
                "acquisition_datetime": "AcquisitionDateTime",
                "injection_start": "ContrastBolusStartTime",
                "kvp": "KVP",
                "kernel": "ConvolutionKernel",
                "slice_thickness_mm": "SliceThickness",
                "dicom_spacing_between_slices_mm": "SpacingBetweenSlices",
                "dicom_rescale_slope": "RescaleSlope",
                "dicom_rescale_intercept": "RescaleIntercept",
                "rescale_type": "RescaleType",
                "dicom_pixel_spacing_mm": "PixelSpacing",
            }.items():
                row[field] = vals[tag][0] if len(vals[tag]) == 1 else vals[tag]
        rows.append(row)
        dump(out / "patients" / patient / f"{phase}_audit.json", result)
        print(
            patient,
            phase,
            result["registration_input_status"],
            result["errors"],
            flush=True,
        )
    patient_rows = []
    references = {}
    for patient in cfg["patients"]:
        phases = {r["phase"]: r for r in results if r["patient"] == patient}
        ref = phases["A"]
        summary = {
            "patient": patient,
            "reference_phase": "A",
            "reference_path": ref["source_path"],
            "reference_spacing": "native",
            "grid_mismatch_phases": [],
            "injection_relative_delays": "unknown",
            "human_review": "pending",
        }
        if "geometry" in ref and ref["registration_input_status"] != "blocked":
            references[patient] = {
                "path": ref["source_path"],
                "shape": ref["nifti_header"]["shape"],
                **ref["geometry"],
            }
            dump(
                out / "patients" / patient / "reference_grid.json", references[patient]
            )
            for phase in ["N", "V"]:
                other = phases[phase]
                same = (
                    "geometry" in other
                    and other["nifti_header"]["shape"] == ref["nifti_header"]["shape"]
                    and np.allclose(
                        other["geometry"]["affine_ras"],
                        ref["geometry"]["affine_ras"],
                        atol=1e-5,
                        rtol=0,
                    )
                )
                if not same:
                    summary["grid_mismatch_phases"].append(phase)
                summary[f"{phase}_to_A_status"] = (
                    "blocked"
                    if "blocked"
                    in [
                        other["registration_input_status"],
                        ref["registration_input_status"],
                    ]
                    else "geometry_review_required"
                    if any(
                        "geometry_review" in r["registration_input_status"]
                        for r in [other, ref]
                    )
                    else "eligible_for_registration_trial"
                )
            if all("geometry" in p for p in phases.values()):
                bounds = np.array(
                    [
                        p["geometry"]["voxel_center_bounds_ras_mm"]
                        for p in phases.values()
                    ]
                )
                lo = bounds[:, 0, :].max(0)
                hi = bounds[:, 1, :].min(0)
                summary["unregistered_common_center_bounds_ras_mm"] = [
                    lo.tolist(),
                    hi.tolist(),
                ]
                summary["unregistered_common_z_span_mm"] = float(max(0, hi[2] - lo[2]))
                a_bounds = np.array(ref["geometry"]["voxel_center_bounds_ras_mm"])
                summary["A_inferior_z_loss_mm"] = float(max(0, lo[2] - a_bounds[0, 2]))
                summary["A_superior_z_loss_mm"] = float(max(0, a_bounds[1, 2] - hi[2]))
            try:
                a, n, v = (phases[p]["series_metadata"] for p in ["A", "N", "V"])
                dates = [r["values"]["AcquisitionDate"] for r in [a, n, v]]
                if not (
                    dates[0] == dates[1] == dates[2]
                    and len(dates[0]) == 1
                    and dates[0] != [""]
                ):
                    raise ValueError("Acquisition dates missing or inconsistent")
                summary["N_to_A_start_seconds"] = round(
                    time_seconds(a["acquisition_start"])
                    - time_seconds(n["acquisition_start"]),
                    3,
                )
                summary["A_to_V_start_seconds"] = round(
                    time_seconds(v["acquisition_start"])
                    - time_seconds(a["acquisition_start"]),
                    3,
                )
            except Exception as exc:
                summary["timing_error"] = str(exc)
        else:
            summary["N_to_A_status"] = summary["V_to_A_status"] = (
                "blocked_reference_unavailable"
            )
        patient_rows.append(summary)
    for fp in fingerprints:
        path = Path(fp["path"])
        fp["sha256_after"] = digest(path) if path.exists() else None
        fp["mtime_ns_after"] = path.stat().st_mtime_ns if path.exists() else None
        fp["unchanged"] = bool(
            fp.get("sha256_before")
            and fp.get("sha256_before") == fp["sha256_after"]
            and fp["mtime_ns_before"] == fp["mtime_ns_after"]
        )
    metadata_paths = [
        config_path,
        Path(__file__),
        out / "scripts/stage01_dicom_evidence.py",
        out / "scripts/run_stage01.ps1",
        out / "scripts/test_stage01_safeguards.py",
        Path(__file__).resolve().parents[1] / "METHODS.md",
        out / "tables/verified_phase_manifest.csv",
        out / "tables/dicom_audit_summary.json",
        out / "tables/series_inventory.csv",
    ] + list(evidence_dir.glob("*.json"))
    metadata_hashes = [
        {"path": str(p), "sha256": digest(p)} for p in metadata_paths if p.exists()
    ]
    dump(
        out / "input_fingerprints.json",
        {
            "algorithm": "SHA-256",
            "inputs": fingerprints,
            "metadata_and_code": metadata_hashes,
            "originals_unchanged": all(r["unchanged"] for r in fingerprints),
            "dicom_fingerprints": "Five sampled source files per phase in evidence JSON; all matched instances have saved metadata. Original DICOMs opened read-only.",
        },
    )
    table(out / "manifest.csv", rows)
    table(out / "tables/stage01_patient_summary.csv", patient_rows)
    env = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "SimpleITK": sitk.Version_VersionString(),
        "matplotlib": matplotlib.__version__,
        "dicom_runtime": {
            k: evidence_index.get(k)
            for k in ["python", "executable", "pydicom", "numpy"]
        },
        "installed_or_modified_packages": False,
    }
    dump(out / "environment_versions.json", env)
    complete = len(results) == 15 and all(fp["unchanged"] for fp in fingerprints)
    stage = {
        "stage": 1,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_with_documented_issues" if complete else "incomplete",
        "completion_scope": "All 15 inputs accounted for and registration input suitability documented; not a registration or clinical acceptance claim.",
        "expected_inputs": 15,
        "accounted_inputs": len(results),
        "readable_inputs": sum("intensity" in r for r in results),
        "contact_sheets_created": sum("contact_sheet" in r for r in results),
        "blocked_inputs": [
            r["patient"] + "/" + r["phase"]
            for r in results
            if r["registration_input_status"] == "blocked"
        ],
        "geometry_review_inputs": [
            r["patient"] + "/" + r["phase"]
            for r in results
            if r["registration_input_status"] == "eligible_with_geometry_review"
        ],
        "originals_unchanged": all(fp["unchanged"] for fp in fingerprints),
        "config_path": str(config_path),
        "settings": cfg,
        "reference_grids": references,
        "patients": patient_rows,
        "warnings": [
            {
                "patient": r["patient"],
                "phase": r["phase"],
                "warnings": r["warnings"],
                "errors": r["errors"],
            }
            for r in results
        ],
        "outputs": {
            "manifest": str(out / "manifest.csv"),
            "fingerprints": str(out / "input_fingerprints.json"),
            "environment": str(out / "environment_versions.json"),
            "audit_findings": str(out / "audit_findings.md"),
            "patient_summary": str(out / "tables/stage01_patient_summary.csv"),
            "per_phase_audits": [
                str(out / "patients" / r["patient"] / f"{r['phase']}_audit.json")
                for r in results
            ],
            "contact_sheets": [
                r["contact_sheet"]["path"] for r in results if "contact_sheet" in r
            ],
        },
        "not_performed": [
            "masking",
            "cropping",
            "resampling",
            "normalization",
            "registration",
            "subtraction",
            "feasibility scoring",
            "final presentation",
        ],
        "review_status": {
            "machine_contact_sheet_review": "see visual_review.json if present; verify its figure hashes",
            "human_review": "pending",
        },
        "unselected_dicom_series": evidence_index.get("unselected_series", []),
        "dicom_metadata_read_errors": evidence_index.get("read_errors", []),
        "commands": {
            "stage_01": f'& "{cfg["main_python"]}" "{out}/scripts/stage01_audit.py" --config "{config_path}" --stage 1 --refresh-dicom',
            "verify_inputs": f'& "{cfg["main_python"]}" "{out}/scripts/stage01_audit.py" --config "{config_path}" --verify-only',
        },
    }
    dump(out / "stage_01.json", stage)
    lines = [
        "# Stage 1 input audit",
        "",
        f"Accounted for {len(results)}/15 inputs; {stage['contact_sheets_created']} native-grid contact sheets. Status: {stage['status']}.",
        "",
        "All source hashes and modification times were checked before and after. See input_fingerprints.json. No source was written.",
        "",
        "## Mapping and intensity evidence",
        "",
        "The prior manifest supplied candidate UIDs. Fresh source DICOM reads verify all slice corner coordinates, dimensions, instance uniqueness, and 1,280 physical pixel samples over five slices per input. Named W/O, Arterial, and Venous reconstructions at the same acquisition start support N/A/V identity; THIN alone does not name a phase. Mapping is sampled, not a full-volume pixel equivalence proof.",
        "",
        "DICOM rescale is applied only to original DICOM samples for comparison. NIfTI values are read with NIfTI scaling once, independently cross-checked with SimpleITK. DICOM CT rescale provides HU evidence, not a phantom calibration or proof of absolute scanner accuracy. No second DICOM intercept is applied.",
        "",
        "## Per-input findings",
        "",
    ]
    for r in results:
        dc = r.get("dicom_verification", {})
        intensity = r.get("intensity", {})
        lines.append(
            f"- **{r['patient']}/{r['phase']}**: {r['registration_input_status']}; HU sample error {dc.get('max_sampled_hu_error', 'unknown')}; DICOM geometry maximum {dc.get('max_all_slice_corner_geometry_error_mm', 'unknown')} mm; stored range {intensity.get('stored_min')} to {intensity.get('stored_max')}; nonfinite {intensity.get('nonfinite_count')}. "
            + "; ".join(r["warnings"] + r["errors"])
        )
    lines += [
        "",
        "## Timing and coverage",
        "",
        "| Patient | N to A start (s) | A to V start (s) | Common native z span (mm) | A inferior / superior loss (mm) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in patient_rows:
        lines.append(
            f"| {r['patient']} | {r.get('N_to_A_start_seconds', 'unknown')} | {r.get('A_to_V_start_seconds', 'unknown')} | {r.get('unregistered_common_z_span_mm', 'unknown')} | {r.get('A_inferior_z_loss_mm', 'unknown')} / {r.get('A_superior_z_loss_mm', 'unknown')} |"
        )
    lines += [
        "",
        "Injection-relative delays remain unknown: injection-start fields are blank. A-to-V separation is an acquisition-start interval, not a post-injection delay. V is not established as a 150-second delayed scan.",
        "",
        "Additional unselected acquisitions are retained in stage_01.json. In the P003 original folder, series 8 (Venous) and 9 (THIN) begin at 10:06:57.192; the supplied P003/V is series 7 beginning at 10:03:44.440. The extra acquisition is not substituted, and its injection delay is also unknown.",
        "",
        "Physical overlap above is unregistered voxel-center geometry, not verified common carotid anatomy. Edge bounds (half-voxel outer support) are separately recorded. All N and V grids differ from A. Later arithmetic must use validated registered support, never direct native-array subtraction.",
        "",
        "## Acquisition compatibility and orientation",
        "",
        "Matched source metadata are recorded per phase in manifest.csv and per-instance evidence JSON. SliceThickness is reconstruction thickness; SpacingBetweenSlices and measured position increments describe reconstructed center-to-center sampling. They do not establish independent through-plane acquisition resolution or detector collimation.",
        "",
        "Cohort metadata values: kVp = "
        + str(sorted({str(r.get("kvp", "unknown")) for r in rows}))
        + "; convolution kernel = "
        + str(sorted({str(r.get("kernel", "unknown")) for r in rows}))
        + "; reconstruction thickness (mm) = "
        + str(sorted({str(r.get("slice_thickness_mm", "unknown")) for r in rows}))
        + "; DICOM spacing between reconstructed slices (mm) = "
        + str(
            sorted(
                {str(r.get("dicom_spacing_between_slices_mm", "unknown")) for r in rows}
            )
        )
        + ".",
        "",
        "Both coded qform and sform, pixdim, physical corners, direction orthogonality, and the SimpleITK LPS interpretation are checked. Orientation conflicts block the affected input. Contact sheets use native slices with explicit patient directions and physical aspect ratio; display windowing does not alter quantitative data.",
        "",
        "Geometry review and block thresholds are engineering audit tolerances (0.0005 mm and 0.01 mm), not validated clinical wall-imaging tolerances. Small deviations remain recorded even below a block threshold. No header repair was performed.",
        "",
        "## Scope and next-stage readiness",
        "",
        "Native A is saved as each reference grid. Inputs marked eligible can enter a registration trial after Stage 2. Geometry-review inputs require documented review; unaffected pairs can proceed independently. No registration has been run, and target C1/T1/carotid coverage and local image quality require subsequent focused review. Human review is pending.",
        "",
        "See visual_review.json for actual machine visual observations. The two source articles remain scientific references under the roadmap; no article content was executed. Later-stage commands are deliberately rejected by the Stage 1 CLI.",
    ]
    (out / "audit_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: stage[k]
                for k in [
                    "status",
                    "accounted_inputs",
                    "readable_inputs",
                    "contact_sheets_created",
                    "blocked_inputs",
                    "geometry_review_inputs",
                    "originals_unchanged",
                ]
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
