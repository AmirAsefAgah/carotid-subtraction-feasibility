"""Independent integrity validation for Stage 4 outputs."""

from __future__ import annotations

import json
import gc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from study_paths import ROOT

PATIENTS = ("P001", "P002", "P003", "P004", "P005")
FORMULAS = {
    "arterial_enhancement": lambda a, n, v: a - n,
    "venous_enhancement": lambda a, n, v: v - n,
    "temporal_difference": lambda a, n, v: v - a,
    "dark_lumen_N": lambda a, n, v: np.float32(2) * n - a,
    "dark_lumen_V": lambda a, n, v: np.float32(2) * v - a,
}


def same_geometry(a, b):
    return a.GetSize() == b.GetSize() and all(
        np.allclose(x, y, atol=1e-6, rtol=0)
        for x, y in (
            (a.GetSpacing(), b.GetSpacing()),
            (a.GetOrigin(), b.GetOrigin()),
            (a.GetDirection(), b.GetDirection()),
        )
    )


def header(path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return reader


def block(path, z0, depth):
    reader = header(path)
    reader.SetExtractIndex([0, 0, z0])
    reader.SetExtractSize([reader.GetSize()[0], reader.GetSize()[1], depth])
    return sitk.GetArrayFromImage(reader.Execute()).astype(np.float32, copy=False)


def same_header(a, b):
    return a.GetSize() == b.GetSize() and all(
        np.allclose(x, y, atol=1e-6, rtol=0)
        for x, y in (
            (a.GetSpacing(), b.GetSpacing()),
            (a.GetOrigin(), b.GetOrigin()),
            (a.GetDirection(), b.GetDirection()),
        )
    )


def main() -> int:
    summary = json.loads((ROOT / "stage_04.json").read_text(encoding="utf-8"))
    errors = []
    checks = []
    records = {p["patient"]: p for p in summary.get("patients", [])}
    if set(records) != set(PATIENTS):
        errors.append(f"Expected five patients, found {sorted(records)}")
    for patient in PATIENTS:
        if patient not in records:
            continue
        record = records[patient]
        stage3 = ROOT / "patients" / patient / "stage_03"
        paths = {
            "A": stage3 / f"{patient}_A_native_crop.nii.gz",
            "N": stage3 / f"{patient}_N_registered_to_A.nii.gz",
            "V": stage3 / f"{patient}_V_registered_to_A.nii.gz",
        }
        a_img = header(paths["A"])
        common_path = Path(record["masks"]["NAV_common"]["path"])
        common_img = header(common_path)
        patient_check = {
            "patient": patient,
            "products": {},
            "identity_max_abs_error_hu": 0.0,
        }
        product_images = {}
        for name, product in record["products"].items():
            image_path = Path(product["quantitative_image"]["path"])
            mask_path = Path(product["validity_mask_path"])
            image = header(image_path)
            mask = header(mask_path)
            product_images[name] = image_path
            if image.GetPixelID() != sitk.sitkFloat32:
                errors.append(f"{patient} {name}: output is not float32")
            if not same_header(a_img, image) or not same_header(a_img, mask):
                errors.append(f"{patient} {name}: geometry mismatch")
            needed = record["products"][name]["required_phases"]
            arrays = {
                phase: sitk.GetArrayFromImage(sitk.ReadImage(str(paths[phase]))).astype(
                    np.float32, copy=False
                )
                for phase in needed
            }
            out = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path))).astype(
                np.float32, copy=False
            )
            valid = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))) > 0
            a = arrays.get("A")
            n = arrays.get("N")
            v = arrays.get("V")
            expected = FORMULAS[name](a, n, v)
            max_error = (
                float(np.max(np.abs(out[valid] - expected[valid])))
                if valid.any()
                else float("inf")
            )
            nonfinite = int((~np.isfinite(out[valid])).sum())
            outside_nonzero = int(np.count_nonzero(out[~valid]))
            if max_error > 0.001:
                errors.append(f"{patient} {name}: formula max error {max_error} HU")
            if nonfinite:
                errors.append(f"{patient} {name}: {nonfinite} nonfinite valid voxels")
            if outside_nonzero:
                errors.append(
                    f"{patient} {name}: {outside_nonzero} nonzero exterior voxels"
                )
            patient_check["products"][name] = {
                "formula_max_abs_error_hu": max_error,
                "nonfinite_valid_voxels": nonfinite,
                "nonzero_outside_mask_voxels": outside_nonzero,
            }
            del arrays, out, valid, expected
            gc.collect()
        identity_max = 0.0
        arrays = {
            name: sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(
                np.float32, copy=False
            )
            for name, path in product_images.items()
            if name
            in ("arterial_enhancement", "venous_enhancement", "temporal_difference")
        }
        valid = sitk.GetArrayFromImage(sitk.ReadImage(str(common_path))) > 0
        error = np.abs(
            arrays["temporal_difference"]
            - (arrays["venous_enhancement"] - arrays["arterial_enhancement"])
        )
        identity_max = float(error[valid].max()) if valid.any() else float("inf")
        if identity_max > 0.001:
            errors.append(f"{patient}: identity max error {identity_max} HU")
        patient_check["identity_max_abs_error_hu"] = identity_max
        del arrays, valid, error
        gc.collect()
        for figure in record["figures"].values():
            if not Path(figure["path"]).exists():
                errors.append(f"{patient}: missing figure {figure['path']}")
        checks.append(patient_check)
    report = {
        "stage": 4,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "scope": "float32, geometry, formula signs/weights, finite valid values, masked zero fill, arithmetic identity, figures; not anatomical acceptance",
        "tolerance": {"absolute_hu": 0.001, "basis": "float32 arithmetic"},
        "errors": errors,
        "patients": checks,
    }
    path = ROOT / "logs" / "stage04_validation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary["independent_validation"] = {
        "path": str(path),
        "status": report["status"],
        "validated_at_utc": report["validated_at_utc"],
        "scope": report["scope"],
        "error_count": len(errors),
    }
    (ROOT / "stage_04.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"status": report["status"], "errors": errors, "report": str(path)},
            indent=2,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
