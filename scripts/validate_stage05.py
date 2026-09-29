"""Independent Stage 5 numerical and missing-data checks against saved ROI labels."""

from pathlib import Path
import json
import csv
import numpy as np
import SimpleITK as sitk
from stage05_assessment import ROOT, stats, sphere_indices, write_json


def slab(path, start, depth):
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()
    r.SetExtractIndex([0, 0, start])
    r.SetExtractSize([r.GetSize()[0], r.GetSize()[1], depth])
    return sitk.GetArrayFromImage(r.Execute())


def main():
    # Sign cancellation and sample SD, plus a rotated physical-space ROI.
    assert stats([-10, 10])["mean"] == 0 and stats([-10, 10])["mean_abs"] == 10
    assert np.isclose(stats([1, 2, 3])["sd"], 1)
    im = sitk.Image([21, 21, 21], sitk.sitkFloat32)
    im.SetSpacing([0.5, 1, 2])
    im.SetOrigin([5, -8, 10])
    im.SetDirection([0, -1, 0, 1, 0, 0, 0, 0, 1])
    center = im.TransformIndexToPhysicalPoint((10, 10, 10))
    idx = sphere_indices(im, center, 2)
    for z, y, x in zip(*idx):
        point = im.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
        assert np.linalg.norm(np.array(point) - center) <= 2 + 1e-8
    assert stats([])["mean"] is None
    s = json.loads((ROOT / "stage_05.json").read_text())
    s4 = json.loads((ROOT / "stage_04.json").read_text())
    assert len(s["patient_product_summary"]) == 25 and len(s["validation"]) == 5
    errors = []
    checks = []
    for p in s4["patients"]:
        pid = p["patient"]
        d = ROOT / "patients" / pid / "stage_03"
        print(pid, flush=True)
        saved = json.loads(
            (
                ROOT / "patients" / pid / "stage_05" / f"{pid}_assessment.json"
            ).read_text()
        )
        labels = sitk.GetArrayFromImage(
            sitk.ReadImage(
                str(
                    ROOT
                    / "patients"
                    / pid
                    / "stage_05"
                    / f"{pid}_assessment_roi_labels.nii.gz"
                )
            )
        )
        zz = np.where(np.any(labels > 0, axis=(1, 2)))[0]
        start = int(zz.min())
        depth = int(zz.max() - start + 1)
        labels = labels[start : start + depth]
        assert set(np.unique(labels)) == {0, 1, 2, 3, 4}
        valid = slab(p["masks"]["NAV_common"]["path"], start, depth) > 0
        paths = {
            ph: d
            / f"{pid}_{'A_native_crop' if ph == 'A' else ph + '_registered_to_A'}.nii.gz"
            for ph in ["A", "N", "V"]
        }
        paths.update(
            {k: Path(v["quantitative_image"]["path"]) for k, v in p["products"].items()}
        )
        maximum = 0.0
        for name, path in paths.items():
            arr = slab(path, start, depth)
            for label, key in enumerate(["lumen_R", "lumen_L", "muscle", "bone"], 1):
                values = arr[(labels == label) & valid].astype(float)
                rec = saved["roi_statistics"][name][key]
                assert len(values) == rec["n"] and len(values) > 1
                for metric, value in [
                    ("mean", values.mean()),
                    ("median", np.median(values)),
                    ("sd", values.std(ddof=1)),
                    ("mean_abs", np.abs(values).mean()),
                ]:
                    delta = abs(float(value) - rec[metric])
                    maximum = max(maximum, delta)
                    if delta > 1e-6:
                        errors.append(f"{pid}/{name}/{key}/{metric}: {delta}")
        checks.append(
            {
                "patient": pid,
                "independent_label_sample_statistics_max_error_hu": maximum,
            }
        )
    for row in s["patient_product_summary"]:
        assert (
            row["wall_cnr"] is None
            and row["wall_minus_lumen_hu"] is None
            and row["local_alignment_error_mm"] is None
            and row["quality_score"] is None
        )
        if row["product"].startswith("dark"):
            assert row["bone_mean_absolute_residual_hu"] is None
        assert row["status"] in ["accepted", "limited", "failed", "pending-review"]
    for product, summary in s["cohort_summary"].items():
        assert sum(summary["status_counts"].values()) == 5
        rows = [r for r in s["patient_product_summary"] if r["product"] == product]
        assert np.isclose(
            np.median([r["lumen_mean_hu"] for r in rows]),
            summary["descriptive"]["lumen_mean_hu"]["median"],
        )
    for f, expected in [
        ("stage05_patient_product.csv", 25),
        ("stage05_carotid_product.csv", 50),
        ("stage05_roi_coverage.csv", 20),
        ("stage05_phase_attenuation.csv", 10),
    ]:
        with (ROOT / "tables" / f).open() as h:
            assert len(list(csv.DictReader(h))) == expected
    result = {
        "stage": 5,
        "status": "PASS" if not errors else "FAIL",
        "checks": checks,
        "errors": errors,
        "scope": "Independent saved-label ROI sampling from actual images; means, medians, SD, mean absolute residual, physical sphere orientation, patient aggregation and honest missing values. Not validation of anatomy or human review.",
    }
    write_json(ROOT / "logs/stage05_validation.json", result)
    s["independent_validation"] = {
        "path": str(ROOT / "logs/stage05_validation.json"),
        "status": result["status"],
        "scope": result["scope"],
    }
    write_json(ROOT / "stage_05.json", s)
    assert not errors, errors
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
