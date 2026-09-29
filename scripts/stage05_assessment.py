"""Measure fixed source-anatomy ROIs for the original five-patient study."""

from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import SimpleITK as sitk
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from stage04_reconstruction import FORMULAS, DIFFERENCES, same_geometry

from study_paths import ROOT

# Selected on source A only, before examining product values. No output-specific recentering.
SEEDS = {
    "P001": {
        "lumen_R": [205, 218, 195],
        "lumen_L": [300, 218, 195],
        "muscle": [210, 288, 195],
        "bone": [218, 260, 195],
    },
    "P002": {
        "lumen_R": [242, 232, 195],
        "lumen_L": [318, 242, 195],
        "muscle": [213, 310, 195],
        "bone": [230, 281, 195],
    },
    "P003": {
        "lumen_R": [186, 215, 193],
        "lumen_L": [282, 222, 193],
        "muscle": [202, 278, 193],
        "bone": [202, 254, 193],
    },
    "P004": {
        "lumen_R": [220, 289, 211],
        "lumen_L": [349, 286, 211],
        "muscle": [245, 349, 211],
        "bone": [254, 320, 211],
    },
    "P005": {
        "lumen_R": [217, 214, 196],
        "lumen_L": [306, 217, 196],
        "muscle": [206, 281, 196],
        "bone": [215, 255, 196],
    },
}
RADII = {"lumen_R": 1.25, "lumen_L": 1.25, "muscle": 3.0, "bone": 1.5}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def stats(values):
    v = np.asarray(values, dtype=np.float64)
    if not len(v):
        return {"n": 0, "mean": None, "median": None, "sd": None, "mean_abs": None}
    if not np.isfinite(v).all():
        raise ValueError("Nonfinite valid ROI samples")
    return {
        "n": len(v),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
        "mean_abs": float(np.abs(v).mean()),
    }


def sphere_indices(image, center_lps, radius):
    """Voxel centres inside a physical sphere; reject a sphere crossing the image grid."""
    center = np.array(image.TransformPhysicalPointToContinuousIndex(center_lps))
    ext = radius / np.array(image.GetSpacing())
    lo = np.floor(center - ext).astype(int)
    hi = np.ceil(center + ext).astype(int)
    if np.any(lo < 0) or np.any(hi >= np.array(image.GetSize())):
        raise ValueError("ROI extends outside grid")
    zz, yy, xx = np.meshgrid(
        np.arange(lo[2], hi[2] + 1),
        np.arange(lo[1], hi[1] + 1),
        np.arange(lo[0], hi[0] + 1),
        indexing="ij",
    )
    xyz = np.stack([xx, yy, zz], axis=-1)
    physical = (
        np.einsum(
            "ij,...j->...i",
            np.array(image.GetDirection()).reshape(3, 3),
            xyz * np.array(image.GetSpacing()),
        )
        + image.GetOrigin()
    )
    inside = np.linalg.norm(physical - center_lps, axis=-1) <= radius + 1e-8
    return tuple(x[inside] for x in (zz, yy, xx))


def metric_row(pid, side, name, samples, coverage, reviewer):
    lu = samples[f"lumen_{side}"]
    mu = samples["muscle"]
    bo = samples["bone"]
    return {
        "patient": pid,
        "side": side,
        "product": name,
        "formula": FORMULAS[name]["formula"],
        "roi_provenance": "AI-assisted source-anatomy placement; exact bifurcation not reviewer-localized",
        "target_common_coverage_fraction": coverage[f"lumen_{side}"]["common"],
        "target_product_coverage_fraction": coverage[f"lumen_{side}"][
            FORMULAS[name]["mask"]
        ],
        "coverage_scope": "sampled lumen sphere only; full bifurcation/CCA extent unmeasured",
        "local_alignment_review": "accepted (cohort reviewer)",
        "local_alignment_error_mm": None,
        "lumen_n": lu["n"],
        "lumen_mean_hu": lu["mean"],
        "lumen_median_hu": lu["median"],
        "muscle_n": mu["n"],
        "muscle_mean_hu": mu["mean"],
        "muscle_sd_hu": mu["sd"],
        "bone_mean_hu": bo["mean"],
        "bone_mean_absolute_residual_hu": bo["mean_abs"]
        if name in DIFFERENCES
        else None,
        "bone_cancellation_required": name in DIFFERENCES,
        "wall_mean_hu": None,
        "wall_minus_lumen_hu": None,
        "wall_cnr": None,
        "wall_visibility": "unmeasured: no reliably localized reviewer wall/plaque ROI",
        "overall_visual_review": reviewer["decision"],
        "quality_score": None,
        "arterial_enhancement_score": None,
        "lumen_suppression_score": None,
        "wall_boundary_score": None,
        "motion_residual_review": "no dominant artifact reported by reviewer; type/severity not separately scored",
        "calcification_residual_review": "unmeasured: no localized calcium reference; not inferred from bone ROI",
        "lumen_minus_muscle_hu": lu["mean"] - mu["mean"]
        if lu["n"] and mu["n"]
        else None,
    }


def load_review(path):
    """Require the original pilot's recorded qualitative acceptance."""
    review = json.loads(Path(path).read_text(encoding="utf-8"))
    if review.get("decision") != "acceptable qualitative visual review":
        raise ValueError(
            "This pilot assessment requires recorded qualitative acceptance"
        )
    return review


def main():
    stamp = datetime.now(timezone.utc).isoformat()
    review = load_review(ROOT / "config/stage05_reviewer_visual_qc.json")
    stage4 = json.loads((ROOT / "stage_04.json").read_text())
    roipath = ROOT / "config/stage05_rois.json"
    if not roipath.exists():
        config = {
            "schema_version": 1,
            "selection": "Source A only; fixed physical sphere reused across all registered phases and products. AI-assisted anatomical samples; not expert-localized.",
            "statistics": "common NAV support; sample SD ddof=1; invalid fill excluded; no per-product ROI changes",
            "patients": {},
        }
        for pid, seeds in SEEDS.items():
            a = sitk.ReadImage(
                str(
                    ROOT / "patients" / pid / "stage_03" / f"{pid}_A_native_crop.nii.gz"
                )
            )
            config["patients"][pid] = {
                k: {
                    "center_lps_mm": list(a.TransformIndexToPhysicalPoint(tuple(c))),
                    "source_crop_index_xyz": c,
                    "radius_mm": RADII[k],
                    "reviewer_localized": False,
                }
                for k, c in seeds.items()
            }
            config["patients"][pid].update(
                {"cca_R": None, "cca_L": None, "wall_R": None, "wall_L": None}
            )
        write_json(roipath, config)
    config = json.loads(roipath.read_text())
    allrows = []
    phase_rows = []
    patient_rows = []
    roi_rows = []
    checks = []
    figures = []
    for patient in stage4["patients"]:
        pid = patient["patient"]
        print(pid, flush=True)
        d = ROOT / "patients" / pid / "stage_03"
        out = ROOT / "patients" / pid / "stage_05"
        out.mkdir(exist_ok=True)
        a = sitk.ReadImage(str(d / f"{pid}_A_native_crop.nii.gz"))
        active = {k: v for k, v in config["patients"][pid].items() if v is not None}
        if any(k.startswith(("wall", "cca")) for k in active):
            raise ValueError(
                "Optional wall/CCA ROIs require separate explicit source mask support; do not treat them as spheres automatically"
            )
        indices = {
            k: sphere_indices(a, v["center_lps_mm"], v["radius_mm"])
            for k, v in active.items()
        }
        z = int(
            round(
                a.TransformPhysicalPointToContinuousIndex(
                    active["lumen_R"]["center_lps_mm"]
                )[2]
            )
        )
        coverage = {k: {} for k in active}
        keep = {}
        masks = {}
        planes = {}
        sampled = {}
        fingerprints = {}
        label = sitk.GetArrayFromImage(a) * 0
        label = label.astype(np.uint8)
        for i, (k, idx) in enumerate(indices.items(), 1):
            label[idx] = i
        lm = sitk.GetImageFromArray(label)
        lm.CopyInformation(a)
        sitk.WriteImage(lm, str(out / f"{pid}_assessment_roi_labels.nii.gz"), True)
        del label, lm
        for key, record in patient["masks"].items():
            im = sitk.ReadImage(record["path"])
            assert same_geometry(a, im)
            arr = sitk.GetArrayViewFromImage(im)
            for k, idx in indices.items():
                keep[(key, k)] = np.array(arr[idx] > 0)
                coverage[k][key] = float(keep[(key, k)].mean())
            if key == "NAV_common":
                masks["plane"] = np.array(arr[z] > 0)
                common_count = int(np.count_nonzero(arr))
            del arr, im
        for k in active:
            coverage[k]["common"] = coverage[k]["NAV_common"]
            roi_rows.append(
                {
                    "patient": pid,
                    "roi": k,
                    "center_lps_mm": json.dumps(active[k]["center_lps_mm"]),
                    "radius_mm": active[k]["radius_mm"],
                    "total_voxels": len(indices[k][0]),
                    **coverage[k],
                }
            )
        body = sitk.ReadImage(str(d / f"{pid}_A_body_crop.nii.gz"))
        den = int(np.count_nonzero(sitk.GetArrayViewFromImage(body)))
        del body
        paths = {
            ph: d
            / f"{pid}_{'A_native_crop' if ph == 'A' else ph + '_registered_to_A'}.nii.gz"
            for ph in ["A", "N", "V"]
        }
        paths.update(
            {
                k: Path(v["quantitative_image"]["path"])
                for k, v in patient["products"].items()
            }
        )
        for key, path in paths.items():
            im = sitk.ReadImage(str(path))
            assert same_geometry(a, im), (pid, key, "geometry")
            arr = sitk.GetArrayViewFromImage(im)
            planes[key] = np.array(arr[z])
            sampled[key] = {}
            for k, idx in indices.items():
                sampled[key][k] = np.array(
                    arr[idx][keep[("NAV_common", k)]], dtype=np.float64
                )
            del arr, im
            fingerprints[key] = sha(path)
            if key in patient["products"]:
                assert (
                    fingerprints[key]
                    == patient["products"][key]["quantitative_image"]["sha256"]
                ), (pid, key, "hash changed")
        source_stats = {
            key: {k: stats(v) for k, v in vals.items()} for key, vals in sampled.items()
        }
        for side in ["R", "L"]:
            k = f"lumen_{side}"
            n = source_stats["N"][k]["mean"]
            ar = source_stats["A"][k]["mean"]
            v = source_stats["V"][k]["mean"]
            phase_rows.append(
                {
                    "patient": pid,
                    "side": side,
                    "roi_status": "AI-assisted source-anatomy sample; bifurcation level unconfirmed",
                    "N_mean_hu": n,
                    "A_mean_hu": ar,
                    "V_mean_hu": v,
                    "A_minus_V_hu": ar - v,
                    "predicted_2N_minus_A_hu": 2 * n - ar,
                    "predicted_2V_minus_A_hu": 2 * v - ar,
                    "observed_2V_minus_A_hu": source_stats["dark_lumen_V"][k]["mean"],
                    "2V_minus_A_lumen_minus_muscle_hu": source_stats["dark_lumen_V"][k][
                        "mean"
                    ]
                    - source_stats["dark_lumen_V"]["muscle"]["mean"],
                }
            )
        max_error = 0.0
        for key in FORMULAS:
            errors = []
            for k in active:
                av, nv, vv = (sampled[p][k] for p in ["A", "N", "V"])
                exp = {
                    "arterial_enhancement": av - nv,
                    "venous_enhancement": vv - nv,
                    "temporal_difference": vv - av,
                    "dark_lumen_N": 2 * nv - av,
                    "dark_lumen_V": 2 * vv - av,
                }[key]
                errors.append(float(np.max(np.abs(sampled[key][k] - exp))))
            max_error = max(max_error, *errors)
            sides = [
                metric_row(pid, side, key, source_stats[key], coverage, review)
                for side in ["R", "L"]
            ]
            for row in sides:
                weak = (
                    key.startswith("dark_lumen") and row["lumen_minus_muscle_hu"] >= 0
                )
                row["status"] = "limited" if weak else "accepted"
                row["reason"] = (
                    "Lumen sample remains at/above muscle; dark-lumen suppression limited despite acceptable overall visual review."
                    if weak
                    else "Sampled targets covered; reviewer accepts alignment and visual quality; product-specific signal interpretable at sampled locations."
                )
                if row["target_common_coverage_fraction"] < 1:
                    row["status"] = "limited"
                    row["reason"] = (
                        "Sampled target has incomplete common support; review accepted overall visual quality."
                    )
            allrows.extend(sides)
            row = {
                "patient": pid,
                "product": key,
                "formula": FORMULAS[key]["formula"],
                "status": "limited"
                if any(r["status"] == "limited" for r in sides)
                else "accepted",
                "reason": " | ".join(dict.fromkeys(r["reason"] for r in sides)),
                "assessment_scope": "pilot image/signal feasibility at source-selected samples; wall feasibility not demonstrated",
                "common_neck_body_coverage_fraction": common_count / den,
                "target_common_coverage_fraction_min": min(
                    r["target_common_coverage_fraction"] for r in sides
                ),
                "lumen_mean_hu": float(np.mean([r["lumen_mean_hu"] for r in sides])),
                "lumen_median_hu": float(
                    np.median(
                        np.concatenate(
                            [sampled[key]["lumen_R"], sampled[key]["lumen_L"]]
                        )
                    )
                ),
                "lumen_patient_summary": "mean of side means; median pooled same-patient voxel samples",
                **{
                    field: sides[0][field]
                    for field in [
                        "muscle_mean_hu",
                        "muscle_sd_hu",
                        "bone_mean_hu",
                        "bone_mean_absolute_residual_hu",
                        "local_alignment_error_mm",
                        "local_alignment_review",
                        "wall_minus_lumen_hu",
                        "wall_cnr",
                        "wall_visibility",
                        "quality_score",
                        "motion_residual_review",
                        "calcification_residual_review",
                    ]
                },
                "right_status": sides[0]["status"],
                "left_status": sides[1]["status"],
            }
            patient_rows.append(row)
        assert max_error < 0.001, (pid, max_error)
        checks.append(
            {
                "patient": pid,
                "actual_product_hashes_match_stage04": True,
                "geometry_pass": True,
                "roi_formula_max_abs_error_hu": max_error,
                "common_neck_body_voxels": common_count,
                "reference_body_voxels": den,
                "common_neck_body_fraction": common_count / den,
                "target_coverage_scope": "sample spheres only; bifurcation extent and proximal CCA coverage not measured",
                "input_sha256": fingerprints,
            }
        )
        # Source map and paired output sheets; all products at identical physical locations/windows.
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(planes["A"], cmap="gray", vmin=-225, vmax=425)
        for k, v in active.items():
            x, y, _ = a.TransformPhysicalPointToContinuousIndex(v["center_lps_mm"])
            ax.add_patch(
                Circle(
                    (x, y),
                    v["radius_mm"] / a.GetSpacing()[0],
                    fill=False,
                    color="yellow",
                )
            )
            ax.text(x + 5, y, k, color="yellow", fontsize=9)
        ax.set_xlim(120, 410)
        ax.set_ylim(410, 140)
        ax.set_title(
            f"{pid} source A; z={z}; image left=patient R\nFixed source-selected spheres; exact bifurcation level unconfirmed"
        )
        fig.savefig(ROOT / "figures" / f"{pid}_stage05_roi_map.png", dpi=140)
        plt.close(fig)
        fig, axes = plt.subplots(2, 8, figsize=(24, 7), constrained_layout=True)
        for row, side in enumerate(["R", "L"]):
            x, y, _ = a.TransformPhysicalPointToContinuousIndex(
                active[f"lumen_{side}"]["center_lps_mm"]
            )
            x = int(round(x))
            y = int(round(y))
            r = 35
            for col, key in enumerate(paths):
                ax = axes[row, col]
                data = np.ma.array(planes[key], mask=~masks["plane"])
                is_diff = key in DIFFERENCES
                lo, hi = (
                    (-300, 300)
                    if is_diff
                    else ((-295, 355) if key.startswith("dark") else (-225, 425))
                )
                cmap = matplotlib.colormaps["RdBu_r" if is_diff else "gray"].copy()
                cmap.set_bad("black")
                ax.imshow(
                    data[y - r : y + r + 1, x - r : x + r + 1],
                    cmap=cmap,
                    vmin=lo,
                    vmax=hi,
                    interpolation="nearest",
                )
                ax.add_patch(
                    Circle(
                        (r, r),
                        1.25 / a.GetSpacing()[0],
                        fill=False,
                        color="yellow",
                        lw=0.6,
                    )
                )
                ax.set_title(
                    f"{side} {key}\nlumen {source_stats[key][f'lumen_{side}']['mean']:.1f}",
                    fontsize=9,
                )
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle(
            f"{pid}: same source-defined location across N/A/V and five outputs | lumen circle; no wall ROI\nSource [-225,425], differences [-300,300], composites [-295,355]; black outside validity; HU-derived values"
        )
        fp = ROOT / "figures" / f"{pid}_stage05_matched_products.png"
        fig.savefig(fp, dpi=130)
        plt.close(fig)
        figures.append(str(fp))
        write_json(
            out / f"{pid}_assessment.json",
            {
                "patient": pid,
                "rois": active,
                "coverage": coverage,
                "roi_statistics": source_stats,
                "input_sha256": fingerprints,
                "review": review,
            },
        )
    csv_write(ROOT / "tables/stage05_roi_coverage.csv", roi_rows)
    csv_write(ROOT / "tables/stage05_phase_attenuation.csv", phase_rows)
    csv_write(ROOT / "tables/stage05_carotid_product.csv", allrows)
    csv_write(ROOT / "tables/stage05_patient_product.csv", patient_rows)
    summaries = {}
    for key in FORMULAS:
        rows = [r for r in patient_rows if r["product"] == key]
        summaries[key] = {
            "patient_denominator": 5,
            "status_counts": {
                s: sum(r["status"] == s for r in rows)
                for s in ["accepted", "limited", "failed", "pending-review"]
            },
            "descriptive": {},
        }
        for field in [
            "lumen_mean_hu",
            "lumen_median_hu",
            "muscle_mean_hu",
            "muscle_sd_hu",
            "bone_mean_hu",
            "bone_mean_absolute_residual_hu",
        ]:
            values = [r[field] for r in rows if r[field] is not None]
            summaries[key]["descriptive"][field] = {
                "n_patients": len(values),
                "min": min(values) if values else None,
                "median": float(np.median(values)) if values else None,
                "max": max(values) if values else None,
            }
    result = {
        "stage": 5,
        "status": "COMPLETE_FOCUSED_ASSESSMENT_WITH_UNMEASURED_OPTIONAL_ITEMS",
        "generated_at_utc": stamp,
        "image_generation_complete_patients": 5,
        "image_generation_complete_products": 25,
        "wall_imaging_feasibility": "not demonstrated; wall localization, wall contrast/CNR and boundary visibility unmeasured",
        "reviewer": review,
        "acceptance_definition": "Adequate sampled target support, reviewer-accepted alignment, interpretable product-specific signal without dominant subtraction artifact. No requirement to identify plaque. Full carotid extent is outside sampled coverage claim.",
        "status_rule": "Patient limited if either side limited. Dark-lumen sample at/above same-output muscle is a descriptive suppression limitation, not a validated threshold. Other sampled signal plus qualitative review supports accepted pilot status; no diagnostic validation.",
        "rating_rubric": {
            "1": "unusable",
            "2": "major artifact with limited assessability",
            "3": "usable with minor artifact",
            "4": "good quality",
        },
        "rating_dimensions": {
            "ordinary": "arterial enhancement/temporal signal and artifact",
            "dark_lumen": "lumen suppression and wall/plaque boundary visibility separately",
        },
        "numeric_ratings": "unmeasured; qualitative review only",
        "optional_unmeasured": [
            "quantitative local landmark error",
            "reviewer-localized bifurcation extent",
            "short proximal CCA segment",
            "wall/plaque ROI, wall-minus-lumen and same-output CNR",
            "localized calcification residual severity",
            "numeric rubric ratings",
        ],
        "wall_cnr_definition": "abs(mean_wall - mean_lumen) / sample_SD_muscle in same product; undefined if SD zero; not calculated without reliably localized reviewer wall ROI",
        "roi_config": {"path": str(roipath), "sha256": sha(roipath)},
        "upstream_stage04_sha256": sha(ROOT / "stage_04.json"),
        "parameters": {
            "lumen_sphere_radius_mm": 1.25,
            "muscle_sphere_radius_mm": 3,
            "bone_sphere_radius_mm": 1.5,
            "sample_sd_ddof": 1,
            "primary_support": "NAV_common",
            "roi_selection": "source A only; physical locations fixed across outputs",
            "resampling_in_stage5": False,
        },
        "parameter_changes": [
            {
                "change": "New fixed source-anatomy ROI centres replace unconfirmed Stage 4 intensity close-up markers",
                "reason": "Some old markers fell on non-carotid anatomy; source-only selection before product measurement; all coordinates preserved in config",
            },
            {
                "change": "No changes to acquisition, registration, reconstruction formulas, weights, quantitative intensities or display windows",
                "reason": "Focused assessment of actual existing products",
            },
        ],
        "validation": checks,
        "patient_product_summary": patient_rows,
        "cohort_summary": summaries,
        "figures": figures,
        "limitations": [
            "Source-selected samples are not expert bifurcation markup; exact segment identity and extent remain unconfirmed.",
            "Muscle and bone spheres are AI-assisted source-anatomy samples, not independently validated tissue labels.",
            "Mean absolute bone residual is mean(abs(voxel difference)); composite bone retention is expected.",
            "No positive difference is labelled plaque enhancement; dark lumen does not establish wall visibility.",
            "Five patients are the independent observations; diagnostic performance was not evaluated.",
        ],
    }
    write_json(ROOT / "stage_05.json", result)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
