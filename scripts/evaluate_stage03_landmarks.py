from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import numpy as np
import SimpleITK as sitk


from study_paths import ROOT


def main() -> int:
    summary = json.loads((ROOT / "stage_03.json").read_text(encoding="utf-8"))
    pairs = {
        (p["patient"], q["moving_phase"]): q
        for p in summary["patients"]
        for q in p["pairs"]
    }
    source = ROOT / "tables" / "stage03_landmarks_template.csv"
    rows = list(csv.DictReader(source.open(encoding="utf-8-sig")))
    results = []
    coordinate_fields = (
        "fixed_A_lps_x_mm",
        "fixed_A_lps_y_mm",
        "fixed_A_lps_z_mm",
        "moving_native_lps_x_mm",
        "moving_native_lps_y_mm",
        "moving_native_lps_z_mm",
    )
    for row in rows:
        if not all(row[field].strip() for field in coordinate_fields):
            continue
        pair = pairs[(row["patient"], row["moving_phase"])]
        fixed = tuple(float(row[field]) for field in coordinate_fields[:3])
        moving = tuple(float(row[field]) for field in coordinate_fields[3:])
        transform = sitk.ReadTransform(pair["transforms"]["final"]["path"])
        mapped = transform.GetInverse().TransformPoint(moving)
        error = float(np.linalg.norm(np.asarray(mapped) - fixed))
        trigger = float(pair["landmarks"]["pilot_review_trigger_mm"])
        results.append(
            {
                **row,
                "moving_mapped_to_A_lps_mm": list(map(float, mapped)),
                "discrepancy_mm": error,
                "pilot_review_trigger_mm": trigger,
                "triggered": error > trigger,
            }
        )
    output = ROOT / "tables" / "stage03_landmark_discrepancies.json"
    output.write_text(
        json.dumps(
            {
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                "completed_landmarks": len(results),
                "status": "MEASURED" if results else "NOT_PERFORMED_OPTIONAL",
                "required_for_stage03_acceptance": False,
                "interpretation": "One-native-voxel discrepancies are pilot review triggers, not validated wall-imaging tolerances.",
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
