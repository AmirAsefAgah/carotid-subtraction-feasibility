from __future__ import annotations

import csv


from study_paths import ROOT

LANDMARKS = (
    ("left_bifurcation", "carotid_bifurcation"),
    ("right_bifurcation", "carotid_bifurcation"),
    ("left_calcium", "calcium_if_corresponding_focus_visible"),
    ("right_calcium", "calcium_if_corresponding_focus_visible"),
    ("soft_tissue_1", "visible_soft_tissue_landmark"),
    ("soft_tissue_2", "visible_soft_tissue_landmark"),
)


def main() -> int:
    output = ROOT / "tables" / "stage03_landmarks_template.csv"
    fields = [
        "patient",
        "moving_phase",
        "landmark",
        "kind",
        "fixed_A_lps_x_mm",
        "fixed_A_lps_y_mm",
        "fixed_A_lps_z_mm",
        "moving_native_lps_x_mm",
        "moving_native_lps_y_mm",
        "moving_native_lps_z_mm",
        "reviewer",
        "notes",
    ]
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for patient in ("P001", "P002", "P003", "P004", "P005"):
            for phase in ("N", "V"):
                for landmark, kind in LANDMARKS:
                    writer.writerow(
                        {
                            "patient": patient,
                            "moving_phase": phase,
                            "landmark": landmark,
                            "kind": kind,
                        }
                    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
