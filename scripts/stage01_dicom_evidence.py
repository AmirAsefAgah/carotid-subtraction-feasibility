"""Read source DICOM evidence for the candidate phase manifest."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pydicom

TAGS = [
    "SeriesInstanceUID",
    "StudyInstanceUID",
    "SOPInstanceUID",
    "SeriesNumber",
    "SeriesDescription",
    "ProtocolName",
    "Modality",
    "Rows",
    "Columns",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "RescaleSlope",
    "RescaleIntercept",
    "RescaleType",
    "AcquisitionDate",
    "AcquisitionTime",
    "AcquisitionDateTime",
    "ContrastBolusStartTime",
    "ContrastBolusStopTime",
    "ContrastBolusAgent",
    "KVP",
    "ConvolutionKernel",
    "FrameOfReferenceUID",
    "GantryDetectorTilt",
    "PixelRepresentation",
    "BitsStored",
    "PixelPaddingValue",
    "ImageType",
]


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def json_value(value):
    if isinstance(value, (list, tuple, pydicom.multival.MultiValue)):
        return [json_value(v) for v in value]
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(cfg["output_root"])
    with (out / "tables/verified_phase_manifest.csv").open(encoding="utf-8-sig") as f:
        prior = list(csv.DictReader(f))
    records = defaultdict(list)
    errors = []
    for folder in sorted({r["dicom_folder"] for r in prior}):
        print("Read DICOM metadata:", folder, flush=True)
        for path in sorted(Path(folder).rglob("*")):
            if not path.is_file():
                continue
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=TAGS)
                uid = str(ds.SeriesInstanceUID)
                rec = {tag: json_value(getattr(ds, tag, "")) for tag in TAGS}
                rec["path"] = str(path)
                records[uid].append(rec)
            except Exception as exc:
                errors.append({"path": str(path), "error": str(exc)})
    dest = out / "tables/stage01_dicom_evidence"
    dest.mkdir(parents=True, exist_ok=True)
    results = []
    for old in prior:
        patient, phase, uid = old["patient"], old["phase"], old["series_uid"]
        result = {
            "patient": patient,
            "phase": phase,
            "candidate_series_uid": uid,
            "prior_manifest_row": old,
            "errors": [],
        }
        try:
            instances = records[uid]
            if not instances:
                raise ValueError("Candidate series missing from original DICOM")
            orient = np.asarray(instances[0]["ImageOrientationPatient"], float)
            normal = np.cross(orient[:3], orient[3:])
            instances.sort(
                key=lambda r: np.dot(
                    np.asarray(r["ImagePositionPatient"], float), normal
                )
            )
            values = {
                tag: sorted({json.dumps(r[tag]) for r in instances})
                for tag in TAGS
                if tag
                not in [
                    "SOPInstanceUID",
                    "ImagePositionPatient",
                    "AcquisitionTime",
                    "path",
                ]
            }
            values = {
                tag: [json.loads(v) for v in entries] for tag, entries in values.items()
            }
            first_time = min(r["AcquisitionTime"] for r in instances)
            siblings = []
            for other_uid, other in records.items():
                r = other[0]
                if (
                    r["StudyInstanceUID"] == instances[0]["StudyInstanceUID"]
                    and other_uid != uid
                    and r["SeriesDescription"] != "THIN"
                    and min(x["AcquisitionTime"] for x in other) == first_time
                ):
                    siblings.append(
                        {
                            k: r[k]
                            for k in [
                                "SeriesInstanceUID",
                                "SeriesNumber",
                                "SeriesDescription",
                            ]
                        }
                    )
            selected = sorted(
                {
                    round(f * (len(instances) - 1))
                    for f in cfg["audit"]["dicom_sample_slice_fractions"]
                }
            )
            samples = []
            for index in selected:
                r = instances[index]
                path = Path(r["path"])
                ds = pydicom.dcmread(path)
                step = cfg["audit"]["dicom_sample_pixel_step"]
                rr, cc = np.meshgrid(
                    np.arange(0, ds.Rows, step),
                    np.arange(0, ds.Columns, step),
                    indexing="ij",
                )
                rr, cc = rr.ravel(), cc.ravel()
                pixels = ds.pixel_array[rr, cc].astype(float)
                samples.append(
                    {
                        "slice_index": index,
                        "path": str(path),
                        "sha256": digest(path),
                        "rows": rr.tolist(),
                        "columns": cc.tolist(),
                        "stored_values": pixels.tolist(),
                        "hu_values": (
                            pixels * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
                        ).tolist(),
                    }
                )
            result.update(
                instances=instances,
                values=values,
                same_acquisition_named_series=siblings,
                acquisition_start=first_time,
                acquisition_end=max(r["AcquisitionTime"] for r in instances),
                samples=samples,
            )
        except Exception as exc:
            result["errors"].append(str(exc))
        path = dest / f"{patient}_{phase}.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(
            {
                "patient": patient,
                "phase": phase,
                "path": str(path),
                "sha256": digest(path),
                "errors": result["errors"],
            }
        )
        print(patient, phase, "evidence saved", flush=True)
    # Include unselected series so additional acquisitions cannot be confused with V.
    selected_uids = {r["series_uid"] for r in prior}
    extra = [
        {
            k: items[0][k]
            for k in [
                "SeriesInstanceUID",
                "SeriesNumber",
                "SeriesDescription",
                "AcquisitionDate",
            ]
        }
        | {
            "folder": str(Path(items[0]["path"]).parent),
            "acquisition_start": min(r["AcquisitionTime"] for r in items),
        }
        for uid, items in records.items()
        if uid not in selected_uids
    ]
    (dest / "index.json").write_text(
        json.dumps(
            {
                "results": results,
                "read_errors": errors,
                "unselected_series": extra,
                "python": sys.version,
                "executable": sys.executable,
                "pydicom": pydicom.__version__,
                "numpy": np.__version__,
                "metadata_files_read": sum(map(len, records.values())),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
