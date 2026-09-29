# Reproducibility

## Without patient data

The synthetic tests cover input geometry and scaling, transform direction, support masks, subtraction arithmetic and review-state handling. All 23 pass with the recorded environment.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt      # Windows: .venv\Scripts\python.exe
.venv/bin/python -m unittest discover -s scripts -p "test_*.py" -v
```

`results/cohort_summary.csv` holds the full-precision cohort statistics used in the README, RESULTS.md and the report.

## Environment

The analysis used Python 3.12.7, NumPy 1.26.4, SimpleITK 2.5.6 and Matplotlib 3.9.2. DICOM evidence was read in a separate Python 3.12.2 environment with pydicom 3.0.2. SciPy is used for connected components and registration helpers, and nibabel for segmentation finalization. TotalSegmentator 2.18.0 was run separately for C1/T1 masks.

`requirements.txt` lists direct dependencies with version ranges; it is not a locked copy of the original environment.

## With the original study data

Reproducing the image processing requires the private source images, phase manifest, segmentation outputs and review records, which are not distributed. The scripts contain pilot-specific patient IDs, ROI seeds and acceptance rules and are not a general-purpose pipeline for new cohorts.

Copy `study.example.json` to `config/study.json` and set the source root, output root and interpreter paths. `CAROTID_STUDY_ROOT` and `CAROTID_SOURCE_ROOT` can be used instead to point to the study directory and its source NIfTI directory. The source layout is `<source_root>/<patient>/nifti/{N,A,V}.nii.gz`.

| Stage | Script | Purpose |
|---|---|---|
| 1 | `stage01_dicom_evidence.py`, `stage01_audit.py` | Check phase selection, geometry and HU scaling against DICOM |
| 2 | `stage02_body_roi.py`, `finalize_stage02_totalseg.py` | Body masks; C1–T1 crop from TotalSegmentator labels |
| 3 | `stage03_registration.py`, `stage03_multilevel_qc.py` | Rigid N→A and V→A registration and QC figures |
| 4 | `stage04_reconstruction.py` | Five signed products |
| 5 | `stage05_assessment.py` | ROI measurements; reads the reviewer decisions in `config/stage05_reviewer_visual_qc.json` |

`validate_stage*.py` check saved stage outputs and update local validation records, so run them only in the study directory.
