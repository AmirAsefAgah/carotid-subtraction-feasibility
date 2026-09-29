# Carotid subtraction feasibility study

Code and aggregate results from a five-patient pilot that computes five subtraction formulas from registered non-contrast (N), arterial (A) and venous (V) neck CT.

**[Interactive report](https://amirasefagah.github.io/carotid-subtraction-feasibility/)**: figures for each patient with synchronized source images, registration checks, products and sampled values.

## Results

All five products were generated for every patient, and all 25 patient–product combinations were accepted for sampled signal. Values are HU-derived: median of the five patient-level means (range).

| Product | Formula | Lumen mean | Muscle mean |
|---|---|---:|---:|
| Arterial enhancement | A−N | 349.4 (304.5 to 430.1) | 4.1 (−6.9 to 20.6) |
| Venous enhancement | V−N | 120.1 (75.0 to 150.8) | 21.1 (2.0 to 48.6) |
| Temporal change | V−A | −229.3 (−308.3 to −218.2) | 12.3 (−2.1 to 29.8) |
| Dark lumen | 2N−A | −301.7 (−395.6 to −259.8) | 30.0 (0.7 to 39.2) |
| Venous washout | 2V−A | −93.4 (−151.9 to −19.0) | 81.4 (32.4 to 103.8) |

Carotid wall-imaging feasibility has not been assessed. See [RESULTS.md](RESULTS.md) for interpretation and limitations, [METHODS.md](METHODS.md) for the processing method, and [results/cohort_summary.csv](results/cohort_summary.csv) for full-precision values.

## Contents

| Path | Description |
|---|---|
| `scripts/` | Input audit, body masks, registration, reconstruction, ROI assessment, validation and tests |
| `results/cohort_summary.csv` | Cohort-level aggregate measurements |
| `docs/` | Interactive report (GitHub Pages) |
| `study.example.json` | Example configuration |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Environment, tests and running on private data |

Source scans, per-patient measurement tables, DICOM inventories and review records are not distributed. The report images are JPEG renderings labelled with study codes only.

## Tests

The tests use synthetic images and need no patient data (Python 3.12):

```bash
pip install -r requirements.txt
python -m unittest discover -s scripts -p "test_*.py" -v
```

## License

[MIT](LICENSE)
