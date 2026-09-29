# Results

The pilot included five patients, each with non-contrast (N), arterial (A), and venous (V) CT. All five fixed-formula products were generated for every patient. Qualitative cohort review accepted all 25 patient–product combinations for sampled image/signal feasibility. No numerical image-quality scores were assigned.

## Sampled signal

| Product | Lumen mean | Muscle mean | Muscle SD |
|---|---:|---:|---:|
| A−N | 349.4 (304.5 to 430.1) | 4.1 (−6.9 to 20.6) | 28.6 (13.4 to 35.5) |
| V−N | 120.1 (75.0 to 150.8) | 21.1 (2.0 to 48.6) | 24.3 (15.4 to 45.0) |
| V−A | −229.3 (−308.3 to −218.2) | 12.3 (−2.1 to 29.8) | 28.3 (13.7 to 32.8) |
| 2N−A | −301.7 (−395.6 to −259.8) | 30.0 (0.7 to 39.2) | 38.8 (20.9 to 51.3) |
| 2V−A | −93.4 (−151.9 to −19.0) | 81.4 (32.4 to 103.8) | 37.9 (20.7 to 65.9) |

Values are HU-derived cohort medians with patient ranges. Each lumen value is the mean of the available left and right side means within one patient. Muscle and bone each have one fixed sample per patient. The patient is the unit of analysis; repeated sides are not independent observations. Muscle SD uses `ddof=1` and includes noise, interpolation correlation, and local tissue heterogeneity.

All 20 sampled tissue spheres had complete common and formula-specific valid support. Common valid neck/body support ranged from 92.33% to 98.83%. This denominator is the arterial body within the study crop; it is not a measure of full carotid or bifurcation coverage.

## Interpretation

A−N gave positive lumen enhancement in all five patients, and V−A gave negative lumen values in all five. For both 2N−A and 2V−A, sampled lumen means were below the same-product muscle signal on both sides in every patient.

The two composites measure different things. 2N−A is the non-contrast image minus the arterial enhancement (N − (A−N)), giving a dark lumen. 2V−A is the venous image plus the A→V change (V + (V−A)), so its lumen value reflects contrast washout between the arterial and venous phases. 2V−A lumen means ranged from −151.9 to −19.0 across patients, following the measured A and V attenuation.

Residual bone, airway and perivascular edges remained visible. Their causes were not isolated and may include motion, local mismatch, partial volume and acquisition differences. Median patient-level mean absolute bone residuals were 86.0 for A−N, 129.0 for V−N and 58.2 for V−A.

## Limitations

- The cohort contains five patients and provides technical pilot evidence only.
- Diagnostic performance remain unmeasured.

The next step is to measure local alignment and wall/plaque boundary visibility on expert-localized anatomy.
