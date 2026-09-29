# Methods

## Inputs and geometry

Each patient contributes calibrated non-contrast (N), arterial (A), and venous (V) CT volumes. The arterial acquisition defines the native reference grid. Input checks compare NIfTI geometry and HU scaling with source DICOM evidence. Sources remain unchanged; intensities are not normalized and DICOM rescale factors are not reapplied to calibrated NIfTI values.

NIfTI coordinates use RAS; DICOM and SimpleITK use LPS. Conversion negates the physical X and Y axes. Geometry checks include spacing, origin, direction, and image size.

## Masks and registration

Body envelopes use a −450 HU threshold, central connected-component selection, and hole filling on native slices. TotalSegmentator C1/T1 masks define the neck region with 15 mm margins. Moving images retain their through-plane support.

N and V are independently registered to A with six-degree-of-freedom rigid transforms. Optimization uses SimpleITK, a fixed seed of `20260913`, one thread, a 1 mm optimization grid, and intensities clipped to [−200, 800] HU for optimization only. The multiresolution schedule uses shrink factors (4, 2, 1) and smoothing sigmas (2, 1, 0) mm, with 160 iterations and 0.012 metric sampling. A local rigid rescue uses 90 iterations and 0.025 sampling.

Candidate selection includes C1/T1 centroid checks. A metric improvement is rejected if it worsens their alignment by more than one native arterial voxel. These centroids are a technical check, not carotid-wall landmarks. No deformable transform is used.

Saved transforms map fixed/output physical points to moving/input physical points. Original moving HU values are sampled once onto the final cropped arterial grid. Valid-support masks distinguish measured data from exterior fill.

## Fixed formulas

| Product | Formula | Quantitative support |
|---|---|---|
| Arterial enhancement | A−N | A/N pairwise |
| Venous enhancement | V−N | N/A/V common |
| Temporal change | V−A | A/V pairwise |
| Dark lumen | 2N−A | A/N pairwise |
| Venous washout | 2V−A | A/V pairwise |

Quantitative products retain signed float32 values. Exterior zero is a masked compatibility fill, not measured tissue. Matched comparisons use common N/A/V neck/body support. No global high-HU deletion or bone suppression is applied.

## Sampling and summaries

Four spheres per patient were selected on source A: left and right lumen candidates (radius 1.25 mm), adjacent paraspinal muscle (3 mm), and bone (1.5 mm). Their physical LPS coordinates remain fixed across phases and products.

Statistics exclude invalid support. Patient-level lumen means average the side means; cohort tables summarize the five patient-level values. Muscle and bone each contribute one sample per patient. SD uses `ddof=1`. For ordinary differences, bone cancellation is summarized by the mean absolute voxel residual.

Display windows are fixed across the cohort: source width/level 650/100, 2N−A and 2V−A width/level 650/30, and a signed difference range of −300 to 300 HU-derived. Display settings do not clip quantitative files.

## Review and validation

Source and artifact hashes bind review records to the corresponding data. Synthetic checks cover scaling, orientation, transform conventions, masks, formula signs, and review invalidation. The original study also recorded saved-volume and ROI numerical validation.
