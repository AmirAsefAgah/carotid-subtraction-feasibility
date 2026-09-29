"""Reviewer-led registration acceptance, independent of optional manual ROIs."""

import hashlib
import json

POLICY = {
    "authority": "Reviewer visual QC of registered A/N/V images",
    "manual_coordinates_required": False,
    "manual_segmentation_required": False,
    "plaque_identification_required": False,
    "manual_landmark_validation": "not performed; no quantitative landmark-registration accuracy claim",
    "automated_qc_role": "supplementary; centroid and candidate-edge scores are not manual accuracy measurements",
    "gross_failure_override": "Known corruption, wrong series/phase, inadequate coverage, major misregistration or severe artifact blocks acceptance",
    "scope": "registration acceptance only; product-specific Stage 5 feasibility is separate",
}


def artifacts(node):
    if isinstance(node, dict):
        if "path" in node and "sha256" in node:
            yield node
        else:
            for value in node.values():
                yield from artifacts(value)


def evidence_id(pair):
    """Bind review to recorded source, transform and quantitative output hashes."""
    evidence = {
        "inputs": {
            k: v for k, v in pair.get("inputs", {}).items() if k.endswith("sha256")
        },
        "transform": pair.get("transforms", {}).get("final", {}).get("sha256"),
        "outputs": {v["path"]: v["sha256"] for v in artifacts(pair.get("outputs", {}))},
    }
    return hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()


def disposition(pair, review=None):
    if pair.get("error") or not pair.get("outputs"):
        return "fail", "Missing/corrupted registration output or pipeline exception."
    if pair.get("gross_technical_failures"):
        return "fail", "Documented gross technical failure: " + str(
            pair["gross_technical_failures"]
        )
    if pair.get("support", {}).get("pairwise_support_fraction", 0) < 0.80:
        return (
            "fail",
            "Pairwise support below retained 80% gross coverage safety screen.",
        )
    if not pair.get("transform_convention", {}).get("roundtrip_pass", False):
        return "fail", "Transform roundtrip integrity check failed."
    if (
        review
        and review.get("decision") == "rejected"
        and review.get("evidence_id") == evidence_id(pair)
    ):
        return "fail", "Reviewer rejected registration."
    if (
        review
        and review.get("decision") == "accepted"
        and review.get("evidence_id") == evidence_id(pair)
    ):
        return (
            "pass",
            "Reviewer visually accepted registered A/N/V alignment; manual landmarks and segmentation are optional.",
        )
    return (
        "review",
        "Reviewer visual QC is absent or applies to different artifacts; manual landmarks are optional.",
    )


def apply_reviews(summary, reviews):
    lookup = {(r["patient"], r["moving_phase"]): r for r in reviews.get("pairs", [])}
    for patient in summary["patients"]:
        for pair in patient["pairs"]:
            review = lookup.get((patient["patient"], pair["moving_phase"]))
            pair["state"], pair["state_basis"] = disposition(pair, review)
            valid_review = review and review.get("evidence_id") == evidence_id(pair)
            pair["reviewer_visual_qc"] = (
                review["decision"] if valid_review else "pending"
            )
            pair["reviewer_visual_qc_provenance"] = review if valid_review else None
            pair["manual_landmark_validation"] = "not performed"
            pair["quantitative_landmark_registration_accuracy_mm"] = None
            pair["automated_qc_role"] = POLICY["automated_qc_role"]
        states = [p["state"] for p in patient["pairs"]]
        patient["state"] = (
            "fail"
            if "fail" in states
            else "partial"
            if len(states) != 2
            else "pass"
            if all(s == "pass" for s in states)
            else "review"
        )
        patient["reviewer_visual_qc"] = (
            "accepted"
            if all(p["reviewer_visual_qc"] == "accepted" for p in patient["pairs"])
            else "pending"
        )
        patient["human_review_status"] = (
            "COMPLETED_VISUAL_ACCEPTANCE"
            if patient["reviewer_visual_qc"] == "accepted"
            else "PENDING_VISUAL_REVIEW"
        )
    states = [p["state"] for p in summary["patients"]]
    summary["status"] = (
        "COMPLETE_WITH_FAILURES"
        if "fail" in states
        else "PARTIAL"
        if len(states) != 5 or "partial" in states
        else "COMPLETE"
        if all(s == "pass" for s in states)
        else "COMPLETE_PENDING_VISUAL_REVIEW"
    )
    summary["acceptance_policy"] = POLICY
    summary["completion_interpretation"] = (
        "Registration acceptance follows recorded reviewer visual QC and gross technical safeguards. No manual landmark accuracy or wall-imaging feasibility is implied."
    )
