"""Configurable weighted scoring with missing-signal renormalization."""

DEFAULT_WEIGHTS = {"audio": 0.30, "chat": 0.25, "speech": 0.20, "visual": 0.25}


def score_highlight(signals: dict, weights: dict | None = None, thresholds: dict | None = None) -> dict:
    """Score present numeric signals on 0–100 and report exactly what contributed."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    present = {name: max(0.0, min(100.0, float(value))) for name, value in signals.items()
               if name in weights and value is not None}
    total_weight = sum(weights[name] for name in present)
    score = sum(value * weights[name] for name, value in present.items()) / total_weight if total_weight else 0.0
    thresholds = {"high": 85, "potential": 70, "review": 50, **(thresholds or {})}
    if score >= thresholds["high"]:
        label = "high-priority highlight"
    elif score >= thresholds["potential"]:
        label = "potential highlight"
    elif score >= thresholds["review"]:
        label = "review later"
    else:
        label = "low-confidence candidate"
    return {"score": round(score, 1), "label": label,
            "available_signals": sorted(present),
            "normalized_weights": {name: round(weights[name] / total_weight, 3) for name in present} if total_weight else {},
            "contributions": {name: round(value * weights[name] / total_weight, 1) for name, value in present.items()} if total_weight else {}}
