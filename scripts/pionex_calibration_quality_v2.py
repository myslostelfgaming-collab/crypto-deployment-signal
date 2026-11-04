#!/usr/bin/env python3
"""Audit-friendly calibration quality helpers for Pionex Phase 4.5/4.6.

This module is intentionally pure-Python and side-effect free.  It does not call
Pionex, place orders, or mutate live state.  It upgrades calibration evidence
without changing the existing Phase 4.5 risk/execution gates.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import math
import statistics
from typing import Any, Iterable, Optional

EPS = 1e-12
DEFAULT_MIN_COHORT_N = 8
DEFAULT_RECENCY_HALF_LIFE_DAYS = 30.0
DEFAULT_RECENCY_FLOOR = 0.25
ACTIVITY_BINS = (
    ("ZERO", 0.0, 0.0),
    ("1_TO_5", 0.0, 5.0),
    ("5_TO_15", 5.0, 15.0),
    ("15_TO_25", 15.0, 25.0),
    ("25_PLUS", 25.0, float("inf")),
)


def _f(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value in (None, ""):
            return default
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _percentile(values: Iterable[float], p: float) -> Optional[float]:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    p = min(1.0, max(0.0, p))
    pos = (len(vals) - 1) * p
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _weighted_mean(values: list[float], weights: list[float]) -> Optional[float]:
    if not values or len(values) != len(weights):
        return None
    total = sum(max(0.0, float(w)) for w in weights)
    if total <= EPS:
        return None
    return sum(float(v) * max(0.0, float(w)) for v, w in zip(values, weights)) / total


def _weighted_median(values: list[float], weights: list[float]) -> Optional[float]:
    if not values or len(values) != len(weights):
        return None
    pairs = sorted((float(v), max(0.0, float(w))) for v, w in zip(values, weights))
    total = sum(w for _, w in pairs)
    if total <= EPS:
        return statistics.median(float(v) for v in values)
    target = total / 2.0
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= target:
            return value
    return pairs[-1][0]


def _round(value: Optional[float], digits: int = 6) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def error_metrics(
    actual: list[float],
    predicted: list[float],
    *,
    weights: Optional[list[float]] = None,
    mape_mask: Optional[list[bool]] = None,
) -> dict[str, Any]:
    """Return absolute, weighted and percentage diagnostics.

    Bias is prediction - actual, so positive bias means systematic overprediction.
    MAPE is explicitly secondary; WAPE is the primary percentage-type metric.
    """
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted lengths differ")
    n = len(actual)
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n:
        raise ValueError("weights length differs")
    if mape_mask is None:
        mape_mask = [abs(float(a)) > EPS for a in actual]
    if len(mape_mask) != n:
        raise ValueError("mape_mask length differs")

    if n == 0:
        return {
            "n": 0,
            "mae": None,
            "median_absolute_error": None,
            "wape_pct": None,
            "mape_pct": None,
            "mape_n": 0,
            "bias": None,
            "rmse": None,
            "weighted_mae": None,
            "weighted_wape_pct": None,
            "weighted_bias": None,
            "secondary_percentage_metric": "MAPE_GUARDED",
            "primary_percentage_metric": "WAPE",
        }

    a = [float(x) for x in actual]
    p = [float(x) for x in predicted]
    errors = [pp - aa for aa, pp in zip(a, p)]
    abs_errors = [abs(e) for e in errors]
    squared = [e * e for e in errors]
    denom = sum(abs(x) for x in a)
    wdenom = sum(max(0.0, w) * abs(x) for x, w in zip(a, weights))
    mape_vals = [
        abs(pp - aa) / abs(aa)
        for aa, pp, keep in zip(a, p, mape_mask)
        if keep and abs(aa) > EPS
    ]
    return {
        "n": n,
        "mae": _round(statistics.fmean(abs_errors)),
        "median_absolute_error": _round(statistics.median(abs_errors)),
        "wape_pct": _round(sum(abs_errors) / denom * 100.0, 4) if denom > EPS else None,
        "mape_pct": _round(statistics.fmean(mape_vals) * 100.0, 4) if mape_vals else None,
        "mape_n": len(mape_vals),
        "bias": _round(statistics.fmean(errors)),
        "rmse": _round(math.sqrt(statistics.fmean(squared))),
        "weighted_mae": _round(_weighted_mean(abs_errors, weights)),
        "weighted_wape_pct": _round(
            sum(max(0.0, w) * e for w, e in zip(weights, abs_errors)) / wdenom * 100.0,
            4,
        ) if wdenom > EPS else None,
        "weighted_bias": _round(_weighted_mean(errors, weights)),
        "secondary_percentage_metric": "MAPE_GUARDED",
        "primary_percentage_metric": "WAPE",
    }


def derive_activity_thresholds(fit_windows: list[dict[str, Any]]) -> dict[str, float]:
    """Derive low-activity thresholds from older fitting evidence only."""
    positive_rounds = [
        float(w["actual_rounds_24h_equiv"])
        for w in fit_windows
        if _f(w.get("actual_rounds_24h_equiv"), 0.0) > EPS
    ]
    positive_profit = [
        float(w["actual_grid_profit_24h_equiv_usdt"])
        for w in fit_windows
        if _f(w.get("actual_grid_profit_24h_equiv_usdt"), 0.0) > EPS
    ]
    # Evidence-driven lower quintile, with transparent practical caps so a regime
    # shift cannot redefine "low" as a high-activity day.
    rounds_p20 = _percentile(positive_rounds, 0.20) or 0.0
    profit_p20 = _percentile(positive_profit, 0.20) or 0.0
    return {
        "zero_rounds_epsilon": 1e-9,
        "zero_profit_epsilon_usdt": 1e-9,
        "low_rounds_max": _round(min(2.5, max(1.0, rounds_p20)), 6) or 2.0,
        "low_profit_max_usdt": _round(min(0.08, max(0.02, profit_p20)), 6) or 0.03,
        "derivation": "lower_20pct_of_positive_fit_activity_with_practical_caps",
        "fit_positive_rounds_p20": _round(rounds_p20),
        "fit_positive_profit_p20_usdt": _round(profit_p20),
    }


def classify_activity(window: dict[str, Any], thresholds: dict[str, float]) -> str:
    rounds = max(0.0, _f(window.get("actual_rounds_24h_equiv"), 0.0) or 0.0)
    profit = max(0.0, _f(window.get("actual_grid_profit_24h_equiv_usdt"), 0.0) or 0.0)
    if rounds <= thresholds["zero_rounds_epsilon"] or profit <= thresholds["zero_profit_epsilon_usdt"]:
        return "ZERO_ACTIVITY"
    if rounds <= thresholds["low_rounds_max"] or profit <= thresholds["low_profit_max_usdt"]:
        return "LOW_ACTIVITY"
    return "NORMAL_ACTIVITY"


def activity_bin(rounds: float) -> str:
    r = max(0.0, float(rounds))
    if r <= EPS:
        return "ZERO"
    if r <= 5.0:
        return "1_TO_5"
    if r <= 15.0:
        return "5_TO_15"
    if r <= 25.0:
        return "15_TO_25"
    return "25_PLUS"


def chronological_split(
    windows: list[dict[str, Any]],
    *,
    max_holdout: int = 6,
    prospective_n: int = 3,
    min_fit: int = 12,
) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(windows, key=lambda w: str(w.get("start_utc") or ""))
    n = len(ordered)
    if n < min_fit + 2:
        return {"fit": ordered, "holdout": [], "prospective": []}
    p = min(prospective_n, max(0, n - min_fit - 2))
    remaining = n - p
    h = min(max_holdout, max(2, remaining // 5))
    if remaining - h < min_fit:
        h = max(0, remaining - min_fit)
    cut_p = n - p if p else n
    cut_h = cut_p - h
    return {
        "fit": ordered[:cut_h],
        "holdout": ordered[cut_h:cut_p],
        "prospective": ordered[cut_p:],
    }


def recency_weights(
    windows: list[dict[str, Any]],
    *,
    half_life_days: float = DEFAULT_RECENCY_HALF_LIFE_DAYS,
    floor: float = DEFAULT_RECENCY_FLOOR,
) -> list[float]:
    if not windows:
        return []
    dates = []
    for w in windows:
        raw = str(w.get("start_utc") or "")
        try:
            dates.append(datetime.fromisoformat(raw.replace("Z", "+00:00")))
        except Exception:
            dates.append(None)
    valid = [d for d in dates if d is not None]
    if not valid:
        return [1.0] * len(windows)
    newest = max(valid)
    out = []
    hl = max(1.0, float(half_life_days))
    for dt in dates:
        if dt is None:
            out.append(float(floor))
            continue
        age_days = max(0.0, (newest - dt).total_seconds() / 86400.0)
        out.append(max(float(floor), 0.5 ** (age_days / hl)))
    return out


def add_geometry_descriptors(window: dict[str, Any]) -> dict[str, Any]:
    out = dict(window)
    lower = _f(out.get("lower_usdt"))
    upper = _f(out.get("upper_usdt"))
    market = _f(out.get("current_price_usdt"))
    grids = int(_f(out.get("grids"), 0.0) or 0)
    if lower is not None and upper is not None and market and upper > lower:
        width = upper - lower
        center = (lower + upper) / 2.0
        width_pct = width / market * 100.0
        out["width_pct_of_market"] = width_pct
        out["center_offset_pct_of_market"] = (center / market - 1.0) * 100.0
        out["grid_spacing_usdt"] = width / max(1, grids - 1)
        out["density_score"] = grids / max(width_pct, 1e-6)
        skew_ratio = (center - market) / width
        if skew_ratio > 1.0 / 6.0:
            out["center_skew"] = "UPPER_SKEW"
        elif skew_ratio < -1.0 / 6.0:
            out["center_skew"] = "LOWER_SKEW"
        else:
            out["center_skew"] = "NEAR_CENTRE"
    return out


def fit_regime_thresholds(fit_windows: list[dict[str, Any]]) -> dict[str, Any]:
    def vals(key: str) -> list[float]:
        return [float(v) for w in fit_windows if (v := _f(w.get(key))) is not None]
    atr = vals("atr14_pct")
    density = vals("density_score")
    efficiency = vals("efficiency_ratio")
    reversal = vals("reversal_rate")
    return {
        "atr14_pct_tertiles": [_percentile(atr, 1/3), _percentile(atr, 2/3)],
        "density_score_tertiles": [_percentile(density, 1/3), _percentile(density, 2/3)],
        "efficiency_median": _percentile(efficiency, 0.5),
        "reversal_median": _percentile(reversal, 0.5),
        "fit_n": len(fit_windows),
    }


def _tertile(value: Optional[float], cuts: list[Optional[float]]) -> str:
    if value is None or len(cuts) != 2 or cuts[0] is None or cuts[1] is None:
        return "UNKNOWN"
    if value <= float(cuts[0]):
        return "LOW"
    if value <= float(cuts[1]):
        return "MEDIUM"
    return "HIGH"


def assign_regime(window: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, str]:
    atr = _f(window.get("atr14_pct"))
    density = _f(window.get("density_score"))
    eff = _f(window.get("efficiency_ratio"))
    rev = _f(window.get("reversal_rate"))
    signed = _f(window.get("signed_move_pct"), 0.0) or 0.0
    eff_cut = _f(thresholds.get("efficiency_median"))
    rev_cut = _f(thresholds.get("reversal_median"))
    if eff is None or rev is None or eff_cut is None or rev_cut is None:
        path = "UNKNOWN"
    elif eff >= eff_cut and rev <= rev_cut:
        path = "TREND_UP" if signed >= 0 else "TREND_DOWN"
    else:
        path = "RANGING_MIXED"
    return {
        "volatility": _tertile(atr, thresholds.get("atr14_pct_tertiles") or []),
        "path": path,
        "density": _tertile(density, thresholds.get("density_score_tertiles") or []),
        "center_skew": str(window.get("center_skew") or "UNKNOWN"),
    }


def geometry_key(window: dict[str, Any]) -> tuple[Any, ...]:
    return (
        round(_f(window.get("lower_usdt"), 0.0) or 0.0, 2),
        round(_f(window.get("upper_usdt"), 0.0) or 0.0, 2),
        int(_f(window.get("grids"), 0.0) or 0),
        round(_f(window.get("quantity_per_grid_eth"), _f(window.get("quantity_per_grid_eth_est"), 0.0)) or 0.0, 6),
    )


def _similar_geometry(a: dict[str, Any], b: dict[str, Any]) -> bool:
    aw = _f(a.get("width_pct_of_market"))
    bw = _f(b.get("width_pct_of_market"))
    ag = int(_f(a.get("grids"), 0.0) or 0)
    bg = int(_f(b.get("grids"), 0.0) or 0)
    if aw is None or bw is None or ag <= 0 or bg <= 0:
        return False
    width_similar = abs(aw - bw) <= max(2.0, 0.20 * max(aw, bw))
    grid_similar = abs(ag - bg) <= max(3, round(0.15 * max(ag, bg)))
    skew_similar = (a.get("center_skew") == b.get("center_skew"))
    return bool(width_similar and grid_similar and skew_similar)


def choose_cohort(
    fit_windows: list[dict[str, Any]],
    target: dict[str, Any],
    *,
    min_n: int = DEFAULT_MIN_COHORT_N,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exact = [w for w in fit_windows if geometry_key(w) == geometry_key(target)]
    tr = target.get("regime") or {}
    similar = [w for w in fit_windows if _similar_geometry(w, target)]
    regime_available = (
        tr.get("volatility") not in (None, "UNKNOWN")
        and tr.get("path") not in (None, "UNKNOWN")
    )
    similar_regime = [
        w for w in similar
        if regime_available
        and (w.get("regime") or {}).get("volatility") == tr.get("volatility")
        and (w.get("regime") or {}).get("path") == tr.get("path")
    ]
    same_regime = [
        w for w in fit_windows
        if regime_available
        and (w.get("regime") or {}).get("volatility") == tr.get("volatility")
        and (w.get("regime") or {}).get("path") == tr.get("path")
    ]
    hierarchy = [
        ("EXACT_GEOMETRY", exact),
        ("SIMILAR_GEOMETRY_SAME_REGIME", similar_regime),
        ("SIMILAR_GEOMETRY", similar),
        ("SAME_MARKET_REGIME", same_regime),
        ("GLOBAL", fit_windows),
    ]
    for level, rows in hierarchy:
        if len(rows) >= min_n or level == "GLOBAL":
            return rows, {
                "fallback_level": level,
                "sample_n": len(rows),
                "target_regime": tr,
                "target_geometry": {
                    "lower_usdt": target.get("lower_usdt"),
                    "upper_usdt": target.get("upper_usdt"),
                    "grids": target.get("grids"),
                    "width_pct_of_market": _round(_f(target.get("width_pct_of_market")), 4),
                    "center_skew": target.get("center_skew"),
                },
                "minimum_cohort_n": min_n,
            }
    return fit_windows, {"fallback_level": "GLOBAL", "sample_n": len(fit_windows)}


def _clip(value: float, bounds: tuple[float, float]) -> float:
    return min(float(bounds[1]), max(float(bounds[0]), float(value)))


def fit_two_stage_factors(
    cohort: list[dict[str, Any]],
    *,
    bounds: tuple[float, float] = (0.35, 1.50),
) -> dict[str, Any]:
    if not cohort:
        return {
            "rounds_scale": 1.0,
            "productivity_scale": 1.0,
            "profit_scale": 1.0,
            "direct_profit_scale": 1.0,
            "sample_n": 0,
            "active_productivity_n": 0,
        }
    weights = recency_weights(cohort)
    rounds_ratios, rounds_weights = [], []
    direct_profit_ratios, profit_weights = [], []
    productivity_ratios, productivity_weights = [], []
    for w, wt in zip(cohort, weights):
        ar = _f(w.get("actual_rounds_24h_equiv"))
        pr = _f(w.get("predicted_paired_rounds_24h_raw"))
        ap = _f(w.get("actual_grid_profit_24h_equiv_usdt"))
        pp = _f(w.get("predicted_paired_profit_24h_raw_usdt"))
        if ar is not None and pr is not None and pr > EPS and ar >= 0:
            rounds_ratios.append(ar / pr)
            rounds_weights.append(wt)
        if ap is not None and pp is not None and pp > EPS and ap >= 0:
            direct_profit_ratios.append(ap / pp)
            profit_weights.append(wt)
        if ar is not None and pr is not None and ap is not None and pp is not None and ar > EPS and pr > EPS and pp > EPS:
            predicted_ppc = pp / pr
            actual_ppc = ap / ar
            if predicted_ppc > EPS:
                productivity_ratios.append(actual_ppc / predicted_ppc)
                productivity_weights.append(wt)

    raw_rounds_scale = _weighted_median(rounds_ratios, rounds_weights) or 1.0
    raw_direct_profit_scale = _weighted_median(direct_profit_ratios, profit_weights) or 1.0
    raw_productivity_scale = _weighted_median(productivity_ratios, productivity_weights) or 1.0
    rounds_scale = _clip(raw_rounds_scale, bounds)
    productivity_scale = _clip(raw_productivity_scale, bounds)
    profit_scale = _clip(rounds_scale * productivity_scale, bounds)
    direct_profit_scale = _clip(raw_direct_profit_scale, bounds)

    counts = {name: 0 for name, _, _ in ACTIVITY_BINS}
    for w in cohort:
        counts[activity_bin(_f(w.get("actual_rounds_24h_equiv"), 0.0) or 0.0)] += 1
    total = max(1, len(cohort))
    probs = {k: round(v / total * 100.0, 4) for k, v in counts.items()}
    return {
        "rounds_scale": _round(rounds_scale),
        "productivity_scale": _round(productivity_scale),
        "profit_scale": _round(profit_scale),
        "direct_profit_scale": _round(direct_profit_scale),
        "unclipped_rounds_scale": _round(raw_rounds_scale),
        "unclipped_productivity_scale": _round(raw_productivity_scale),
        "unclipped_direct_profit_scale": _round(raw_direct_profit_scale),
        "sample_n": len(cohort),
        "active_productivity_n": len(productivity_ratios),
        "recency_half_life_days": DEFAULT_RECENCY_HALF_LIFE_DAYS,
        "recency_floor": DEFAULT_RECENCY_FLOOR,
        "activity_distribution_pct": probs,
        "method": "two_stage_activity_scale_times_conditional_profit_per_cycle_scale",
    }


def prepare_windows(
    windows: list[dict[str, Any]],
    *,
    feature_enricher: Optional[Any] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    enriched = []
    for raw in windows:
        w = add_geometry_descriptors(raw)
        if feature_enricher is not None:
            extra = feature_enricher(w) or {}
            w.update(extra)
        enriched.append(w)
    split0 = chronological_split(enriched)
    thresholds = derive_activity_thresholds(split0["fit"])
    for w in enriched:
        w["activity_status"] = classify_activity(w, thresholds)
    split1 = chronological_split(enriched)
    regime_thresholds = fit_regime_thresholds(split1["fit"])
    for w in enriched:
        w["regime"] = assign_regime(w, regime_thresholds)
    return enriched, thresholds, regime_thresholds


def _predict_with_walk_forward(
    fit: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    bounds: tuple[float, float],
    min_cohort_n: int,
) -> tuple[list[float], list[float], list[dict[str, Any]]]:
    history = list(fit)
    pp, pr, meta = [], [], []
    for target in eval_rows:
        cohort, cohort_meta = choose_cohort(history, target, min_n=min_cohort_n)
        factors = fit_two_stage_factors(cohort, bounds=bounds)
        pred_p = (_f(target.get("predicted_paired_profit_24h_raw_usdt"), 0.0) or 0.0) * float(factors["profit_scale"])
        pred_r = (_f(target.get("predicted_paired_rounds_24h_raw"), 0.0) or 0.0) * float(factors["rounds_scale"])
        pp.append(pred_p)
        pr.append(pred_r)
        meta.append({"cohort": cohort_meta, "factors": factors})
        # Prospective/walk-forward: only after evaluating this target can it join history.
        history.append(target)
    return pp, pr, meta


def _metric_pair(rows: list[dict[str, Any]], pred_profit: list[float], pred_rounds: list[float]) -> dict[str, Any]:
    actual_profit = [_f(w.get("actual_grid_profit_24h_equiv_usdt"), 0.0) or 0.0 for w in rows]
    actual_rounds = [_f(w.get("actual_rounds_24h_equiv"), 0.0) or 0.0 for w in rows]
    weights = recency_weights(rows)
    normal_mask = [w.get("activity_status") == "NORMAL_ACTIVITY" for w in rows]
    return {
        "profit": error_metrics(actual_profit, pred_profit, weights=weights, mape_mask=normal_mask),
        "rounds": error_metrics(actual_rounds, pred_rounds, weights=weights, mape_mask=normal_mask),
    }


def legacy_global_holdout_metrics(
    fit: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    *,
    bounds: tuple[float, float],
) -> dict[str, Any]:
    profit_ratios = [
        float(w["actual_grid_profit_24h_equiv_usdt"]) / float(w["predicted_paired_profit_24h_raw_usdt"])
        for w in fit
        if (_f(w.get("predicted_paired_profit_24h_raw_usdt"), 0.0) or 0.0) > EPS
        and (_f(w.get("actual_grid_profit_24h_equiv_usdt"), -1.0) or 0.0) >= 0
    ]
    rounds_ratios = [
        float(w["actual_rounds_24h_equiv"]) / float(w["predicted_paired_rounds_24h_raw"])
        for w in fit
        if (_f(w.get("predicted_paired_rounds_24h_raw"), 0.0) or 0.0) > EPS
        and (_f(w.get("actual_rounds_24h_equiv"), -1.0) or 0.0) >= 0
    ]
    ps = _clip(statistics.median(profit_ratios) if profit_ratios else 1.0, bounds)
    rs = _clip(statistics.median(rounds_ratios) if rounds_ratios else 1.0, bounds)
    pp = [(_f(w.get("predicted_paired_profit_24h_raw_usdt"), 0.0) or 0.0) * ps for w in holdout]
    pr = [(_f(w.get("predicted_paired_rounds_24h_raw"), 0.0) or 0.0) * rs for w in holdout]
    return {"profit_scale": _round(ps), "rounds_scale": _round(rs), **_metric_pair(holdout, pp, pr)}


def quality_status(
    metrics: dict[str, Any],
    *,
    fit_n: int,
    holdout_n: int,
    low_share: float,
    cohort_fallbacks: list[str],
) -> dict[str, Any]:
    p = metrics.get("profit") or {}
    r = metrics.get("rounds") or {}
    pw = _f(p.get("wape_pct"), 999.0) or 999.0
    rw = _f(r.get("wape_pct"), 999.0) or 999.0
    def metric_or_default(value: Any, default: float = 999.0) -> float:
        parsed = _f(value)
        return float(default if parsed is None else parsed)

    pbias = abs(metric_or_default(p.get("bias")))
    rbias = abs(metric_or_default(r.get("bias")))
    pmae = metric_or_default(p.get("mae"))
    rmae = metric_or_default(r.get("mae"))
    if fit_n < 12 or holdout_n < 2:
        status = "INSUFFICIENT_EVIDENCE"
    else:
        # Thresholds are deliberately practical and multi-metric, not tuned to
        # make the model look good.  WAPE dominates; absolute error/bias/sample
        # quality can downgrade but MAPE cannot independently fail calibration.
        score = 0
        score += 0 if pw <= 35 else 1 if pw <= 55 else 2 if pw <= 80 else 3
        score += 0 if rw <= 35 else 1 if rw <= 55 else 2 if rw <= 80 else 3
        score += 0 if pmae <= 0.08 else 1 if pmae <= 0.15 else 2
        score += 0 if rmae <= 6 else 1 if rmae <= 10 else 2
        score += 1 if pbias > 0.12 else 0
        score += 1 if rbias > 8 else 0
        score += 1 if low_share > 0.40 else 0
        score += 1 if cohort_fallbacks and all(x == "GLOBAL" for x in cohort_fallbacks) else 0
        if score <= 2:
            status = "CALIBRATION_STRONG"
        elif score <= 5:
            status = "CALIBRATION_ACCEPTABLE"
        elif score <= 8:
            status = "CALIBRATION_CAUTION"
        else:
            status = "CALIBRATION_WEAK"
    forecast_confidence = {
        "CALIBRATION_STRONG": "HIGH",
        "CALIBRATION_ACCEPTABLE": "MEDIUM",
        "CALIBRATION_CAUTION": "LOW",
        "CALIBRATION_WEAK": "LOW",
        "INSUFFICIENT_EVIDENCE": "LOW",
    }[status]
    # Direction can remain useful with moderate absolute error, but an exact
    # percentage uplift should be called numerically reliable only when both
    # primary WAPE measures are materially tighter.  This prevents a merely
    # "acceptable" calibration from overstating +X% geometry improvements.
    magnitude_reliable = (
        status == "CALIBRATION_STRONG"
        or (status == "CALIBRATION_ACCEPTABLE" and pw <= 40.0 and rw <= 40.0 and pbias <= 0.08 and rbias <= 5.0)
    )
    return {
        "status": status,
        "forecast_confidence": forecast_confidence,
        "directional_signal_reliable": status in {"CALIBRATION_STRONG", "CALIBRATION_ACCEPTABLE", "CALIBRATION_CAUTION"},
        "magnitude_estimate_reliable": magnitude_reliable,
        "primary_quality_metrics": ["profit_wape_pct", "rounds_wape_pct", "profit_mae_usdt", "rounds_mae", "bias", "sample_coverage"],
        "mape_role": "SECONDARY_DIAGNOSTIC_ONLY",
    }


def upgrade_calibration(
    legacy: dict[str, Any],
    *,
    exclusions: Optional[list[dict[str, Any]]] = None,
    feature_enricher: Optional[Any] = None,
    scale_bounds: tuple[float, float] = (0.35, 1.50),
    min_cohort_n: int = DEFAULT_MIN_COHORT_N,
) -> dict[str, Any]:
    """Add v2 evidence alongside legacy fields without reinterpreting them."""
    out = deepcopy(legacy)
    windows_raw = list(legacy.get("windows") or [])
    windows, activity_thresholds, regime_thresholds = prepare_windows(
        windows_raw, feature_enricher=feature_enricher
    )
    split = chronological_split(windows)
    fit, holdout, prospective = split["fit"], split["holdout"], split["prospective"]

    hold_pp, hold_pr, hold_meta = _predict_with_walk_forward(
        fit, holdout, bounds=scale_bounds, min_cohort_n=min_cohort_n
    )
    # Prospective starts with all completed holdout evidence, still strictly older.
    pros_pp, pros_pr, pros_meta = _predict_with_walk_forward(
        fit + holdout, prospective, bounds=scale_bounds, min_cohort_n=min_cohort_n
    )
    hold_metrics = _metric_pair(holdout, hold_pp, hold_pr)
    prospective_metrics = _metric_pair(prospective, pros_pp, pros_pr)
    legacy_holdout = legacy_global_holdout_metrics(fit, holdout, bounds=scale_bounds)

    # Also replay the exact newest validation span used by the legacy production
    # report. This gives an apples-to-apples old-vs-new comparison while still
    # keeping the stricter fit/holdout/prospective split above for v2 governance.
    legacy_n = int((legacy.get("holdout_validation") or {}).get("validation_windows") or 6)
    legacy_n = min(max(2, legacy_n), max(2, len(windows) - 12)) if len(windows) >= 14 else 0
    if legacy_n:
        latest_fit = windows[:-legacy_n]
        latest_eval = windows[-legacy_n:]
        latest_pp, latest_pr, latest_meta = _predict_with_walk_forward(
            latest_fit, latest_eval, bounds=scale_bounds, min_cohort_n=min_cohort_n
        )
        latest_v2_metrics = _metric_pair(latest_eval, latest_pp, latest_pr)
        latest_legacy_metrics = legacy_global_holdout_metrics(latest_fit, latest_eval, bounds=scale_bounds)
    else:
        latest_eval, latest_meta = [], []
        latest_v2_metrics = {"profit": error_metrics([], []), "rounds": error_metrics([], [])}
        latest_legacy_metrics = latest_v2_metrics

    counts = {"NORMAL_ACTIVITY": 0, "LOW_ACTIVITY": 0, "ZERO_ACTIVITY": 0}
    for w in windows:
        counts[w.get("activity_status", "NORMAL_ACTIVITY")] += 1
    low_share = (counts["LOW_ACTIVITY"] + counts["ZERO_ACTIVITY"]) / max(1, len(windows))
    fallbacks = [m["cohort"].get("fallback_level") for m in hold_meta]
    q = quality_status(
        hold_metrics,
        fit_n=len(fit),
        holdout_n=len(holdout),
        low_share=low_share,
        cohort_fallbacks=[str(x) for x in fallbacks if x],
    )

    current_reference = windows[-1] if windows else {}
    current_cohort, current_cohort_meta = choose_cohort(fit + holdout + prospective, current_reference, min_n=min_cohort_n) if windows else ([], {"fallback_level": "GLOBAL", "sample_n": 0})
    current_factors = fit_two_stage_factors(current_cohort, bounds=scale_bounds)

    def improvement(new: Any, old: Any) -> Optional[float]:
        n, o = _f(new), _f(old)
        if n is None or o is None or abs(o) < EPS:
            return None
        return round((o - n) / abs(o) * 100.0, 4)

    v2 = {
        "schema": "pionex_paired_cycle_calibration_v2",
        "version": "2.0",
        "status": q["status"],
        "method": "chronological_walk_forward_cohort_calibration_with_activity_productivity_decomposition",
        "primary_percentage_metric": "WAPE",
        "secondary_percentage_metric": "GUARDED_MAPE_NORMAL_ACTIVITY_ONLY",
        "activity_thresholds": activity_thresholds,
        "activity": {
            "normal_windows": counts["NORMAL_ACTIVITY"],
            "low_activity_windows": counts["LOW_ACTIVITY"],
            "zero_activity_windows": counts["ZERO_ACTIVITY"],
            "low_or_zero_share_pct": round(low_share * 100.0, 4),
        },
        "validation": {
            "fit_windows": len(fit),
            "holdout_windows": len(holdout),
            "prospective_windows": len(prospective),
            "fit_range": [fit[0].get("start_utc"), fit[-1].get("end_utc")] if fit else None,
            "holdout_range": [holdout[0].get("start_utc"), holdout[-1].get("end_utc")] if holdout else None,
            "prospective_range": [prospective[0].get("start_utc"), prospective[-1].get("end_utc")] if prospective else None,
            "same_geometry_windows": len(windows),
            "excluded_geometry_change_windows": len(exclusions or []),
            "chronological_no_shuffle": True,
        },
        "regime_thresholds_fit_only": regime_thresholds,
        "holdout": hold_metrics,
        "prospective": prospective_metrics,
        "legacy_global_holdout_comparison": legacy_holdout,
        "latest_legacy_holdout_backtest": {
            "window_count": len(latest_eval),
            "range": [latest_eval[0].get("start_utc"), latest_eval[-1].get("end_utc")] if latest_eval else None,
            "v2": latest_v2_metrics,
            "legacy_global": latest_legacy_metrics,
            "v2_cohorts": [m.get("cohort") for m in latest_meta],
            "improvement_vs_legacy_pct": {
                "profit_wape": improvement((latest_v2_metrics.get("profit") or {}).get("wape_pct"), (latest_legacy_metrics.get("profit") or {}).get("wape_pct")),
                "profit_mae": improvement((latest_v2_metrics.get("profit") or {}).get("mae"), (latest_legacy_metrics.get("profit") or {}).get("mae")),
                "rounds_wape": improvement((latest_v2_metrics.get("rounds") or {}).get("wape_pct"), (latest_legacy_metrics.get("rounds") or {}).get("wape_pct")),
                "rounds_mae": improvement((latest_v2_metrics.get("rounds") or {}).get("mae"), (latest_legacy_metrics.get("rounds") or {}).get("mae")),
            },
            "note": "Strict walk-forward v2 replay on the exact newest validation span used by the legacy report; no future observations are used for earlier forecasts.",
        },
        "holdout_improvement_vs_legacy_pct": {
            "profit_wape": improvement((hold_metrics.get("profit") or {}).get("wape_pct"), (legacy_holdout.get("profit") or {}).get("wape_pct")),
            "profit_mae": improvement((hold_metrics.get("profit") or {}).get("mae"), (legacy_holdout.get("profit") or {}).get("mae")),
            "rounds_wape": improvement((hold_metrics.get("rounds") or {}).get("wape_pct"), (legacy_holdout.get("rounds") or {}).get("wape_pct")),
            "rounds_mae": improvement((hold_metrics.get("rounds") or {}).get("mae"), (legacy_holdout.get("rounds") or {}).get("mae")),
        },
        "current_reference_cohort": current_cohort_meta,
        "current_reference_factors": current_factors,
        "holdout_window_calibration": hold_meta,
        "prospective_window_calibration": pros_meta,
        "recency_weighting": {
            "enabled": True,
            "half_life_days": DEFAULT_RECENCY_HALF_LIFE_DAYS,
            "minimum_weight": DEFAULT_RECENCY_FLOOR,
            "unweighted_and_weighted_metrics_reported": True,
        },
        "quality": q,
        "window_exclusions": exclusions or [],
        "windows": windows,
    }
    out["calibration_v2"] = v2
    out["calibration_quality_v2"] = {
        **q,
        "profit": hold_metrics.get("profit"),
        "rounds": hold_metrics.get("rounds"),
        "activity": v2["activity"],
        "validation": v2["validation"],
        "cohort": current_cohort_meta,
        "legacy_mape": {
            "profit_mape_pct": (legacy.get("holdout_validation") or {}).get("profit_mape_pct"),
            "rounds_mape_pct": (legacy.get("holdout_validation") or {}).get("rounds_mape_pct"),
            "role": "SECONDARY_LEGACY_DIAGNOSTIC",
        },
    }
    return out


def classify_state_pair_for_calibration(
    start: dict[str, Any],
    end: dict[str, Any],
    *,
    min_h: float = 14.0,
    max_h: float = 30.0,
    geometry_fields: tuple[str, ...] = (
        "lower_limit_usdt", "upper_limit_usdt", "grids", "quantity_per_grid_eth"
    ),
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Explain whether a state pair is eligible for geometry-performance calibration.

    This helper exists so geometry edits are never silently dropped.  It expects
    state rows with `_dt` datetimes (the same shape used by the existing Pionex
    readers) and returns an explicit reason for every excluded pair.
    """
    start_dt = start.get("_dt")
    end_dt = end.get("_dt")
    if start_dt is None or end_dt is None:
        return {"eligible": False, "reason": "CALIBRATION_WINDOW_EXCLUDED_MISSING_TIMESTAMP"}
    elapsed_h = (end_dt - start_dt).total_seconds() / 3600.0
    base = {
        "start_utc": start_dt.isoformat(),
        "end_utc": end_dt.isoformat(),
        "elapsed_h": round(elapsed_h, 4),
    }
    changed: list[str] = []
    for field in geometry_fields:
        a, b = start.get(field), end.get(field)
        if field == "grids":
            try:
                same = int(float(a)) == int(float(b))
            except Exception:
                same = False
        else:
            fa, fb = _f(a), _f(b)
            same = fa is not None and fb is not None and abs(fa - fb) <= tolerance
        if not same:
            changed.append(field)
    if changed:
        return {
            **base,
            "eligible": False,
            "reason": "CALIBRATION_WINDOW_EXCLUDED_GEOMETRY_CHANGE",
            "changed_geometry_fields": changed,
            "duration_eligible": bool(min_h <= elapsed_h <= max_h),
        }
    if not (min_h <= elapsed_h <= max_h):
        return {**base, "eligible": False, "reason": "CALIBRATION_WINDOW_EXCLUDED_DURATION"}
    return {**base, "eligible": True, "reason": "CALIBRATION_WINDOW_ELIGIBLE_SAME_GEOMETRY"}


def audit_state_pair_exclusions(
    states: list[dict[str, Any]],
    *,
    min_h: float = 14.0,
    max_h: float = 30.0,
) -> dict[str, Any]:
    records = [
        classify_state_pair_for_calibration(a, b, min_h=min_h, max_h=max_h)
        for a, b in zip(states, states[1:])
    ]
    eligible = [r for r in records if r.get("eligible")]
    geometry_excluded = [
        r for r in records
        if r.get("reason") == "CALIBRATION_WINDOW_EXCLUDED_GEOMETRY_CHANGE"
    ]
    return {
        "pair_audit_n": len(records),
        "same_geometry_eligible_pair_count": len(eligible),
        "excluded_geometry_change_count": len(geometry_excluded),
        "geometry_change_exclusions": geometry_excluded,
        "all_pair_classifications": records,
    }


def target_factors(
    calibration: dict[str, Any],
    target: dict[str, Any],
    *,
    min_cohort_n: int = DEFAULT_MIN_COHORT_N,
    bounds: tuple[float, float] = (0.35, 1.50),
) -> dict[str, Any]:
    v2 = calibration.get("calibration_v2") or {}
    windows = list(v2.get("windows") or [])
    if not windows:
        return {
            "profit_scale": float(calibration.get("profit_scale_applied") or 1.0),
            "rounds_scale": float(calibration.get("rounds_scale_applied") or 1.0),
            "cohort": {"fallback_level": "LEGACY_GLOBAL", "sample_n": 0},
            "method": "legacy_fallback",
        }
    target2 = add_geometry_descriptors(target)
    target2["regime"] = assign_regime(target2, v2.get("regime_thresholds_fit_only") or {})
    cohort, meta = choose_cohort(windows, target2, min_n=min_cohort_n)
    factors = fit_two_stage_factors(cohort, bounds=bounds)
    return {**factors, "cohort": meta, "method": "v2_cohort_two_stage"}
