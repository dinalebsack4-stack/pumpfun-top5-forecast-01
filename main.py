#!/usr/bin/env python3
"""Deterministic Monte Carlo forecast for pump.fun top-5 market caps.

Uses only Python's standard library. Reads /input/task.json and
/input/history.ndjson (overridable for local runs) and writes forecast.json.
"""
from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
DRAWS = 1400
MAX_CANDIDATES = 200
def parse_time(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def logcap(row: dict) -> float:
    # log10 of cap in SOL, matching the competition's scoring bins.
    return math.log10(int(row["market_cap_lamports"])) - 9.0


def linear_slope(points: list[tuple[float, float]]) -> float:
    """Robust-ish log-cap slope per hour using a Theil-Sen median."""
    if len(points) < 2:
        return 0.0
    slopes = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dt = points[j][0] - points[i][0]
            if dt > 1e-6:
                slopes.append((points[j][1] - points[i][1]) / dt)
    return median(slopes) if slopes else 0.0


def normal_cdf(x: float, mean: float, sd: float) -> float:
    return 0.5 * (1.0 + math.erf((x - mean) / (sd * math.sqrt(2.0))))


def main() -> None:
    task = json.loads((INPUT_DIR / "task.json").read_text())
    raw_marks = [json.loads(line) for line in (INPUT_DIR / "history.ndjson").read_text().splitlines() if line.strip()]
    marks = [m for m in raw_marks if m.get("ok") and isinstance(m.get("rows"), list)]
    if not marks:
        raise RuntimeError("history contains no successful marks")

    mark_times = [parse_time(m["at"]) for m in marks]
    series: dict[str, list[tuple[float, float, dict]]] = defaultdict(list)
    for t, mark in zip(mark_times, marks):
        for row in mark["rows"]:
            try:
                cap = logcap(row)
                if math.isfinite(cap):
                    series[row["mint"]].append((t, cap, row))
            except (KeyError, ValueError, TypeError, OverflowError):
                continue

    last_time = mark_times[-1]
    # The current chart is the most reliable candidate pool. Include recent
    # near-misses as well, but stale names receive a rank-based survival penalty.
    latest_mark = marks[-1]
    last_rows = latest_mark["rows"]
    current = {r["mint"]: r for r in last_rows if "mint" in r}
    candidates = []
    for mint, points in series.items():
        t, cap, row = points[-1]
        age_hours = max(0.0, (last_time - t) / 3600.0)
        if mint not in current and age_hours > 2.5:
            continue
        # The mark only records a bounded chart. A token that fell out is
        # unlikely to recover to the top, so shrink its starting level.
        rank = int(row.get("rank", 200))
        stale_penalty = min(1.2, 0.12 * age_hours + (0.28 if mint not in current else 0.0))
        pts = [(pt[0], pt[1]) for pt in points[-8:]]
        slope = linear_slope(pts)
        # Limit extrapolation of short-lived pump.fun bursts.
        slope = max(-0.24, min(0.24, slope))
        # Volatility from adjacent changes; use a floor for unobserved variation.
        changes = [abs(pts[i][1] - pts[i - 1][1]) for i in range(1, len(pts))]
        vol = median(changes) if changes else 0.28
        sigma_hour = max(0.22, min(0.75, 0.19 + 0.50 * vol))
        candidates.append({
            "mint": mint,
            "cap": cap - stale_penalty,
            "slope": slope,
            "sigma": sigma_hour,
            "rank": rank,
            "age": age_hours,
            "row": row,
        })

    # Always retain current top chart, ordered by current rank; cap work/memory.
    candidates.sort(key=lambda c: (0 if c["mint"] in current else 1, c["rank"], c["mint"]))
    candidates = candidates[:MAX_CANDIDATES]
    if len(candidates) < task["places"]:
        raise RuntimeError("not enough candidate mints in history")

    # Use the forecast horizons in task.json. Monte Carlo over each answer
    # horizon; average distributions because one forecast is scored at all reads.
    read_times = [parse_time(x) for x in task["reads"]]
    horizons = [max(0.0, (t - last_time) / 3600.0) for t in read_times]
    rng = random.Random(7122026)
    places = int(task["places"])
    place_probs: list[dict[str, float]] = [defaultdict(float) for _ in range(places)]
    size_probs: list[dict[int, float]] = [defaultdict(float) for _ in range(places)]
    expected_place: dict[str, list[float]] = defaultdict(lambda: [0.0] * places)

    # The unobserved pool includes coins born after the cut and late graduates.
    # Its weight rises with forecast horizon; residual probability is deliberately
    # left unnamed in the output and shared by all answer-chart coins.
    edges = task["size"]["edges"]
    for horizon in horizons:
        known_share = 0.72 * math.exp(-horizon / 24.0)
        accum_place = [defaultdict(float) for _ in range(places)]
        accum_size = [defaultdict(float) for _ in range(places)]
        accum_rank = defaultdict(lambda: [0.0] * places)
        for _ in range(DRAWS):
            sampled = []
            for c in candidates:
                # Trend decays with time, avoiding unbounded linear extrapolation.
                trend_hours = 2.8 * (1.0 - math.exp(-horizon / 2.8))
                mean = c["cap"] + c["slope"] * trend_hours
                sd = c["sigma"] * math.sqrt(max(0.5, horizon))
                # Common volatility is intentional: volatile names can move
                # dramatically, while rank is determined by relative outcomes.
                value = rng.gauss(mean, sd)
                sampled.append((value, c["mint"]))
            sampled.sort(key=lambda pair: (-pair[0], pair[1]))
            top = sampled[:places]
            for pos, (value, mint) in enumerate(top):
                accum_place[pos][mint] += 1.0 / DRAWS
                accum_rank[mint][pos] += 1.0 / DRAWS
                bin_index = 0
                while bin_index < len(edges) and value >= edges[bin_index]:
                    bin_index += 1
                accum_size[pos][bin_index] += 1.0 / DRAWS
        # Downweight seen-cohort probabilities to leave mass for unseen coins.
        for place in range(places):
            for mint, p in accum_place[place].items():
                place_probs[place][mint] += p * known_share / len(horizons)
            # Mix simulated seen-cohort sizes with a broad extrapolated unseen
            # market-cap prior. The probabilities are binned and then weighted.
            for bi, p in accum_size[place].items():
                size_probs[place][bi] += p * known_share / len(horizons)
            # Approximate an unknown winner's cap distribution with a lognormal
            # prior centered on large pump.fun winners; no external data used.
            unseen_share = 1.0 - known_share
            mu = 3.65 + 0.012 * horizon
            sd = 0.62 + 0.018 * horizon
            for bi in range(len(edges) + 1):
                lo = -math.inf if bi == 0 else edges[bi - 1]
                hi = math.inf if bi == len(edges) else edges[bi]
                p = max(0.0, normal_cdf(hi, mu, sd) - normal_cdf(lo, mu, sd))
                size_probs[place][bi] += p * unseen_share / len(horizons)
        for mint, ranks in accum_rank.items():
            for pos in range(places):
                expected_place[mint][pos] += ranks[pos] / len(horizons)

    # Cap named mass to avoid falsely treating the known cohort as exhaustive.
    # Use only the strongest candidates at each place, leaving the rest shared.
    probabilities = []
    for place in range(places):
        ranked = sorted(
            ((mint, p) for mint, p in place_probs[place].items() if mint in current),
            key=lambda item: (-item[1], item[0]),
        )[:100]
        # Renormalize only for floating noise; expected named mass remains below 1.
        total = sum(p for _, p in ranked)
        if total > 0.985:
            scale = 0.985 / total
            ranked = [(mint, p * scale) for mint, p in ranked]
        for mint, p in ranked:
            if p > 1e-8:
                probabilities.append({"mint": mint, "place": place + 1, "p": round(p, 8)})

    # Sizes are a complete averaged distribution, with tiny values omitted only
    # when their accumulated probability rounds to zero at 8 decimal places.
    sizes = []
    for place in range(places):
        total = sum(size_probs[place].values())
        if total <= 0:
            size_probs[place][40] = 1.0
            total = 1.0
        for bi in range(len(edges) + 1):
            p = size_probs[place].get(bi, 0.0) / total
            if p >= 0.000000005:
                sizes.append({"place": place + 1, "bin": bi, "p": round(p, 8)})

    # Tie-break guess: greedily pick distinct coins maximizing averaged
    # probability of occupying the corresponding place.
    top5 = []
    for pos in range(places):
        available = [m for m in expected_place if m not in top5]
        if not available:
            available = [c["mint"] for c in candidates if c["mint"] not in top5]
        best = max(available, key=lambda m: (expected_place[m][pos], -sum(expected_place[m][:pos]), m))
        top5.append(best)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / task["output"]
    result = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    out_path.write_text(json.dumps(result, separators=(",", ":"), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
