"""
Verification for check_radar_gaps.py: every severity tier must be producible
on demand (Pattern AP Phase 4 — a signal you cannot make fire is decoration).

Each case builds a synthetic "present frames" list with a hole in a known
position, runs the real find_gaps() + classify(), and asserts the tier. The
upstream probe is stubbed so the cases are deterministic and offline; case 5
is the one that exercises the 404 path.

Expectation: all 7 cases PASS. Any FAIL means the alert would misclassify a
real gap — either paging on something unactionable (fatigue) or, worse, staying
silent on a recoverable one.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import check_radar_gaps as crg  # noqa: E402

NOW = datetime(2026, 9, 22, 12, 0)
STEP = timedelta(minutes=5)


def build(hole_start: datetime, n_slots: int, span_days: int = 40):
    """All 5-min slots from span_days ago to NOW, minus n_slots at hole_start."""
    start = NOW - timedelta(days=span_days)
    hole = {hole_start + STEP * i for i in range(n_slots)}
    out, t = [], start
    while t <= NOW:
        if t not in hole:
            out.append(t)
        t += STEP
    return out


def run(present, floods=(), upstream=True):
    crg.probe_upstream = lambda g, s: upstream          # stub the network
    gaps = crg.find_gaps(present, NOW)
    crg.classify(gaps, list(floods), probe=True)
    return gaps


def pick(gaps, hole_start):
    for g in gaps:
        if g["start"] == hole_start:
            return g
    return None


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<46} got={got!s:<28} want={want}")
    return ok


def main() -> None:
    results = []

    # 1. In-window, aged past AGING_THRESHOLD_H, upstream has it -> CRITICAL.
    h = NOW - timedelta(days=5)
    g = pick(run(build(h, 6)), h)
    results.append(check("5d-old recoverable gap", (g["severity"], g["recoverable"]),
                         ("CRITICAL", True)))

    # 2. Recent gap -> WARN only; the daily catch-up scraper is expected to heal it.
    h = NOW - timedelta(days=1)
    g = pick(run(build(h, 6)), h)
    results.append(check("1d-old gap", g["severity"], "WARN"))

    # 3. Past retention -> INFO, and never actionable.
    h = NOW - timedelta(days=30)
    g = pick(run(build(h, 6)), h)
    results.append(check("30d-old gap", (g["severity"], g["recoverable"]),
                         ("INFO", False)))

    # 4. Overlapping a flood label -> CRITICAL even while still recent, because
    #    RadarDataset._contiguous would silently drop the event's sample.
    h = NOW - timedelta(days=1)
    flood = h + STEP * 2
    g = pick(run(build(h, 6), floods=[flood]), h)
    results.append(check("gap overlapping a flood label",
                         (g["severity"], bool(g["floods"])), ("CRITICAL", True)))

    # 5. In-window but NEA serves 404 -> INFO, must NOT page (the 2026-09-16 and
    #    2026-09-17 gaps are real instances of this).
    h = NOW - timedelta(days=5)
    g = pick(run(build(h, 6), upstream=False), h)
    results.append(check("in-window gap, upstream 404", g["severity"], "INFO"))

    # 6. A flood-overlapping gap that NEA cannot serve is still not actionable --
    #    nothing to fetch -- so it must not page either.
    h = NOW - timedelta(days=5)
    g = pick(run(build(h, 6), floods=[h + STEP * 2], upstream=False), h)
    results.append(check("flood gap, upstream 404", g["severity"], "INFO"))

    # 7. The freshest slots are expected to be briefly absent and must not alert.
    h = NOW - timedelta(minutes=20)
    g = pick(run(build(h, 4)), h)
    results.append(check("gap inside fresh-lag window", g, None))

    print(f"\n{sum(results)}/{len(results)} passed")
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
