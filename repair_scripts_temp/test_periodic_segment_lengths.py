"""Verification for the periodic_true_segment_lengths fix in
OSCC_postprocessing/binary_ops/masking.py.

Covers:
  - Original docstring examples (regression).
  - The mask[0]==True case that triggered the production failures
    (DS300, Nozzle 0/2/5/7).
  - Boundary cases: empty, all-False, all-True.
  - A synthetic 10-plume injector mask (720 bins, all plume widths 28 bins)
    rotated through every offset; sum(widths_deg) must equal total_deg for
    every rotation -- this is the invariant violated by the production bug.
"""
import numpy as np
from OSCC_postprocessing.binary_ops.masking import (
    periodic_true_segment_lengths,
    periodic_true_segment_angles,
)


def to_list(arr):
    # Handle both numpy and cupy backends transparently.
    if hasattr(arr, "get"):
        arr = arr.get()
    return sorted(int(x) for x in np.asarray(arr).tolist())


def to_numpy(arr):
    if hasattr(arr, "get"):
        arr = arr.get()
    return np.asarray(arr)


def assert_equal(name, got, expected):
    g, e = to_list(got), sorted(expected)
    ok = g == e
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={g} expected={e}")
    return ok


print("=== Docstring regression ===")
ok1 = assert_equal("docstring case 1", periodic_true_segment_lengths([False, True, True, False, True]), [2, 1])
ok2 = assert_equal("docstring case 2", periodic_true_segment_lengths([True, True, True]), [3])

print("\n=== Bug-trigger case: mask starts on True, multiple segments ===")
# [T,T,F,T,T,F,F,T] periodic -> {7,0,1} wrap = 3 cells, plus {3,4} = 2 cells
ok3 = assert_equal("mask starts True, two segments",
                   periodic_true_segment_lengths([True, True, False, True, True, False, False, True]),
                   [3, 2])
# Single segment that wraps
ok4 = assert_equal("single wrap segment",
                   periodic_true_segment_lengths([True, False, False, True, True]),
                   [3])

print("\n=== Boundary cases ===")
ok5 = assert_equal("empty",       periodic_true_segment_lengths([]), [])
ok6 = assert_equal("all False",   periodic_true_segment_lengths([False, False, False]), [])
ok7 = assert_equal("all True",    periodic_true_segment_lengths([True, True, True, True]), [4])

print("\n=== 10-plume injector simulation: invariant sum(widths) == total ===")
n_bins, n_plumes, plume_w_bins = 720, 10, 28  # 14 deg per plume, 36 deg spacing
gap = (n_bins - n_plumes * plume_w_bins) // n_plumes  # 44 bins gap
print(f"  n_bins={n_bins}, n_plumes={n_plumes}, plume_w={plume_w_bins} bins, gap={gap} bins")
print(f"  expected per-plume width = {plume_w_bins * 360 / n_bins} deg")
print(f"  expected total occupied  = {n_plumes * plume_w_bins * 360 / n_bins} deg")

worst_dev = 0.0
all_ok = True
for offset in range(0, n_bins, 7):  # sample many offsets including odd ones
    base = np.zeros(n_bins, dtype=bool)
    for k in range(n_plumes):
        s = (offset + k * (plume_w_bins + gap)) % n_bins
        for j in range(plume_w_bins):
            base[(s + j) % n_bins] = True
    total_deg = float(base.sum() * 360.0 / n_bins)
    widths_deg = to_numpy(periodic_true_segment_angles(base))
    n_seg = int(widths_deg.size)
    sum_deg = float(np.sum(widths_deg))
    dev = abs(sum_deg - total_deg)
    worst_dev = max(worst_dev, dev)
    if n_seg != n_plumes or dev > 1e-6:
        all_ok = False
        print(f"  [FAIL] offset={offset}: n_seg={n_seg} sum={sum_deg:.2f} total={total_deg:.2f}")
print(f"  Tested {n_bins // 7 + 1} rotational offsets; worst |sum-total|={worst_dev:.2e} deg")
print(f"  [{'PASS' if all_ok else 'FAIL'}] invariant sum(widths) == total holds for all offsets")

print("\n=== Sanity: bug-trigger offset would have produced ratio ~25x without fix ===")
# Recreate a layout we observed in the audit: 9 segments, total~119 deg, sum_w was 2999 deg
n_bins2, n_plumes2 = 720, 9
plume_w = round(119.0 / n_plumes2 / 360.0 * n_bins2)  # ~26 bins per plume
mask2 = np.zeros(n_bins2, dtype=bool)
spacing = n_bins2 // n_plumes2
for k in range(n_plumes2):
    s = (k * spacing) % n_bins2  # offset 0 -> mask[0] = True
    for j in range(plume_w):
        mask2[(s + j) % n_bins2] = True
total2 = float(mask2.sum() * 360 / n_bins2)
widths2 = to_numpy(periodic_true_segment_angles(mask2))
print(f"  9-segment mask with mask[0]=True: total={total2:.1f} deg, sum(widths)={float(np.sum(widths2)):.1f} deg")
print(f"  per-segment widths: {[round(float(w),1) for w in widths2.tolist()]}")
print(f"  [{'PASS' if abs(float(np.sum(widths2)) - total2) < 1e-6 else 'FAIL'}] equality holds")

print("\nALL PASS" if all([ok1, ok2, ok3, ok4, ok5, ok6, ok7, all_ok]) else "\nFAILURES")
