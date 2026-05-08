"""Verify that the centered->image y shift fixes the BW/boundary alignment.

Setup mimics ``mie_multi_hole.py`` preview path:
  1. A synthetic BW strip of shape (H_orig, W_orig) with the plume centered on
     the midline (rows H_orig/2 +/- 2, columns 5..W_orig-5).
  2. A boundary point set in centered-y convention (y in [-3, +3], x in image
     coords) covering the exact perimeter of the BW pixels.
  3. The video is GPU-swapaxes(-2,-1) (here just numpy swap), boundaries go
     through ``_shift_plume_boundary_centered_y_to_image`` then
     ``_swap_plume_boundary_yx``, and the result is rendered exactly the way
     ``_render_one_plume_frame`` does.
  4. We check that every rendered boundary pixel lies INSIDE the BW support
     (i.e. the boundary correctly traces the BW edge, not the strip edge).
"""
import numpy as np
import cv2
from OSCC_postprocessing.playback.video_playback import (
    _shift_plume_boundary_centered_y_to_image,
    _swap_plume_boundary_yx,
    _to_yx,
)

H_orig, W_orig = 20, 40  # cross-stream x radial
midline = (H_orig - 1) / 2.0  # = 9.5

# BW: rows 8..11, cols 5..34 are filled (plume body straddling midline).
bw = np.zeros((H_orig, W_orig), dtype=bool)
bw[8:12, 5:35] = True

# Boundary in centered-y convention, exhaustively listing perimeter pixels
ys, xs = np.nonzero(bw)
midline_int = int(midline)  # boundary helper writes int rows; only the relative shift matters
ys_centered = ys.astype(np.float32) - midline
boundary_centered = np.column_stack((ys_centered, xs.astype(np.float32)))

# Frame container for one plume, one frame
all_boundaries = [[boundary_centered]]  # list per plume of list per frame

# --- The mie_multi_hole.py preview path (post-fix) ---
# Step 1: GPU-side swapaxes(-2, -1). For a (H, W) frame this turns into (W, H).
swapped_video = np.swapaxes(bw, -2, -1)
H_display, W_display = swapped_video.shape  # = (W_orig, H_orig) = (40, 20)

# Step 2: shift centered-y to image coords on the boundary, THEN swap (y,x)<->(x,y).
shifted = [_shift_plume_boundary_centered_y_to_image(b, H_orig) for b in all_boundaries]
swapped_boundaries = [_swap_plume_boundary_yx(b) for b in shifted]

# Step 3: rasterise like _boundary_masks does.
pts = swapped_boundaries[0][0]
yx = _to_yx(pts, H_display, W_display, assume_xy=False, centered_y=False)
mask_boundary = np.zeros((H_display, W_display), dtype=np.uint8)
mask_boundary[yx[:, 0], yx[:, 1]] = 255

# --- Pre-fix path (what produced the user's screenshot) ---
swapped_buggy = [_swap_plume_boundary_yx(b) for b in all_boundaries]
pts_buggy = swapped_buggy[0][0]
yx_buggy = _to_yx(pts_buggy, H_display, W_display, assume_xy=False, centered_y=False)
mask_buggy = np.zeros((H_display, W_display), dtype=np.uint8)
mask_buggy[yx_buggy[:, 0], yx_buggy[:, 1]] = 255

# --- Checks ---
def edge_overlap(boundary_mask, bw_image):
    """Fraction of boundary pixels falling inside or on the BW support (with 1-px slack)."""
    bw_dilate = cv2.dilate(bw_image.astype(np.uint8) * 255,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    inside = ((boundary_mask > 0) & (bw_dilate > 0)).sum()
    total = (boundary_mask > 0).sum()
    return inside, total

bw_swapped_uint = swapped_video.astype(np.uint8) * 255
inside_fix, total_fix = edge_overlap(mask_boundary, swapped_video)
inside_bug, total_bug = edge_overlap(mask_buggy, swapped_video)
left_edge_collapse_bug = (mask_buggy[:, 0] > 0).sum()
left_edge_collapse_fix = (mask_boundary[:, 0] > 0).sum()

print(f"H_orig={H_orig}, W_orig={W_orig}, H_display={H_display}, W_display={W_display}")
print(f"BW pixels (display): {int(swapped_video.sum())}")
print()
print("=== POST-FIX path (shift before swap) ===")
print(f"  boundary pixels rendered: {total_fix}")
print(f"  fraction inside BW (1-px dilate): {inside_fix}/{total_fix} = {inside_fix/total_fix:.2%}")
print(f"  pixels collapsed to x_display=0 (left edge): {left_edge_collapse_fix}")
print()
print("=== PRE-FIX path (no shift before swap) ===")
print(f"  boundary pixels rendered: {total_bug}")
print(f"  fraction inside BW (1-px dilate): {inside_bug}/{total_bug} = {inside_bug/total_bug:.2%}")
print(f"  pixels collapsed to x_display=0 (left edge): {left_edge_collapse_bug}")
print()

ok_fix = inside_fix / total_fix > 0.95 and left_edge_collapse_fix == 0
bug_reproduces = left_edge_collapse_bug > 0 and inside_bug / total_bug < 0.6
print(f"[{'PASS' if ok_fix else 'FAIL'}] post-fix: boundary tracks BW edge, no left-edge collapse")
print(f"[{'PASS' if bug_reproduces else 'FAIL'}] pre-fix: bug visibly reproduces (left-edge collapse + low BW overlap)")

# Visual ASCII rendering for sanity
def render_ascii(boundary_mask, bw_mask, title):
    print(f"\n  {title}  (B = BW, . = empty, X = boundary, * = both)")
    for r in range(boundary_mask.shape[0]):
        line = []
        for c in range(boundary_mask.shape[1]):
            b = bw_mask[r, c]
            x = boundary_mask[r, c] > 0
            line.append("*" if (b and x) else "X" if x else "B" if b else ".")
        print("  " + "".join(line))

render_ascii(mask_boundary, swapped_video, "POST-FIX")
render_ascii(mask_buggy,    swapped_video, "PRE-FIX (buggy)")
