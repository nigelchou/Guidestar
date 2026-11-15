"""
Edge exclusion for starfish spots, derived ONLY from starfish's own images.

Per FOV, each edge border = max over rounds of
    max(zero-fill strip width, phase-correlation shift vs REF_ROUND) + PADDING
and a spot is kept only inside the window decodePixels-style:
    y in [upper+1, ydim-lower),  x in [left+1, xdim-right)
"""
import os

import numpy as np
import pandas as pd
from skimage.registration import phase_cross_correlation

# ---- Main folder = the repository folder, which holds Data/ (layout described in Data/README.md). ----
# Run this from the folder the file is in, or set the environment variable GUIDESTAR_MAIN_DIR to the repository path.
main_dir = os.environ.get('GUIDESTAR_MAIN_DIR', '..')


STARFISH_DIR = (os.path.join(main_dir, "Data/benchmarking/starfish/brain/starfish_result"))
STARFISH_CSV = os.path.join(STARFISH_DIR, "lm50GS_brain_starfish_multifov.csv")
OUT_CSV      = STARFISH_CSV.replace(".csv", "_edgecrop.csv")
BORDER_CSV   = STARFISH_CSV.replace(".csv", "_edgecrop_borders.csv")

PADDING           = 30   # same safety margin as BFT ImageData.border_padding
REF_ROUND         = 0
MAX_TRUSTED_SHIFT = 100  # px; larger phase-corr shifts are treated as failures


def zero_strip_widths(img):
    """Widths (top, bottom, left, right) of all-zero strips at the image edges,
    i.e. the fill left behind when starfish translates a round."""
    zero_rows = ~np.any(img != 0, axis=1)
    zero_cols = ~np.any(img != 0, axis=0)

    def leading(v):
        return len(v) if v.all() else int(np.argmin(v))

    return (leading(zero_rows), leading(zero_rows[::-1]),
            leading(zero_cols), leading(zero_cols[::-1]))


def starfish_borders(npy_path, padding=PADDING):
    """Return borders, image size and per-round diagnostics for one FOV."""
    stack = np.load(npy_path, mmap_mode="r")  # (round, ch, z, y, x)
    n_rounds, _, _, ydim, xdim = stack.shape
    ref = np.asarray(stack[REF_ROUND, 0, 0], dtype=np.float32)

    shifts, strips, failed = [], [], []
    for r in range(n_rounds):
        img = np.asarray(stack[r, 0, 0], dtype=np.float32)
        strips.append(zero_strip_widths(img))

        if r == REF_ROUND:
            shifts.append((0.0, 0.0))
            continue
        shift, _, _ = phase_cross_correlation(ref, img)
        if np.any(np.abs(shift) > MAX_TRUSTED_SHIFT):
            failed.append(r)
            shift = (0.0, 0.0)
        shifts.append(tuple(shift))

    shifts = np.asarray(shifts)   # (round, 2): (dy, dx) to move round onto ref
    strips = np.asarray(strips)   # (round, 4): top, bottom, left, right

    # +dy means the round lacks data at the top, -dy at the bottom (same as BFT)
    shift_edges = np.array([
        max(0.0, shifts[:, 0].max()), max(0.0, -shifts[:, 0].min()),
        max(0.0, shifts[:, 1].max()), max(0.0, -shifts[:, 1].min()),
    ])
    edges = np.maximum(np.ceil(shift_edges), strips.max(axis=0)).astype(int)
    upper, lower, left, right = (edges + padding).tolist()

    diag = dict(max_shift_px=float(np.abs(shifts).max()),
                max_zero_strip_px=int(strips.max()),
                failed_rounds=",".join(map(str, failed)))
    return (upper, lower, left, right), (ydim, xdim), diag


df = pd.read_csv(STARFISH_CSV).reset_index(drop=True)
keep = np.zeros(len(df), dtype=bool)
border_rows = []

for fov, idx in df.groupby("fov").groups.items():
    npy_path = os.path.join(STARFISH_DIR, f"{fov}_projected.npy")
    if not os.path.exists(npy_path):
        print(f"{fov}: {os.path.basename(npy_path)} not found -> "
              f"{len(idx)} spots dropped")
        continue

    (up, lo, le, ri), (ydim, xdim), diag = starfish_borders(npy_path)

    sub = df.loc[idx]
    yi = np.round(sub["y"].to_numpy()).astype(int)
    xi = np.round(sub["x"].to_numpy()).astype(int)
    m = ((yi >= up + 1) & (yi < ydim - lo) &
         (xi >= le + 1) & (xi < xdim - ri))
    keep[np.asarray(idx)] = m

    border_rows.append(dict(fov=fov, ydim=ydim, xdim=xdim,
                            upper=up, lower=lo, left=le, right=ri,
                            n_in=len(idx), n_kept=int(m.sum()), **diag))
    print(f"{fov}: borders U{up} D{lo} L{le} R{ri}  "
          f"kept {m.sum():,}/{len(idx):,}  {diag}")

borders = pd.DataFrame(border_rows)
print("\nBorder px (min / median / max over FOVs):")
for side in ["upper", "lower", "left", "right"]:
    print(f"  {side:<5}: {borders[side].min()} / "
          f"{borders[side].median():.0f} / {borders[side].max()}")
print(f"\nSpots kept: {keep.sum():,} / {len(df):,} ({100 * keep.mean():.1f}%)")

df[keep].to_csv(OUT_CSV, index=False)
borders.to_csv(BORDER_CSV, index=False)
print(f"Saved: {OUT_CSV}\nSaved: {BORDER_CSV}")