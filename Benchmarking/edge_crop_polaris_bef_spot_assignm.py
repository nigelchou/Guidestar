"""
Edge exclusion for Polaris spots, using ONLY Polaris's own registration shifts
(logged by the Polaris notebook), with the BFT rule:
    border = max shift towards that edge over all bits + PADDING
    keep   row in [upper+1, YDIM-lower),  col in [left+1, XDIM-right)
"""
import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

# ---- Main folder = the repository folder, which holds Data/ (layout described in Data/README.md). ----
# Run this from the folder the file is in, or set the environment variable GUIDESTAR_MAIN_DIR to the repository path.
main_dir = os.environ.get('GUIDESTAR_MAIN_DIR', '..')


# LM50 mBrain
# POL_DIR    = os.path.join(main_dir, "Data/20250728_Lm50GS_mbrain_3cT3/output/Relaxed Bernoulli_thresh0.85_GS_chrco_img_regi_no_butter2d")
# SHIFT_LOG  = os.path.join(POL_DIR, "shift_registration_polaris_splitfish_method_lm50gs_nobutter2d.txt")
# OUT_DIR    = os.path.join(POL_DIR, "polaris_LM50GS_result_edgecrop")
# BORDER_CSV = os.path.join(OUT_DIR, "polaris_edgecrop_borders.csv")

# LM39 Tissue
POL_DIR    = os.path.join(main_dir, "Data/benchmarking/polaris/liver/Relaxed Bernoulli_thresh0.85_GS_chrco_img_regi_final")
SHIFT_LOG  = os.path.join(os.path.dirname(POL_DIR), "shift_registration_polaris_final.txt")
OUT_DIR    = os.path.join(POL_DIR, "polaris_LM39tissue_result_edgecrop")
BORDER_CSV = os.path.join(OUT_DIR, "polaris_edgecrop_borders.csv")

# LM39 Cell
# POL_DIR    = os.path.join(main_dir, "Data/benchmarking/polaris/cell/Relaxed Bernoulli_thresh0.85_GS_chrco_img_regi_final")
# SHIFT_LOG  = os.path.join(os.path.dirname(POL_DIR), "shift_registration_polaris_GS.txt")
# OUT_DIR    = os.path.join(POL_DIR, "polaris_LM39cell_result_edgecrop")
# BORDER_CSV = os.path.join(OUT_DIR, "polaris_edgecrop_borders.csv")

YDIM, XDIM = 2048, 2048   # Polaris notebook builds 2048 x 2048 stacks
PADDING    = 30           # same safety margin as BFT ImageData.border_padding
ROW_COL    = "x"          # Polaris 'x' = image row
COL_COL    = "y"          # Polaris 'y' = image column

LINE_RE = re.compile(
    r"Bit (\d+) in fov: (\S+) shift merfish by ([-\d.]+), ([-\d.]+)")


def read_shift_log(path):
    """fov -> list of (dy, dx) for every logged bit, across all runs in the log."""
    shifts = defaultdict(list)
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                shifts[m.group(2)].append((float(m.group(3)), float(m.group(4))))
    return shifts


def borders_from_shifts(shift_list, padding=PADDING):
    """(upper, lower, left, right) exactly like BFT, reference bit included as (0, 0).
    Shifts were applied with fourier_shift, so +dy wraps into the top rows and
    -dy wraps into the bottom rows (same convention as BFT)."""
    s = np.vstack([(0.0, 0.0)] + shift_list)
    upper = int(np.ceil(max(0.0, s[:, 0].max()))) + padding
    lower = int(np.ceil(max(0.0, -s[:, 0].min()))) + padding
    left = int(np.ceil(max(0.0, s[:, 1].max()))) + padding
    right = int(np.ceil(max(0.0, -s[:, 1].min()))) + padding
    return upper, lower, left, right


os.makedirs(OUT_DIR, exist_ok=True)
shift_log = read_shift_log(SHIFT_LOG)
border_rows = []
n_in_total = n_kept_total = 0

for f in sorted(glob.glob(os.path.join(POL_DIR, "*_Polaris_prediction.csv"))):
    fov = os.path.basename(f).replace("_Polaris_prediction.csv", "")
    if fov not in shift_log:
        print(f"{fov}: no shifts in log -> skipped (not written)")
        continue

    up, lo, le, ri = borders_from_shifts(shift_log[fov])

    df = pd.read_csv(f, index_col=0)
    r = np.round(df[ROW_COL].to_numpy()).astype(int)
    c = np.round(df[COL_COL].to_numpy()).astype(int)
    m = ((r >= up + 1) & (r < YDIM - lo) &
         (c >= le + 1) & (c < XDIM - ri))

    df[m].to_csv(os.path.join(OUT_DIR, os.path.basename(f)))

    n_in_total += len(df)
    n_kept_total += int(m.sum())
    border_rows.append(dict(fov=fov, upper=up, lower=lo, left=le, right=ri,
                            n_log_entries=len(shift_log[fov]),
                            n_in=len(df), n_kept=int(m.sum())))
    print(f"{fov}: borders U{up} D{lo} L{le} R{ri}  kept {m.sum():,}/{len(df):,}")

borders = pd.DataFrame(border_rows)
borders.to_csv(BORDER_CSV, index=False)

print("\nBorder px (min / median / max over FOVs):")
for side in ["upper", "lower", "left", "right"]:
    print(f"  {side:<5}: {borders[side].min()} / "
          f"{borders[side].median():.0f} / {borders[side].max()}")
print(f"\nSpots kept: {n_kept_total:,} / {n_in_total:,} "
      f"({100 * n_kept_total / n_in_total:.1f}%)")
print(f"Cropped CSVs in: {OUT_DIR}")