"""
Assign spot/transcript predictions (GS RF, BFT, or Polaris) to Cellpose-derived
DAPI segmentation masks, optionally dilating cell boundaries via a Euclidean
distance transform (EDT) so that nearby pixels are claimed by the closest cell.

Spot rows do not carry hyb/channel; a hardcoded gene -> "{channel}_{hyb}" dict
tags each spot, and a channel-name -> stack-slice-index map resolves which
slice of the 3-channel ome.tif to correlate. Computed shifts are cached to disk
so reruns do not recompute them.

Originates from EDT_spot_assignment_2.py.

Typical use:
    from edt_spot_assignment_image_reg import (
        EDTSpotAssigner, SpotConfig, CONFIGS,
    )

    cfg = CONFIGS["GS"]
    result = EDTSpotAssigner(cfg).run()
    result.summary()

30 Sep 2026: updated _dilated_labels to run multiple methods quickly using shared dilated masks
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import tifffile
from PIL import Image
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.measure import regionprops
import sys
import datetime

class Tee:
    """Write to the console and to a log file at the same time."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
            s.flush()          # so the log is up to date even if the run crashes

    def flush(self):
        for s in self.streams:
            s.flush()


def start_log(log_path: str | Path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")   # "a" = append, "w" = overwrite
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)          # warnings + tracebacks too
    print(f"\n######## run started {datetime.datetime.now():%Y-%m-%d %H:%M:%S} ########")
    return log_file

# Your own subpixel translation routine.
from Guidestar_colocalisation.ColocalisationPipelineFunctions_v3 import register_translation

# ---- Main folder = the repository folder, which holds Data/ (layout described in Data/README.md). ----
# Run this from the folder the file is in, or set the environment variable GUIDESTAR_MAIN_DIR to the repository path.
main_dir = os.environ.get('GUIDESTAR_MAIN_DIR', '.')


Image.MAX_IMAGE_PIXELS = 1_000_000_000

# Imaging constant from the microscope metadata: 188 um across 2048 px.
PIXEL_SIZE_UM = 188 / 2048

# Spot metadata columns to carry through to per-cell aggregation, if present.
# Mean is taken for continuous quality metrics; count of spots is always added.
SPOT_METADATA_COLS = ("size", "misid", "bft_score", "probability")


# --------------------------------------------------------------------------- #
# EDT boundary expansion
# --------------------------------------------------------------------------- #
def expand_image(img: np.ndarray, pad: int, dtype=bool) -> np.ndarray:
    """Return ``img`` surrounded by ``pad`` pixels of zero-padding."""
    expanded = np.zeros(np.asarray(img.shape) + 2 * pad, dtype=dtype)
    expanded[pad:-pad, pad:-pad] = img
    return expanded


def expand_boundaries(
    label_image: np.ndarray,
    pad: int = 30,
    distance_upper_limit: float = np.inf,
    edt_on_inside: bool = False,
    store_crops: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, Optional[dict]]:
    """
    Expand each cell's boundary up to ``pad`` pixels from its edge.

    Pixels within ``pad`` of more than one cell are assigned to the nearest one
    (smallest Euclidean distance).
    """
    padded_shape = np.asarray(label_image.shape) + 2 * pad

    # Smallest distance to any cell seen so far, and the winning label.
    min_distance = np.full(padded_shape, distance_upper_limit, dtype=np.float64)
    labels = np.zeros(padded_shape, dtype=np.uint32)

    if verbose:
        print(f"Initialized arrays of shape {tuple(padded_shape)}")

    region_dict: Optional[dict] = {} if store_crops else None
    cell_rows = []

    for region_num, region in enumerate(regionprops(label_image)):
        minr, minc, maxr, maxc = region.bbox
        y0, x0 = region.centroid

        # The padded crop spans the original bbox plus 2*pad in each dimension.
        maxr += 2 * pad
        maxc += 2 * pad

        cell_label = int(label_image[region.coords[0, 0], region.coords[0, 1]])
        cell_rows.append({"Y": y0, "X": x0, "Cell Label": cell_label})

        crop = binary_fill_holes(region.image)
        expanded_crop = expand_image(crop, pad, dtype=bool)

        # Distance from each outside pixel to the cell.
        edt_crop = ndimage.distance_transform_edt(~expanded_crop)
        if edt_on_inside:  # subtract inside distance so overlaps resolve correctly
            edt_crop -= ndimage.distance_transform_edt(expanded_crop)
        edt_crop[edt_crop > pad] = distance_upper_limit

        if verbose:
            print(
                f"region {region_num + 1}: bbox={(minr, minc, maxr, maxc)}, "
                f"crop shape={expanded_crop.shape}"
            )

        # Claim pixels that are closer to this cell than to any seen so far.
        dist_view = min_distance[minr:maxr, minc:maxc]
        label_view = labels[minr:maxr, minc:maxc]
        update_mask = edt_crop < dist_view
        dist_view[update_mask] = edt_crop[update_mask]
        label_view[update_mask] = cell_label

        if store_crops:
            region_dict[region_num] = ((minr, minc, maxr, maxc), edt_crop)

    labels = labels[pad:-pad, pad:-pad]
    cell_location = pd.DataFrame(cell_rows, columns=["Y", "X", "Cell Label"])
    return labels, cell_location, min_distance, region_dict


# --------------------------------------------------------------------------- #
# Registration (spot round -> DAPI frame), per (hyb, channel, fov)
# --------------------------------------------------------------------------- #
def read_raw_img(path: str | Path, channel: Optional[int] = None) -> np.ndarray:
    """
    Load an image as a 2-D float array. If ``channel`` is given, take that
    channel slice from a multi-channel stack (assumes channel is the first
    axis, i.e. shape (C, Y, X)). Otherwise return the array as-is.
    """
    arr = tifffile.imread(path)
    if channel is not None:
        arr = arr[channel, ...]
    return arr.astype(np.float64)

def load_codebook_map(
    gene_to_code: dict[str, str],
    channel_name_to_index: dict[str, int],
) -> dict[str, tuple[int, int]]:
    """
    Build a gene -> (hyb, channel_index) mapping from a hardcoded codebook dict
    of the form ``{"Hcn1": "Cy7_06", ...}`` (i.e. "{channel_name}_{hyb}").

    ``channel_name_to_index`` resolves each channel name (e.g. "Cy7") to the
    integer slice index within the 3-channel ome.tif stack.
    """
    out: dict[str, tuple[int, int]] = {}
    for gene, code in gene_to_code.items():
        chan_name, _, hyb_str = code.rpartition("_")
        if not chan_name or not hyb_str.isdigit():
            raise ValueError(f"Cannot parse codebook entry {gene!r}: {code!r}")
        if chan_name not in channel_name_to_index:
            raise ValueError(
                f"Channel {chan_name!r} (gene {gene!r}) not in "
                f"channel_name_to_index map {sorted(channel_name_to_index)}"
            )
        out[str(gene)] = (int(hyb_str), channel_name_to_index[chan_name])
    return out


def load_fov_tracker(path: str | Path) -> tuple[dict[int, str], set[int]]:
    """
    Parse starfish's fov_tracker.txt (a printed defaultdict, one
    ``'000_004': 4,`` entry per line; a trailing ``!`` marks an excluded FOV).

    Returns (index -> "AAA_BBB" map, set of indices flagged with '!').
    """
    import re
    pattern = re.compile(r"'(\d{3}_\d{3})'\s*:\s*(\d+)\s*,?\s*(!)?")
    idx_to_fov: dict[int, str] = {}
    excluded: set[int] = set()
    with open(path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            fov, idx, bang = m.group(1), int(m.group(2)), m.group(3)
            idx_to_fov[idx] = fov
            if bang:
                excluded.add(idx)
    if not idx_to_fov:
        raise ValueError(f"No FOV entries parsed from {path}")
    return idx_to_fov, excluded


# Hardcoded codebook: gene -> "{channel_name}_{hyb}".
GENE_TO_CODE: dict[str, str] = {
    "Hcn1": "Cy7_06", "Grik3": "Cy7_07", "Itpr1": "Cy7_08", "Kcna2": "Cy5_07",
    "Cacna1a": "Alexa594_13", "Foxo1": "Alexa594_06", "Scn3a": "Alexa594_07",
    "Slc1a2": "Cy5_10", "Cdk6": "Cy5_12", "Cacna1i": "Alexa594_09",
    "Cacna1e": "Alexa594_11",
}

# Channel name -> slice index within the 3-channel ome.tif stack. ADJUST to
# match the actual channel order in your image files.
CHANNEL_NAME_TO_INDEX: dict[str, int] = {
    "Alexa594": 0,
    "Cy5": 1,
    "Cy7": 2,
}

# --------------------------------------------------------------------------- #
# Assignment
# --------------------------------------------------------------------------- #
@dataclass
class AssignmentResult:
    genes: pd.DataFrame
    cells: pd.DataFrame
    min_distance: Optional[np.ndarray] = None
    region_dict: Optional[dict] = None

    def summary(self) -> None:
        n_drop = int(self.genes["cell"].isna().sum())
        n_keep = int(self.genes["cell"].notna().sum())
        n_cells = self.genes.loc[self.genes["cell"].notna(), "cell"].nunique()
        print(f"Dropped transcripts:  {n_drop}")
        print(f"Assigned transcripts: {n_keep}")
        print(f"Valid cells:          {n_cells}")


class EDTSpotAssigner:
    """Assigns spots to (optionally dilated) cell masks for one config."""

    def __init__(self, cfg: SpotConfig) -> None:
        self.cfg = cfg

    # -- spot loading -------------------------------------------------------- #
    def _load_spots(self) -> pd.DataFrame:
        cfg = self.cfg
        if cfg.name == "polaris":
            return self._load_polaris_spots()
        if cfg.name == "starfish":
            return self._load_starfish_spots()

        # bft / bft_peaks / GS — unchanged
        spots = pd.read_csv(cfg.csv_source)
        if cfg.name == "bft":
            spots = spots[spots["bft_result"] == 1]
        elif cfg.name == "bft_peaks":
            spots = spots[spots["bfs"] < 1]
        else:  # GS
            spots = spots[spots["pred"] == 1]

        return spots

    def _load_polaris_spots(self) -> pd.DataFrame:
        """Concatenate per-FOV Polaris predictions and keep probability > threshold."""
        src = Path(self.cfg.csv_source)
        files = sorted(src.glob(POLARIS_FILE_GLOB))
        if not files:
            raise FileNotFoundError(f"No {POLARIS_FILE_GLOB} files found in {src}")

        frames = []
        for file in files:
            df = pd.read_csv(file, index_col=0)
            df["fov"] = "_".join(file.name.split("_")[0:2])  # "000_004"
            frames.append(df)
        spots = pd.concat(frames, ignore_index=True)

        # Match the GS/BFT schema: 'gene' column, and no Polaris 'cell_id'
        # that could be mistaken for the assigned cell.
        spots = spots.rename(columns={"predicted_name": "gene"})
        spots = spots.drop(columns=["cell_id"], errors="ignore")

        n_total = len(spots)
        spots = spots[spots["probability"] >= POLARIS_PROB_THRESHOLD]
        print(f"Polaris: kept {len(spots)}/{n_total} spots with "
              f"probability > {POLARIS_PROB_THRESHOLD} across {len(files)} FOVs")
        return spots

    def _load_starfish_spots(self) -> pd.DataFrame:
        """Load the multi-FOV starfish decoded table and keep passing spots."""
        spots = pd.read_csv(self.cfg.csv_source)

        # "fov_056" -> 56 -> "005_006", looked up in starfish's fov_tracker.txt
        idx_to_fov, excluded = load_fov_tracker(_STARFISH_FOV_TRACKER)
        fov_idx = spots["fov"].str.extract(r"(\d+)$", expand=False).astype(int)

        unknown = sorted(set(fov_idx) - set(idx_to_fov))
        if unknown:
            raise ValueError(f"starfish FOV indices not in fov_tracker: {unknown}")
        flagged = sorted(set(fov_idx) & excluded)
        if flagged:
            print(f"  [warn] starfish CSV contains FOVs marked '!' in tracker: "
                  f"{[idx_to_fov[i] for i in flagged]}")

        spots["fov_starfish"] = spots["fov"]
        spots["fov"] = fov_idx.map(idx_to_fov)

        # starfish: x = column, y = row. Swap to the GS convention used by
        # _assign_fov (x = row, y = column).
        spots = spots.rename(columns={"x": "y", "y": "x", "target": "gene"})

        n_total = len(spots)
        keep = spots["passes_thresholds"].astype(str).str.lower().eq("true")
        spots = spots[keep & spots["gene"].notna()]
        print(f"starfish: kept {len(spots)}/{n_total} spots passing thresholds "
              f"across {spots['fov'].nunique()} FOVs")
        return spots

    # -- mask loading / dilation -------------------------------------------- #
    @staticmethod
    def _read_mask(path: Path) -> np.ndarray:
        return np.array(Image.open(path).convert("L"))

    @staticmethod
    def _fov_from_filename(path: Path) -> str:
        parts = path.name.split("_")  # DAPI_000_004_cp_masks.png
        return f"{parts[1]}_{parts[2]}"

    def _dapi_path_for(self, fov: str) -> Path:
        return Path(self.cfg.dapi_image_dir) / self.cfg.dapi_image_template.format(fov=fov)

    def _dilated_labels(self, img_array: np.ndarray, fov: str) -> np.ndarray:
        cfg = self.cfg
        cache_dir = Path(cfg.save_path) / "edt_cache"  # shared by all methods
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cache_dir / f"{fov}_labels_{cfg.dilated_px}px.tiff"
        if cache.is_file():
            return tifffile.imread(cache)

        print(f"Running EDT with {cfg.dilated_px}px dilation for FOV {fov}...")
        labels, cell_location, _, _ = expand_boundaries(
            img_array, pad=cfg.dilated_px, edt_on_inside=True,
            store_crops=False, verbose=False,
        )
        cell_location.to_csv(cache_dir / f"{fov}_cell_location.csv")
        tifffile.imwrite(cache, labels)
        return labels

    @staticmethod
    def _drop_tiny_labels(img_array: np.ndarray, min_size: int = 2) -> np.ndarray:
        """Zero out labels with fewer than ``min_size`` pixels."""
        counts = np.bincount(img_array.ravel())
        tiny = np.where(counts < min_size)[0]
        tiny = tiny[tiny != 0]
        if tiny.size:
            img_array[np.isin(img_array, tiny)] = 0
        return img_array

    def _assign_fov(
            self, gene: pd.DataFrame, img_array: np.ndarray, fov: str, label_offset: int
    ) -> pd.DataFrame:
        """Assign each spot to the cell label under its (row=y, col=x) pixel."""
        gene = gene.reset_index(drop=True).copy()
        gene["cell"] = pd.NA

        h, w = img_array.shape
        xs = gene["y"].to_numpy(dtype=np.float64) # x and y are flipped
        ys = gene["x"].to_numpy(dtype=np.float64)
        xi = np.round(xs).astype(np.int64)
        yi = np.round(ys).astype(np.int64)

        # vectorised lookup instead of per-spot loop
        in_bounds = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        lab = np.zeros(len(gene), dtype=np.int64)
        lab[in_bounds] = img_array[yi[in_bounds], xi[in_bounds]]
        gene["cell"] = pd.Series(np.where(lab > 0, lab + label_offset, np.nan)).astype("Int64")

        n_oob = int((~in_bounds).sum())
        if n_oob:
            print(f"  [info] fov {fov}: {n_oob} spots outside frame")
        return gene

    @staticmethod
    def _aggregate_metadata(genes: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize spot metadata per assigned cell: transcript count plus the
        mean of any available quality columns (spot size, misid, bft_score,
        probability). Returned frame is keyed on cell_id.
        """
        assigned = genes[genes["cell"].notna()].copy()
        if assigned.empty:
            return pd.DataFrame(columns=["cell_id", "n_transcripts"])

        agg: dict[str, tuple[str, str]] = {"n_transcripts": ("cell", "size")}
        for col in SPOT_METADATA_COLS:
            if col in assigned.columns:
                agg[f"mean_{col}"] = (col, "mean")

        out = assigned.groupby("cell").agg(**agg).reset_index()
        return out.rename(columns={"cell": "cell_id"})

    # -- main routine -------------------------------------------------------- #
    def run(self) -> AssignmentResult:
        cfg = self.cfg
        os.makedirs(cfg.save_path, exist_ok=True)
        os.makedirs(cfg.output_dir, exist_ok=True)

        spots = self._load_spots()
        fov_set = set(spots["fov"].unique())

        all_genes: list[pd.DataFrame] = []
        cell_records: list[dict] = []
        label_offset = 0  # makes cell labels globally unique across FOVs
        min_distance = None
        start = time.time()

        for mask_file in cfg.mask_dir.glob("*.png"):
            fov = self._fov_from_filename(mask_file)
            if fov not in fov_set:
                continue

            img_array = self._read_mask(mask_file)
            if cfg.dilation:
                img_array = self._dilated_labels(img_array, fov)
                print(f"EDT cumulative time: {time.time() - start:.1f}s")

            img_array = self._drop_tiny_labels(img_array)

            gene = self._assign_fov(
                spots[spots["fov"] == fov], img_array, fov, label_offset
            )
            all_genes.append(gene)

            for region in regionprops(img_array):
                cell_records.append({
                    "cell_id": region.label + label_offset,
                    "cell_local_id": region.label,
                    "fov": fov,
                    "centroid_x": region.centroid[0],
                    "centroid_y": region.centroid[1],
                    "size_px": region.area,
                })

            label_offset += int(img_array.max())

        all_gene_df = (
            pd.concat(all_genes, ignore_index=True) if all_genes else pd.DataFrame()
        )
        cell_df = pd.DataFrame(cell_records)

        # Attach per-cell transcript counts + metadata summaries.
        if not cell_df.empty:
            meta = self._aggregate_metadata(all_gene_df)
            cell_df = cell_df.merge(meta, on="cell_id", how="left")
            cell_df["n_transcripts"] = cell_df["n_transcripts"].fillna(0).astype(int)
            assert cell_df["cell_id"].is_unique, "cell_id values are not unique"

        print("Assigned spots & computed cell sizes.")
        self._write_config()
        self._write_outputs(all_gene_df, cell_df)
        return AssignmentResult(all_gene_df, cell_df, min_distance)

    # -- output -------------------------------------------------------------- #
    def _write_config(self) -> None:
        cfg = self.cfg
        txt_path = os.path.join(cfg.output_dir, f"{cfg.name}_SpotConfig.txt")
        with open(txt_path, "w") as f:
            for key, value in asdict(cfg).items():
                f.write(f"{key}: {value}\n")
        print(f"SpotConfig saved to {txt_path}")

    def _write_outputs(self, genes: pd.DataFrame, cells: pd.DataFrame) -> None:
        cfg = self.cfg
        if cfg.dilation:
            genes.to_csv(os.path.join(
                cfg.output_dir,
                f"EDT_spotsassig_{cfg.name}_allfovs_with_dilation_{cfg.dilated_px}.csv",
            ))
            cells.to_csv(os.path.join(
                cfg.output_dir,
                f"{cfg.name}_cell_location_with_dilation_{cfg.dilated_px}.csv",
            ))
        else:
            genes.to_csv(os.path.join(
                cfg.save_path, f"EDT_spotsassig_{cfg.name}_no_dila_allfovs.csv"))
            cells.to_csv(os.path.join(
                cfg.save_path, f"{cfg.name}_no_dila_cell_location.csv"))
        print("Done!")


# --------------------------------------------------------------------------- #
# Global-coordinate conversion
# --------------------------------------------------------------------------- #
def load_stage_tiles(metadata: dict) -> pd.DataFrame:
    """Extract per-FOV XYStage origins (in px) from MicroManager metadata."""
    tiles = []
    for pos in metadata["Summary"]["StagePositions"]:
        label = "_".join(pos.get("label", "unknown").split("_")[1:])
        subpos = [sp for sp in pos.get("subpositions", [])
                  if sp["stageName"] == "XYStage"]
        if subpos:
            tiles.append((label, subpos[0]["x"], subpos[0]["y"]))

    if not tiles:
        raise ValueError("No XYStage positions found in metadata.")

    tiles_df = pd.DataFrame(tiles, columns=["fov", "x_stage_um", "y_stage_um"])
    tiles_df["x_stage_um"] -= tiles_df["x_stage_um"].min()
    tiles_df["y_stage_um"] -= tiles_df["y_stage_um"].min()
    tiles_df["x_origin_px"] = tiles_df["x_stage_um"] / PIXEL_SIZE_UM
    tiles_df["y_origin_px"] = tiles_df["y_stage_um"] / PIXEL_SIZE_UM
    return tiles_df


def cell_global_coord_conversion(
    metadata: dict, spot_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert per-FOV centroids to a global pixel coordinate frame."""
    tiles_df = load_stage_tiles(metadata)
    merged = spot_df.merge(tiles_df, on="fov", how="left")

    # Per-tile axis flip; tile 000_000 sits at top-right of the stitched image.
    max_x_tile = tiles_df["x_origin_px"].max()
    merged["y_global_px"] = merged["y_origin_px"] + merged["centroid_x"]
    merged["x_global_px"] = max_x_tile - merged["x_origin_px"] + merged["centroid_y"]
    return merged, tiles_df


def load_metadata(image_folder: str, metadata_file: str) -> dict:
    with open(os.path.join(image_folder, metadata_file), "r") as f:
        return json.load(f)

# --------------------------------------------------------------------------- #
# USER SETTINGS  — edit this block, nothing else
# --------------------------------------------------------------------------- #
@dataclass
class SpotConfig:
    """Paths and parameters for one spot-assignment run."""

    name: str                # "bft", "GS", or "polaris"
    csv_source: str          # CSV file (bft/GS) or folder of per-FOV CSVs (polaris)
    save_path: str           # directory for top-level outputs
    mask_dir: Path           # directory of Cellpose mask PNGs
    output_dir: str          # directory for per-FOV dilated masks/plots
    dilation: bool = True
    dilated_px: int = 10
    # Registration: per (hyb, channel, fov) image shift + per-FOV spot pre-align.
    dapi_image_dir: Optional[str] = None     # folder of raw DAPI images per fov
    hyb_root_dir: Optional[str] = None       # parent folder containing hyb{N}/ dirs
    # Filename templates. {fov} -> "000_000", {hyb} -> hyb number.
    dapi_image_template: str = "DAPI_1_MMStack_1-Pos_{fov}.ome.tif"
    moving_subdir_template: str = "hyb{hyb}_1"
    moving_image_template: str = "hyb{hyb}_1_MMStack_1-Pos_{fov}.ome.tif"
    hyb_pad: int = 2                         # zero-pad hyb in paths (QC used "06")
    shift_sign: int = 1                      # +1 => corrected = detected + shift
    shift_cache_csv: Optional[str] = None    # where computed shifts are saved

    VALID_NAMES = ("bft", "bft_peaks", "GS", "polaris", "starfish")

    def __post_init__(self) -> None:
        if self.name not in self.VALID_NAMES:
            raise ValueError(
                f"Invalid name {self.name!r}. Expected one of {self.VALID_NAMES}."
            )
        self.mask_dir = Path(self.mask_dir)

def _make_config(name: str) -> SpotConfig:
    _save = f"{_BASE}/Final_cellpose_assign_{dilate_val}px_dilation/"

    # only the per-type differences live here
    if name == "polaris":
        csv_source = _POLARIS_DIR
    elif name == "starfish":
        csv_source = _STARFISH_CSV
    else:  # bft / bft_peaks / GS
        csv_source = _MERFISH_CSV

    return SpotConfig(
        name=name,
        csv_source=csv_source,
        save_path=_save,
        mask_dir=Path(f"{_BASE}/cellpose_training"),
        output_dir=f"{_save}{name}",
        dilation=True,
        dilated_px=dilate_val,
        dapi_image_dir=_DAPI_DIR,
        hyb_root_dir=_HYB_ROOT,
        shift_sign=1,
    )

_BASE = os.path.join(main_dir, "Data/20250728_Lm50GS_mbrain_3cT3")
# MERFISH; Guidestar loading
_MERFISH_CSV = os.path.join(_BASE, 'output/user_86FOVs_mag0_10_lc350_20250929_1705/final_ver/GS_model_result/merfish_full_data_GS_pred_output_p_crop_edge.csv') # GS output
# Polaris loading
_POLARIS_DIR = f"{_BASE}/output/Relaxed Bernoulli_thresh0.85_GS_chrco_img_regi_no_butter2d/polaris_LM50GS_result_edgecrop" #/polaris_LM50GS_result_edgecrop"
POLARIS_PROB_THRESHOLD = 0.95
POLARIS_FILE_GLOB = "*_Polaris_prediction.csv"   # e.g. 000_004_Polaris_prediction.csv
# starfish loading
_STARFISH_ROOT = os.path.join(main_dir, "Data/benchmarking/starfish/brain")
_STARFISH_CSV = f"{_STARFISH_ROOT}/starfish_result/lm50GS_brain_starfish_multifov.csv" # lm50GS_brain_starfish_multifov_edgecrop.csv" #lm50GS_brain_starfish_multifov.csv"
_STARFISH_FOV_TRACKER = f"{_STARFISH_ROOT}/LM50GS/fov_tracker.txt"
dilate_val = 10

type_to_config = ['GS', 'bft', 'starfish', 'polaris', 'bft_peaks'] #['bft_peaks' #'bft', 'GS'] #, 'polaris']

# DAPI is per-fov under DAPI_1/; hyb folders sit under _HYB_ROOT as hyb{N}_1/.
_DAPI_DIR = f"{_BASE}/DAPI_1"
_HYB_ROOT = f"{_BASE}/S10"                   # parent of the hyb{N}_1 folders

CONFIGS: dict[str, SpotConfig] = {
    name: _make_config(name) for name in type_to_config
}

# --------------------------------------------------------------------------- #
# QC plot
# --------------------------------------------------------------------------- #
import matplotlib
matplotlib.use("Agg")  # save to file, no blocking windows
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_fov_assignment(img_array, gene, fov, out_path=None, max_points=None):
    """
    Overlay spots on the cell mask for one FOV.
    Assigned spots (landed in a cell) are green, unassigned are red.

    img_array : 2-D label image for this FOV (same one passed to _assign_fov)
    gene       : the per-FOV spot DataFrame returned by _assign_fov
                 (must have 'x', 'y', 'cell' columns)
    """
    assigned = gene["cell"].notna()

    # Show cells as a faint binary background so spots stay visible.
    bg = (img_array > 0).astype(float)
    cmap = ListedColormap(["#000000", "#3a3a5a"])  # outside / inside cell

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(bg, cmap=cmap, interpolation="nearest")

    xa = gene.loc[assigned, "x"].to_numpy()
    ya = gene.loc[assigned, "y"].to_numpy()
    xu = gene.loc[~assigned, "x"].to_numpy()
    yu = gene.loc[~assigned, "y"].to_numpy()

    if max_points:  # subsample if there are a lot of spots
        for xs, ys in ((xa, ya), (xu, yu)):
            pass  # (keep simple; subsample below if needed)

    ax.scatter(yu, xu, s=6, c="#ff4d4d", label=f"unassigned ({len(xu)})",
               alpha=0.6, linewidths=0)
    ax.scatter(ya, xa, s=6, c="#4dff88", label=f"assigned ({len(xa)})",
               alpha=0.8, linewidths=0)

    ax.set_title(f"FOV {fov}: spot assignment")
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")
    ax.legend(loc="upper right", framealpha=0.8)
    ax.set_xlim(0, img_array.shape[1])
    ax.set_ylim(img_array.shape[0], 0)  # image convention: y increases downward

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    log_file = start_log(
        f"{_BASE}/Final_cellpose_assign_{dilate_val}px_dilation/"
        f"spot_assignment_log_{datetime.datetime.now():%Y%m%d_%H%M}.txt"
    )
    try:
        metadata = load_metadata(
            image_folder=rf"{_BASE}\DAPI_1".replace("/", "\\"),
            metadata_file="DAPI_1_MMStack_1-Pos_000_000_metadata.txt",
        )
        failed = []
        for name, cfg in CONFIGS.items():
            print(f"\n===== {name} =====")
            t0 = time.time()
            try:
                result = EDTSpotAssigner(cfg).run()
                result.summary()
                result.genes.to_csv(
                    cfg.save_path + f"EDT_spotsassig_{cfg.name}_allfovs_global_coord.csv")
                cells_global, _ = cell_global_coord_conversion(metadata, result.cells)
                cells_global.to_csv(cfg.save_path + f"EDT_{cfg.name}_cell_location.csv")
                print(f"{name}: done in {(time.time() - t0) / 60:.1f} min")
            except Exception:
                traceback.print_exc()
                failed.append(name)

        qc_fov = "003_001"
        for name in type_to_config:  # QC plot for every method that succeeded
            if name in failed:
                continue
            try:
                cfg = CONFIGS[name]
                df = pd.read_csv(cfg.save_path + f"EDT_spotsassig_{name}_allfovs_global_coord.csv")
                img_array = tifffile.imread(
                    Path(cfg.save_path) / "edt_cache" / f"{qc_fov}_labels_{cfg.dilated_px}px.tiff")
                gene_fov = df[df["fov"].astype(str) == qc_fov]
                print(f"QC {name}: FOV {qc_fov}: {len(gene_fov)} spots, mask shape {img_array.shape}")
                plot_fov_assignment(img_array, gene_fov, qc_fov,
                                    out_path=Path(cfg.output_dir) / f"QC_{name}_{qc_fov}.png")
            except Exception:
                traceback.print_exc()
        print("\nFailed:", failed or "none")
    finally:
        print(f"######## run finished {datetime.datetime.now():%Y-%m-%d %H:%M:%S} ########")
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        log_file.close()

if __name__ == "__main__":
    main()