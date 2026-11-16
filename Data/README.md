# Data folder

The raw images and intermediate outputs are too large for GitHub and are deposited separately (link to be added).
Download them into this folder using the layout below. Scripts that define `main_dir` then run without editing any path:
`main_dir` defaults to the repository folder, and can be pointed elsewhere with the environment variable `GUIDESTAR_MAIN_DIR`.

Only this README is tracked by git; everything else in `Data/` is ignored.

```
Data/
├── 100uM/                                   # AML-12 cell line (LM39, 25 FOVs): raw .dax/.inf images
│   ├── calibration/
│   ├── codebook/                            # codebook_data.tsv, fpkm_data.tsv
│   └── output/
│       ├── user_25FOVs_mag0_10_lc400_<DATE_TIME>/   # decoding output of MainScript_MERFISH_cell.py
│       └── colocalisation/                  # colocalisation csv files (e.g. colocaldist4_20240610/)
├── 60uM/                                    # liver tissue (LM39, 49 FOVs): raw .dax/.inf images
│   ├── calibrationDory/
│   ├── codebook/
│   └── output/
│       ├── user_49FOVs_mag0_10_lc400_<DATE_TIME>/   # decoding output of MainScript_MERFISH_tissue.py
│       └── colocalisation/                  # colocalisation csv files (e.g. colocaldist4_20240723_try2/)
├── 20250728_Lm50GS_mbrain_3cT3/             # mouse brain (LM50GS, 86 FOVs)
│   ├── S10/hybNN_1/                         # raw .ome.tif images, one folder per hybridisation round
│   ├── calibrationTriton/
│   ├── codebook/                            # codebook_data.tsv, fpkm_data.txt
│   ├── DAPI_1/, DAPI_single_frame/, Fused_DAPI.tif
│   ├── cellpose_training/                   # Cellpose masks
│   ├── Final_cellpose_assign_10px_dilation/ # spot-to-cell assignment and cell typing output
│   └── output/
├── benchmarking/
│   ├── polaris/cell/, polaris/liver/        # Polaris output for the cell line and liver
│   ├── starfish/cell/, starfish/liver/, starfish/brain/   # starfish input (experiment.json) and starfish_result/
│   └── merlin/                              # MERlin analysis folders
└── model_optimisation/<DATE>/               # output of the Guidestar_model optimisation scripts
```

Notes

* The scripts for the main figures (`MainScript_MERFISH_cell.py`, `MainScript_MERFISH_tissue.py`, `Colocalisation_Main_v1.py`, the `Fig*` scripts) are unchanged from the first release: they either ask for the data folder or use the placeholders `/path_to_raw_data/` and `/path_to_output_folder/`.
* The config files `genebitdict_brain.yml`, `genebitdict_cell_benchmarking.yml` and `genebitdict_liver_benchmarking.yml` hold `raw_folder` and `merfish_outpath` relative to the repository folder. If your decoding output folder has a different date stamp, change it there.
* `LM50GS_post_cellpose_analysis_all_methods_10px.R` is run from the repository folder and expects the Allen reference files (`expression_matrix.hdf5`, `metadata.csv`) there.
