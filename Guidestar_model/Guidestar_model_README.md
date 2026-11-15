# Training and Optimisation Pipeline

This folder contains the training, optimisation, and evaluation workflow of the Guidestar classifier, which separates true from false MERFISH RNA calls.

## Pipeline Overview

The complete pipeline should be run in the following order:

### 1. Decode Data

Run the decoding pipeline (`MainScript_MERFISH_cell.py`, `MainScript_MERFISH_tissue.py` or `MainScript_MERFISH_LM50GS_mbrain_analysis.py`).

### 2. `Colocalisation_Main_v1` (brain: `Colocalisation_Main_v1_LM50GS_0728mBrain_set1`)

Compute colocalisation features and generate the feature set used for downstream model training.

### 3. `NumberOfGuideGenes_cell` / `NumberOfGuideGenes_liver` / `NumberOfGuideGenes_LM50GS_mbrain`

Determine the optimal number of guide genes and prepare guide-gene-based features.

### 4. `ImbalanceSamplingMethod_cell` / `ImbalanceSamplingMethod_liver` / `ImbalanceSamplingMethod_LM50GS`

Apply class imbalance correction and sampling strategies to improve training performance.

### 5. `FeatureSelection` (brain: `FeatureSelection_LM50GS`)

Perform feature selection to identify the most informative features for classification.

### 6. `HyperparameterOptimization` (brain: `HyperparameterOptimization_LM50GS`)

Run hyperparameter optimisation using manual cross-validation.

This step evaluates multiple parameter combinations and records model performance metrics.

### 7. `TestSetResults`

Evaluate the final model on the held-out test set and generate the comparison figures used in the manuscript.

This script can also be used to apply the selected model to the full MERFISH dataset.

## Model Selection

For each optimisation run, select the model achieving the highest F1 score during cross-validation.

The chosen model should then be evaluated using `TestSetResults` and used for downstream analyses.
