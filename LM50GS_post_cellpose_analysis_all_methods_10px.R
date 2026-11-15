## LM50GS post-cellpose analysis — all methods in one script
##
## Replaces:
##   LM50GS_post_cellpose_analysis_GS_10px_final.R
##   LM50GS_post_cellpose_analysis_bft_10px_final.R
##   LM50GS_post_cellpose_analysis_polaris_starfish_10px_final.R
##   LM50GS_post_cellpose_analysis_BFT_final_10px_varying_bft.R   (-> run_bft_sweep)
##
## Every method goes through the SAME pipeline (QC -> area normalisation ->
## SingleR against the HIP 19-subclass reference -> marker purity -> per-cell
## annotation table), so scores are directly comparable. Output folders and
## file names are unchanged, so downstream notebooks keep working.

suppressPackageStartupMessages({
  library(dplyr); library(tidyverse); library(patchwork)
  library(Matrix); library(data.table)
  library(Seurat)
  library(SingleCellExperiment); library(scater); library(SingleR)
  library(hdf5r); library(HDF5Array)
})

############################################################################
### USER SETTINGS
############################################################################
d_px     <- 10
base_dir <- "Data/20250728_Lm50GS_mbrain_3cT3"
load_dir <- file.path(base_dir, paste0("Final_cellpose_assign_", d_px, "px_dilation"))

# methods to run (must match cfg.name in the python spot assignment)
methods_to_run <- c("GS", "bft") #, "polaris", "starfish")

# output sub-folder per method (kept identical to the old scripts)
out_subdir <- c(GS       = "gs_seurat_analysis",
                bft      = "bft_seurat_analysis_19subclass") # ,
                # polaris  = "polaris_seurat_analysis_19subclass",
                # starfish = "starfish_seurat_analysis_19subclass")

# methods that also get PCA/UMAP clustering, silhouette, kNN consistency,
# Moran's I and per-cell-type UMAP/spatial plots (previously bft script only)
extra_analysis <- c("bft")

# BFT threshold sweep on the bft_peaks spot assignment (varying_bft script)
run_bft_sweep   <- FALSE
sweep_out_dir   <- file.path(load_dir, "vary_bft_seurat_analysis_19subclass", "misid")
sweep_misid_csv <- file.path(base_dir, "output/user_86FOVs_mag0_10_lc350_20250929_1705/final_ver/BFT_misid_brain_extended.csv")
sweep_n_points  <- 8      # thresholds evenly spaced in misid
sweep_bft_cap   <- Inf    # the old script computed a bft <= 0.709 filter but never
                          # used it; Inf reproduces the old behaviour, 0.709 applies it

# reference (Allen 10x) — cell-filtered object is cached after the first run
ref_h5        <- "expression_matrix.hdf5"
ref_meta      <- "metadata.csv"
ref_cache_rds <- file.path(load_dir, "sceZ_HIP_19subclass_cells.rds")
keep_list <- c("Astro", "Endo", "SMC-Peri", "Oligo", "VLMC", "Pvalb", "L6 CT CTX", "L6b CTX",
               "Micro-PVM", "L6 IT CTX", "CA2-IG-FC", "Sst", "Sst Chodl", "Sncg", "CA3", "CR",
               "CA1-ProS", "Lamp5", "DG")

# QC / processing
max_percent_blank <- 10
n_hvg             <- 115

############################################################################
### Reference
############################################################################
load_reference_cells <- function() {
  if (file.exists(ref_cache_rds)) return(readRDS(ref_cache_rds))
  message("Building reference (first run only)...")
  f <- hdf5r::H5File$new(ref_h5, mode = "r")
  genes <- f[["data/gene"]][]; cells <- f[["data/samples"]][]
  f$close_all()
  sce <- SingleCellExperiment(assays  = list(counts = t(HDF5Array(ref_h5, "data/counts"))),
                              rowData = DataFrame(gene = genes),
                              colData = DataFrame(cell = cells))
  colnames(sce) <- sce$cell
  meta <- read.csv(ref_meta)
  m <- match(colnames(sce), meta$sample_name)
  sce$subclass_label <- meta$subclass_label[m]
  sce$region_label   <- meta$region_label[m]
  sce <- sce[, !is.na(m)]
  sce <- sce[, sce$subclass_label %in% keep_list]
  sce <- sce[, sce$region_label %in% "HIP"]
  rownames(sce) <- rowData(sce)$gene
  saveRDS(sce, ref_cache_rds)   # HDF5-backed: small file, points to ref_h5
  sce
}

# gene subset + logNorm AFTER subsetting (as in all the old scripts)
reference_for <- function(sceZ_all, genes) {
  ref <- sceZ_all[rownames(sceZ_all) %in% genes, ]
  counts(ref) <- as(as.matrix(counts(ref)), "dgCMatrix")
  logNormCounts(ref)
}

############################################################################
### Shared helpers
############################################################################
spot_metrics <- function(spots) {
  data.frame(n_spots         = nrow(spots),
             n_cells_input   = length(unique(spots$cell)),
             n_genes_input   = length(unique(spots$gene)),
             blank_spot_rate = mean(grepl("^Blank-", spots$gene)) * 100,
             n_blank_spots   = sum(grepl("^Blank-", spots$gene)))
}

make_seurat <- function(spots, cell_location, project) {
  mat <- Matrix(table(spots$gene, spots$cell), sparse = TRUE)
  obj <- CreateSeuratObject(counts = mat, project = project, min.cells = 3, min.features = 5)
  rownames(cell_location) <- cell_location$cell_id   # AddMetaData matches by rowname
  obj <- AddMetaData(obj, cell_location)
  stopifnot(!anyNA(obj$size_px))                      # normalisation denominator
  obj[["percent.blank"]] <- PercentageFeatureSet(obj, pattern = "^Blank-")
  obj
}

# cell-level blank metrics, run AFTER QC filtering
blank_metrics <- function(obj) {
  cts <- GetAssayData(obj, layer = "counts")
  bg  <- grep("^Blank-", rownames(obj), value = TRUE)
  blank_cnt <- if (length(bg)) Matrix::colSums(cts[bg, , drop = FALSE]) else
    setNames(rep(0, ncol(obj)), colnames(obj))
  total_cnt <- Matrix::colSums(cts)
  real_cnt  <- total_cnt - blank_cnt
  row <- data.frame(
    n_cells         = ncol(obj),
    n_genes         = nrow(obj),
    n_blank_probes  = length(bg),
    median_counts   = median(obj$nCount_RNA),
    median_feats    = median(obj$nFeature_RNA),
    median_size_px  = median(obj$size_px),
    blank_pooled    = sum(blank_cnt) / sum(total_cnt) * 100,   # not mean-of-ratios
    blank_median    = median(obj$percent.blank),
    blank_mean      = mean(obj$percent.blank),
    blank_per_area  = median(blank_cnt / obj$size_px),
    real_per_area   = median(real_cnt  / obj$size_px)
  )
  # the old scripts used two different "snr_area" definitions; keep both, named
  row$snr_area_median <- row$real_per_area / (row$blank_per_area + 1e-9)  # GS/bft/polaris/starfish scripts
  row$snr_pooled      <- sum(real_cnt) / sum(blank_cnt)                   # sweep script
  list(row = row, blank_cnt = blank_cnt, total_cnt = total_cnt)
}

normalise_and_scale <- function(obj) {
  cts <- GetAssayData(obj, layer = "counts")
  obj <- SetAssayData(obj, layer = "data", new.data = log1p(t(t(cts) / obj$size_px)))
  obj <- FindVariableFeatures(obj, selection.method = "vst",
                              nfeatures = min(n_hvg, nrow(obj) - 1))  # panel shrinks at low bft
  ScaleData(obj, features = rownames(obj))
}

run_singler <- function(obj, ref) {
  sce <- as.SingleCellExperiment(DietSeurat(obj))
  SingleR(test = sce, ref = ref, labels = ref$subclass_label,
          assay.type.test = "logcounts", de.method = "wilcox")
}

assigned_score <- function(pred) {
  pred$scores[cbind(seq_len(nrow(pred)), match(pred$labels, colnames(pred$scores)))]
}

# Astro / Endo / OPC are mutually exclusive, so co-expression within one mask is
# bleed-through. Run BEFORE pruned cells are dropped (denominator = all QC cells).
compute_marker_purity <- function(obj, tag = NA) {
  X  <- GetAssayData(obj, layer = "data")
  gs <- function(g) intersect(g, rownames(X))
  sets <- list(astro = gs(c("Aqp4", "Slc1a2", "Slc1a3", "Kcnj10")),
               endo  = gs(c("Flt1", "Vcam1", "Ldlr")),
               opc   = gs(c("Pdgfra")))
  S <- sapply(sets, function(g)
    if (length(g)) Matrix::colMeans(X[g, , drop = FALSE]) else rep(0, ncol(X)))
  rownames(S) <- colnames(X)
  tot  <- rowSums(S)
  keep <- tot > 0
  pur  <- apply(S[keep, , drop = FALSE], 1, max) / tot[keep]
  lab  <- obj$singler
  med_in <- function(target, score) {
    i <- !is.na(lab) & lab == target
    if (sum(i)) median(S[i, score]) else NA_real_
  }
  list(
    per_cell = data.frame(cell_id = rownames(S), S, total = tot, stringsAsFactors = FALSE) %>%
      dplyr::mutate(purity = ifelse(total > 0, pmax(astro, endo, opc) / total, NA_real_)),
    summary = data.frame(
      tag            = tag,
      n_cells_scored = sum(keep),
      n_astro_genes  = length(sets$astro),
      n_endo_genes   = length(sets$endo),
      n_opc_genes    = length(sets$opc),
      median_purity  = median(pur),
      q25_purity     = unname(quantile(pur, .25)),
      frac_mixed_60  = mean(pur < 0.60),
      frac_mixed_80  = mean(pur < 0.80),
      endo_in_astro  = med_in("Astro", "endo"),
      astro_in_endo  = med_in("Endo",  "astro"),
      opc_in_astro   = med_in("Astro", "opc"),
      astro_in_opc   = med_in("Oligo", "astro"),
      stringsAsFactors = FALSE
    )
  )
}

safe_name <- function(x) gsub("/", "_", x)

############################################################################
### Extra analysis (previously bft script only)
############################################################################
run_clustering <- function(obj, out_dir) {
  obj <- RunPCA(obj, features = VariableFeatures(obj), verbose = FALSE)
  ggsave(file.path(out_dir, "pca_plt.png"), DimPlot(obj, reduction = "pca") + NoLegend(),
         width = 8, height = 6, dpi = 300, bg = "white")
  png(file.path(out_dir, "15pca_plt.png"), width = 2000, height = 3000, res = 300)
  DimHeatmap(obj, dims = 1:15, cells = 500, balanced = TRUE)
  dev.off()

  set.seed(0)
  obj <- FindNeighbors(obj, dims = 1:25, k.param = 25)
  obj <- FindClusters(obj, resolution = 0.8)
  obj <- RunUMAP(obj, dims = 1:25, n.neighbors = 25, seed.use = 0, min.dist = 0.15)
  ggsave(file.path(out_dir, "umap.png"), DimPlot(obj, reduction = "umap"),
         width = 8, height = 6, dpi = 300)

  top <- FindAllMarkers(obj, only.pos = TRUE) %>%
    group_by(cluster) %>% dplyr::filter(avg_log2FC > 1) %>% slice_head(n = 5) %>% ungroup()
  ggsave(file.path(out_dir, "heatmap1.png"),
         DoHeatmap(obj, features = top$gene) + NoLegend(), width = 8, height = 6, dpi = 300)
  avg <- AverageExpression(obj, return.seurat = TRUE)
  ggsave(file.path(out_dir, "aveheatmap.png"),
         DoHeatmap(avg, features = top$gene) + NoLegend(), width = 8, height = 6, dpi = 300)
  obj
}

run_label_extras <- function(srat, method, out_dir, top_markers) {
  library(cluster); library(FNN); library(spdep)
  labels <- srat$singler

  ggsave(file.path(out_dir, "singler_heatmap.png"),
         DoHeatmap(srat, features = top_markers$gene, size = 3, angle = 90) + NoLegend(),
         width = 8, height = 6, dpi = 300)

  ## silhouette + kNN label consistency in PCA space (now saved to CSV)
  pca_mat <- Embeddings(srat, reduction = "pca")[, 1:25]
  sil <- silhouette(as.numeric(factor(labels)), dist(pca_mat))
  sil_df <- data.frame(cell = rownames(pca_mat), label = labels, silhouette = sil[, "sil_width"])
  write.csv(sil_df %>% group_by(label) %>%
              summarize(mean_silhouette = mean(silhouette),
                        median_silhouette = median(silhouette), n = n()),
            file.path(out_dir, "silhouette_summary.csv"), row.names = FALSE)

  knn <- get.knn(pca_mat, k = 20)
  srat$knn_consistency <- sapply(seq_len(nrow(pca_mat)), function(i)
    mean(labels[knn$nn.index[i, ]] == labels[i]))
  write.csv(data.frame(label = labels, consistency = srat$knn_consistency) %>%
              group_by(label) %>%
              summarize(mean_consistency = mean(consistency),
                        median_consistency = median(consistency), n = n()),
            file.path(out_dir, "knn_consistency_summary.csv"), row.names = FALSE)

  ## Moran's I per cell type (now saved to CSV)
  coords <- as.matrix(srat@meta.data[, c("x_global_px", "y_global_px")])
  lw <- nb2listw(knn2nb(knearneigh(coords, k = 25)), style = "W")
  moran_df <- do.call(rbind, lapply(unique(labels), function(ct) {
    t <- moran.test(as.numeric(labels == ct), lw)
    data.frame(cell_type = ct, moran_I = t$estimate[["Moran I statistic"]], p_value = t$p.value)
  }))
  write.csv(moran_df, file.path(out_dir, "moran_I_per_celltype.csv"), row.names = FALSE)

  ## per-cell-type UMAP and spatial highlight plots
  colnames(coords) <- c("spatial_1", "spatial_2")
  srat[["spatial"]] <- CreateDimReducObject(embeddings = coords, key = "spatial_",
                                            assay = DefaultAssay(srat))
  umap_dir    <- file.path(out_dir, "singler_umap")
  spatial_dir <- file.path(out_dir, paste0(method, "_singler_spatial"))
  dir.create(umap_dir, showWarnings = FALSE); dir.create(spatial_dir, showWarnings = FALSE)
  for (ct in levels(Idents(srat))) {
    hl <- list(WhichCells(srat, idents = ct)); names(hl) <- ct
    p <- DimPlot(srat, cells.highlight = hl, cols.highlight = "#E41A1C", cols = "gray85") +
      ggtitle(ct)
    ggsave(file.path(umap_dir, paste0(safe_name(ct), "_umap_plot.png")), p, dpi = 300)
    p <- DimPlot(srat, reduction = "spatial", cells.highlight = hl,
                 cols.highlight = "#E41A1C", cols = "gray85") +
      scale_x_reverse() + scale_y_reverse() + coord_equal() + ggtitle(ct)
    ggsave(file.path(spatial_dir, paste0(safe_name(ct), "_spatial_plot.png")), p,
           width = 3000, height = 2800, units = "px")
  }
  invisible(srat)
}

############################################################################
### Per-method pipeline (GS / bft / polaris / starfish)
############################################################################
run_method <- function(method, sceZ_all) {
  message("==== ", method, " ====")
  out_dir <- file.path(load_dir, out_subdir[[method]])
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  extras <- method %in% extra_analysis

  spots <- read.csv(file.path(load_dir, sprintf("EDT_spotsassig_%s_allfovs_global_coord.csv", method)))
  cell_location <- read.csv(file.path(load_dir, sprintf("EDT_%s_cell_location.csv", method)))
  spots <- na.omit(spots)   # unassigned spots have cell = NA
  sm  <- spot_metrics(spots)
  obj <- make_seurat(spots, cell_location, paste0(method, "_pred"))
  rm(spots, cell_location)
  print(obj)

  ## QC
  ggsave(file.path(out_dir, "qc_plt.png"),
         VlnPlot(obj, features = c("nFeature_RNA", "nCount_RNA", "percent.blank"), ncol = 3),
         width = 8, height = 6, dpi = 300)
  ggsave(file.path(out_dir, "cor_plt.png"),
         FeatureScatter(obj, "nCount_RNA", "percent.blank") +
           FeatureScatter(obj, "nCount_RNA", "nFeature_RNA"),
         width = 10, height = 6, dpi = 300)
  obj <- subset(obj, subset = percent.blank < max_percent_blank)

  bm <- blank_metrics(obj)
  metrics <- cbind(method = method, d_px = d_px, sm, bm$row)
  metrics$snr_area <- metrics$snr_area_median          # old column name kept
  write.csv(metrics, file.path(out_dir, "cell_level_blank_metrics.csv"), row.names = FALSE)
  print(t(metrics))

  ## normalise by cell area, HVG, scale
  obj <- normalise_and_scale(obj)
  top10 <- head(VariableFeatures(obj), 10)
  ggsave(file.path(out_dir, "hvg_plt.png"),
         LabelPoints(plot = VariableFeaturePlot(obj), points = top10, repel = FALSE),
         width = 8, height = 6, dpi = 300, bg = "white")
  if (extras) obj <- run_clustering(obj, out_dir)

  ## SingleR (reference restricted to this method's genes, logNorm after subsetting)
  ref  <- reference_for(sceZ_all, rownames(obj))
  message("reference genes used: ", nrow(ref))
  pred <- run_singler(obj, ref)
  saveRDS(pred, file.path(out_dir, "singler_pred_tasic.rds"))
  print(table(pred$pruned.labels, useNA = "always"))
  obj$singler <- pred$pruned.labels

  purity <- compute_marker_purity(obj, tag = paste0(method, "_", d_px, "px"))
  write.csv(purity$summary,  file.path(out_dir, "marker_purity_summary.csv"),  row.names = FALSE)
  write.csv(purity$per_cell, file.path(out_dir, "marker_purity_per_cell.csv"), row.names = FALSE)

  srat <- SetIdent(obj, value = "singler")
  if (extras) ggsave(file.path(out_dir, "singler_umap.png"),
                     DimPlot(srat, label = TRUE, label.size = 3),
                     width = 10, height = 8, dpi = 300)
  srat <- subset(srat, idents = levels(srat))   # drop pruned (NA) cells

  top_markers <- FindAllMarkers(srat, only.pos = TRUE) %>%
    group_by(cluster) %>% dplyr::filter(avg_log2FC > 1) %>% slice_head(n = 5) %>% ungroup()
  avg_srat <- AverageExpression(srat, return.seurat = TRUE)
  ggsave(file.path(out_dir, "singler_avg_heatmap.png"),
         DoHeatmap(avg_srat, features = top_markers$gene) + NoLegend(),
         width = 8, height = 6, dpi = 300)
  if (extras) srat <- run_label_extras(srat, method, out_dir, top_markers)

  ## per-cell annotation table, joined by cell id
  score_df <- data.frame(cell_id    = rownames(pred),
                         label_raw  = pred$labels,
                         score      = assigned_score(pred),
                         delta_next = pred$delta.next,
                         is_pruned  = is.na(pred$pruned.labels),
                         stringsAsFactors = FALSE)
  coords_df <- data.frame(cell_id     = colnames(srat),
                          singler     = as.character(srat$singler),
                          x_global_px = srat$x_global_px,
                          y_global_px = srat$y_global_px,
                          stringsAsFactors = FALSE)
  final <- dplyr::inner_join(score_df, coords_df, by = "cell_id")
  stopifnot(!anyNA(final$x_global_px), all(final$label_raw == final$singler))
  write.csv(final, file.path(out_dir, "singler_cell_annotations.csv"), row.names = FALSE)

  final %>% group_by(singler) %>%
    summarise(n = n(), mean_score = mean(score), median_score = median(score),
              mean_delta = mean(delta_next), median_delta = median(delta_next)) %>%
    arrange(desc(n)) %>% print(n = Inf)

  invisible(list(obj = obj, srat = srat, pred = pred, final = final, metrics = metrics))
}

############################################################################
### BFT threshold sweep on bft_peaks (previously the varying_bft script)
############################################################################
run_threshold <- function(thr, spots_all, cell_location, ref) {
  set.seed(0)
  sp <- spots_all[spots_all$bfs <= thr, ]
  if (nrow(sp) == 0) return(NULL)
  tag <- sprintf("bfs%.3f", thr)
  message(sprintf("=== bfs <= %.3f : %d spots ===", thr, nrow(sp)))

  sm  <- spot_metrics(sp)
  obj <- make_seurat(sp, cell_location, tag)
  obj <- subset(obj, subset = percent.blank < max_percent_blank)
  if (ncol(obj) < 50) return(NULL)
  bm  <- blank_metrics(obj)
  obj <- normalise_and_scale(obj)

  pred <- run_singler(obj, ref)
  obj$singler <- pred$pruned.labels
  saveRDS(pred, file.path(sweep_out_dir, sprintf("pred_%s.rds", tag)))

  cells <- rownames(pred)
  score <- assigned_score(pred)
  annot <- data.frame(cell_id = cells, bfs_threshold = thr,
                      label_raw = pred$labels, singler = pred$pruned.labels,
                      score = score, delta_next = pred$delta.next,
                      is_pruned = is.na(pred$pruned.labels),
                      n_count = bm$total_cnt[cells], n_blank = bm$blank_cnt[cells],
                      percent_blank = obj$percent.blank[cells], size_px = obj$size_px[cells],
                      obj@meta.data[cells, c("x_global_px", "y_global_px")],
                      stringsAsFactors = FALSE)
  write.csv(annot, file.path(sweep_out_dir, sprintf("annot_%s.csv", tag)), row.names = FALSE)

  purity <- compute_marker_purity(obj, tag = tag)$summary
  purity <- cbind(bfs_threshold = thr, purity[, setdiff(names(purity), "tag")])
  write.csv(purity, file.path(sweep_out_dir, sprintf("purity_%s.csv", tag)), row.names = FALSE)

  r <- bm$row
  summary_row <- data.frame(
    bfs_threshold = thr, n_spots = sm$n_spots, n_cells_input = sm$n_cells_input,
    n_cells = r$n_cells, n_genes = r$n_genes, n_blank_probes = r$n_blank_probes,
    median_counts = r$median_counts, median_feats = r$median_feats,
    median_size_px = r$median_size_px, blank_spot_rate = sm$blank_spot_rate,
    blank_pooled = r$blank_pooled, blank_median = r$blank_median,
    blank_per_area = r$blank_per_area, real_per_area = r$real_per_area,
    snr_area = r$snr_pooled,                 # old sweep definition (pooled ratio)
    snr_area_median = r$snr_area_median,     # per-method definition, for comparison
    mean_blank = r$blank_mean,
    frac_pruned = mean(is.na(pred$pruned.labels)),
    mean_score = mean(score), mean_delta = mean(pred$delta.next, na.rm = TRUE),
    n_celltypes = length(unique(na.omit(pred$pruned.labels))))
  list(summary = summary_row, purity = purity)
}

select_thresholds <- function() {
  td <- fread(sweep_misid_csv)
  td <- td[td$bft <= sweep_bft_cap, ]
  targets <- seq(min(td$misid), max(td$misid), length.out = sweep_n_points)
  idx <- sapply(targets, function(t) which.min(abs(td$misid - t)))
  bft_at_05 <- td$bft[which.min(abs(td$misid - 0.05))]   # misid = 5 %
  sort(unique(c(td$bft[idx], bft_at_05)))
}

run_bft_sweep_all <- function(sceZ_all) {
  dir.create(sweep_out_dir, showWarnings = FALSE, recursive = TRUE)
  spots_all <- na.omit(fread(file.path(load_dir, "EDT_spotsassig_bft_peaks_allfovs_global_coord.csv")))
  stopifnot(all(c("bfs", "gene", "cell") %in% names(spots_all)))
  spots_all <- as.data.frame(spots_all)
  cell_location <- read.csv(file.path(load_dir, "EDT_bft_peaks_cell_location.csv"))

  # one reference for the whole sweep, on the full panel (as the old script did)
  ref <- reference_for(sceZ_all, unique(spots_all$gene))

  thresholds <- select_thresholds()
  message("thresholds: ", paste(sprintf("%.3f", thresholds), collapse = ", "))
  res <- Filter(Negate(is.null), lapply(thresholds, function(t) {
    r <- run_threshold(t, spots_all, cell_location, ref); gc(); r }))

  sweep  <- do.call(rbind, lapply(res, `[[`, "summary"))
  purity <- do.call(rbind, lapply(res, `[[`, "purity"))
  write.csv(sweep,  file.path(sweep_out_dir, "bfs_sweep_summary_bymisid.csv"), row.names = FALSE)
  write.csv(purity, file.path(sweep_out_dir, "bfs_marker_purity_bymisid.csv"), row.names = FALSE)

  # combined per-cell table (old script looked for "^annot_misid", which never
  # matched the "annot_bfs..." files it wrote)
  annot_files <- file.path(sweep_out_dir, sprintf("annot_bfs%.3f.csv", sweep$bfs_threshold))
  fwrite(rbindlist(lapply(annot_files, fread), fill = TRUE),
         file.path(sweep_out_dir, "annot_all_misid.csv"))

  print(sweep[, c("bfs_threshold", "n_spots", "n_cells", "n_genes",
                  "blank_spot_rate", "snr_area", "mean_delta", "frac_pruned")])
  invisible(sweep)
}

############################################################################
### Run
############################################################################
sceZ_all <- load_reference_cells()
print(table(sceZ_all$subclass_label))

results <- setNames(lapply(methods_to_run, run_method, sceZ_all = sceZ_all), methods_to_run)

# one table comparing all methods side by side
fwrite(rbindlist(lapply(results, `[[`, "metrics"), fill = TRUE),
       file.path(load_dir, "all_methods_cell_level_blank_metrics.csv"))

if (run_bft_sweep) sweep <- run_bft_sweep_all(sceZ_all)
