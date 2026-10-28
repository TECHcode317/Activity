# HAR-15: Leakage-Controlled Image-Based Human Activity Classification

Reference implementation for 15-class human activity classification from single RGB images with

* a fine-tuned **CLIP ViT-B/16** (`vit_base_patch16_clip_224.openai`, proposed), and
* an **EfficientNet-B0** baseline (`efficientnet_b0`, ImageNet weights).

Both models take an RGB image as input and return one of 15 activity classes. The repository contains no pose
estimation, detection, tracking or temporal component, and it makes no claim about clinical monitoring.

The repository provides everything needed to audit the reported results: the near-duplicate grouping code, the fixed
group and split assignments, the training and evaluation code for both models, the per-image prediction records, and a
script that recomputes every reported metric from those records.

## Repository layout

| Path | Content |
|---|---|
| `make_splits.py` | Near-duplicate audit (64-bit dHash, Hamming distance <= 4, connected components) and group-stratified splitting. |
| `train.py` | Training, evaluation, cross-validation and efficiency benchmark for one model (`--model vit` or `--model effnet`). |
| `har_common.py` | Data loading, grouping, split functions, GPU augmentation, prediction. |
| `metrics.py` | Metric computation (CPU only). |
| `verify_metrics.py` | Recomputes all reported metrics from `predictions_*.csv` (CPU only, no images required). |
| `manifests/` | Fixed assignments written by `make_splits.py`: `groups.csv`, `split_main.csv`, `split_cv.csv`. |
| `results/<model>/` | Outputs written by `train.py`. |
| `requirements.txt` | Python dependencies. |

## Requirements

* Python 3.10 or newer
* One NVIDIA GPU for training and benchmarking (experiments were run on a Tesla T4 in a Kaggle notebook)
* About 2 GB of GPU memory for the cached image tensor, plus model memory

```
pip install -r requirements.txt
```

The exact library versions, GPU name and Python version of every run are written to
`results/<model>/all_results.json`.

## Data

The experiments use the public Kaggle dataset `meetnagadia/human-action-recognition-har-dataset`
(12,600 labelled images, 15 classes, `Training_set.csv` and a `train/` folder).

```
pip install kaggle
kaggle datasets download -d meetnagadia/human-action-recognition-har-dataset -p data --unzip
```

Set the dataset location with `--root` or the environment variable `HAR_ROOT`; the folder containing
`Training_set.csv` is located automatically below that path. The images are third-party data and are not redistributed
in this repository. The manifests refer to images by file name only.

## Reproducing the experiments

```
python make_splits.py --root "data/Human Action Recognition"
python train.py --model vit --cv
python train.py --model effnet
python verify_metrics.py --main results/vit/predictions_main.csv --cv results/vit/predictions_cv.csv
python verify_metrics.py --main results/effnet/predictions_main.csv
```

| Option | Meaning |
|---|---|
| `--root` | Dataset folder (default: `$HAR_ROOT`). |
| `--manifests` | Folder with the split manifests (default `manifests`). |
| `--out` | Output folder (default `results`). |
| `--cache` | Path of the cached 224 x 224 image tensor (default `cache_224.npy`). |
| `--cv` | Run 5-fold group-stratified cross-validation (ViT only, 4 epochs per fold). |
| `--no-benchmark` | Skip the latency, FLOPs and power measurement. |

Pretrained weights are downloaded by `timm` on first use; the run stops with an error if they cannot be loaded.

On Kaggle: enable the GPU and Internet, upload the repository files, and run the commands above with a leading `!`.

## Grouping and splitting

`make_splits.py` implements the leakage control described in the manuscript.

1. Each image is resized to 224 x 224 (bilinear) and converted to a 64-bit difference hash (8 x 9 grayscale grid).
2. Two images are linked when the Hamming distance of their hashes is at most 4; connected components of this graph
   are the image groups (12,438 groups for the 12,600 images).
3. `StratifiedGroupKFold` (seed 42) produces the fixed split: 7 folds with one fold as test set, then 6 folds of the
   remainder with one fold as validation set (9,005 / 1,796 / 1,799 images). No group appears in more than one subset;
   the script asserts this.
4. The same procedure with 5 outer folds produces the cross-validation folds.

Augmentation is applied on the fly to training batches only, after splitting.

### Manifest files

| File | Columns |
|---|---|
| `manifests/groups.csv` | `filename`, `label`, `group` |
| `manifests/split_main.csv` | `filename`, `label`, `group`, `split` (`train`, `val` or `test`) |
| `manifests/split_cv.csv` | `filename`, `label`, `group`, `fold1` ... `fold5` (`train`, `val` or `test` per fold) |

`train.py` reads these manifests, so training always uses the stored assignments.

## Prediction records

| File | Content |
|---|---|
| `results/<model>/predictions_main.csv` | `filename`, `label_true`, `label_pred` and the 15 class probabilities `p_<class>` for every image of the fixed test split. |
| `results/vit/predictions_cv.csv` | The same columns plus `fold` for the test images of all five folds. |

The probabilities are the flip-averaged SoftMax outputs used for the reported metrics.

## Verifying the reported metrics

```
python verify_metrics.py --main results/vit/predictions_main.csv --expect-accuracy 91.44
```

`verify_metrics.py` recomputes accuracy, balanced accuracy, macro and weighted precision, recall and F1, macro one-vs-rest
AUC, the per-class report, a 95% confidence interval for accuracy and, with `--cv`, the fold-wise accuracy and macro-F1
with mean and sample standard deviation. It checks that `label_pred` equals the arg max of the stored probabilities.
With `--expect-accuracy` it exits with an error if the recomputed accuracy differs from the given value.

Reference values from the manuscript: ViT test accuracy 91.44% (1,645 of 1,799 images); 5-fold cross-validation
accuracy 91.53 +/- 0.74%.

## Implementation of the two models

| | CLIP ViT-B/16 | EfficientNet-B0 |
|---|---|---|
| Pretrained weights | OpenAI CLIP (`vit_base_patch16_clip_224.openai`) | ImageNet (`efficientnet_b0`) |
| Input | 224 x 224 x 3 RGB | 224 x 224 x 3 RGB |
| Optimiser | `torch.optim.AdamW` (decoupled weight decay), betas (0.9, 0.999) | `torch.optim.AdamW`, one parameter group |
| Base / peak learning rate | 1e-4 (base rate eta_0) | 6e-4 |
| Layer-wise learning-rate decay | 0.75; head rate 5 x eta_0 | none |
| Schedule | 1-epoch linear warm-up, cosine decay, floor 1% | One-cycle, 15% warm-up (`OneCycleLR`) |
| Weight decay | 0.05, not applied to bias, normalisation, `cls_token`, `pos_embed` | 0.01 on all parameters |
| Batch size / epochs | 32 / 8 (4 per cross-validation fold) | 64 / 15 |
| Loss | Cross-entropy, label smoothing 0.1 | Cross-entropy, label smoothing 0.1 |
| Regularisation | Drop-path 0.1, dropout 0.1, gradient-norm clipping 1.0 | Dropout 0.3 |
| EMA of weights | decay 0.998, `min(0.998, (1+t)/(10+t))` | none |
| Precision | FP16 autocast | FP16 autocast |
| Memory format | default | channels-last |
| Augmentation (training only) | horizontal flip, zoom 0.65-1.0, rotation +/-0.2 rad, translation, brightness and contrast 0.8-1.2 | same, zoom 0.7-1.0 |
| Test-time augmentation | horizontal-flip averaging | horizontal-flip averaging |
| Model selection | best validation accuracy (EMA weights) | best validation accuracy (raw weights) |
| Seed | 42 | 42 |

Layer-wise learning rates follow `eta_l = eta_0 * 0.75^(L+1-l)` with `L = 12` transformer blocks, `l = 0` for the patch
and position embeddings and `l = 1..12` for the blocks.

Training loss is the label-smoothed cross-entropy. For the ViT, validation loss is the plain negative log-likelihood of
the EMA model's SoftMax output (no label smoothing, no flip averaging) and is logged for monitoring only; it is not
comparable with the training loss, whose minimum for 15 classes and smoothing 0.1 is approximately 0.556.

## Efficiency measurement protocol

`train.py` measures the trained model on random 224 x 224 x 3 tensors (no disk access or image decoding) in FP16 autocast.

| Quantity | Protocol |
|---|---|
| Parameters, GFLOPs | `torch.utils.flop_counter` on a 1 x 3 x 224 x 224 input; GFLOPs = 2 x MACs. |
| Latency | Mean of 100 forward passes at batch size 1 after 10 warm-up passes, with CUDA synchronisation. |
| Throughput | Mean over 30 timed passes at batch size 32 (ViT) or 64 (EfficientNet-B0). |
| Power | `nvidia-smi --query-gpu=power.draw` polled every 100 ms by a background thread while the energy benchmark runs. |
| Energy workload | ViT: 80 passes at batch size 32; EfficientNet-B0: 150 passes at batch size 64. |
| Energy per image | Mean power x elapsed time / number of images. The elapsed time includes the 10 warm-up passes. |

`all_results.json` also stores the number of power samples, the elapsed time, the number of images and the steady-state
energy per image (mean power / throughput), so the energy values can be checked against power, time and image count.

The values are whole-board readings of a data-centre GPU in a shared cloud notebook, not edge-device measurements.
The sensor has a limited accuracy and averaging window, so absolute energies are indicative and the ratio between the two
models is the more reliable quantity. The two models are benchmarked at the batch sizes of their training configuration.

## Outputs and their use in the manuscript

| Output | Manuscript item |
|---|---|
| `manifests/*.csv` | Group assignments, train/validation/test split, cross-validation folds |
| `results/<model>/results_summary.csv`, `classification_report.txt` | Test metrics and per-class results |
| `results/<model>/predictions_main.csv`, `predictions_cv.csv` | Per-image predictions behind all reported metrics |
| `results/<model>/confusion_matrix.csv`, `confusion_matrix*.png`, `per_class_f1.png` | Confusion matrix, per-class F1 |
| `results/<model>/training_history.csv`, `training_curves.png` | Training and validation curves |
| `results/vit/cv_folds.csv`, `kfold_validation.png` | Cross-validation results |
| `results/<model>/all_results.json` | Hyperparameters, environment, parameters, GFLOPs, latency, throughput, power, energy |
| `results/vit/complexity_comparison.csv`, `gflops_comparison.png` | Complexity comparison with reference networks |

## Reproducibility notes

* All random generators are seeded (seed 42) and the data split is fixed by the manifests.
* GPU kernels are not bit-wise deterministic; repeated runs with identical seeds can differ by about one percentage point
  of accuracy. The prediction files are the reference for the reported numbers.
* `StratifiedGroupKFold` assignments can depend on the scikit-learn version; the manifests are the authoritative split.
* The best weights of each run are saved as `results/<model>_best_weights.pt` (not tracked by git).

## Scope

The code evaluates image-level classification of 15 generic activities on a public dataset. It does not evaluate pose
estimation, routine or missed-activity monitoring, patient data, or any clinical outcome.

## Dataset and licence

The dataset is distributed by its owners on Kaggle under its own terms. Please cite the dataset and the manuscript when
using this code.
