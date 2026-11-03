# SCMNet

Research code for gland instance segmentation in histopathology images.
SCMNet combines relation-aware visual state-space encoding, evidence-guided
pyramid fusion, and high-resolution prediction of foreground, local affinities,
contact gaps and instance embeddings.

## Model

The released default model has **25,505,688 trainable parameters (25.505688M)**.
Its encoder widths are `96/192/384/768` and depths are `2/2/9/2`.

| Component | Parameters |
|---|---:|
| Encoder | 22,137,907 |
| Pyramid decoder | 3,339,491 |
| High-resolution refinement and heads | 28,290 |
| Total | **25,505,688** |

Counts are measured by `profile_model.py`; the report is in
[`docs/parameter_counts.json`](docs/parameter_counts.json).
The released configuration uses decoder/PPM width 128, image-detail width 32,
refinement width 64 and embedding dimension 8.

```python
import torch
from models import SCMNet

model = SCMNet().eval()
with torch.no_grad():
    foreground_logits, aux = model(torch.randn(1, 3, 32, 48), return_aux=True)
# aux: relation_logits, contact_logits, instance_embedding,
#      encoder_relation_logits, fusion_alpha
```

## Installation

Use Python 3.10 or newer. Install a PyTorch build matching your CUDA environment
from the [official instructions](https://pytorch.org/get-started/locally/), then:

```bash
pip install -r requirements.txt
```

GPU training requires the fused selective-scan extension supplied by
[Mamba](https://github.com/state-spaces/mamba#installation). Follow its installation
instructions for your PyTorch, CUDA toolkit and compiler combination; the
training entry point checks that the extension is available.

With current Mamba releases, explicitly enable the CUDA scan build:

```bash
MAMBA_KEEP_CUDA_BUILD=TRUE pip install mamba-ssm --no-build-isolation
```

The default Mamba package installation may omit `selective_scan_cuda`; see the
upstream installation page for prebuilt-wheel and local-build options.

For small-input verification, the package includes a PyTorch reference scan that
runs on CPU. Local correctness checks were run with Python 3.12 and PyTorch
2.4.1 CPU. The CPU reference is intended for correctness checks, not slide-scale
training or throughput measurements.

## Data

Use integer instance masks: `0` is background and each gland has a distinct
positive ID. Images and masks must have matching spatial dimensions and stems.
The loader also accepts the suffixes `_mask`, `_label`, `_gt`, `_anno`,
`_annotation`, and `_anno_label` for masks. RGB instance labels are decoded into
integer IDs.

Prepare patch training data and whole-image validation/test data separately:

```text
TRAIN_ROOT/
  images/train/<image>.png
  labels_instance_cc/train/<image>.png
EVAL_ROOT/
  images/val/<image>.png
  labels_instance_cc/val/<image>.png
  images/test/<image>.png
  labels_instance_cc/test/<image>.png
```

Keep the dataset-defined or patient-level split before preparing patches.
Foreground, affinity and contact targets are derived from the transformed
instance IDs during training. Training uses geometric and color augmentation;
inference normalizes RGB values to `[-1, 1]` and merges overlapping tiles.

- GlaS: obtain data through the [GlaS challenge](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/).
- CRAG: obtain data through the [CRAG dataset](https://warwick.ac.uk/fac/cross_fac/tia/data/mildnet/).
- GZMH: institution-controlled data; access follows the data owner's process.

Datasets and patient information are not distributed with the source code.

## Training

Obtain the VMamba Tiny encoder checkpoint (`vmamba_tiny_e292.pth`) from the
[Swin-UMamba instructions](https://github.com/JiarunLiu/Swin-UMamba).
Only shape-compatible encoder tensors are loaded for initialization.

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python train.py \
  --train_data_path /path/to/train_patches \
  --val_data_path /path/to/whole_images \
  --pretrained /path/to/vmamba_tiny_e292.pth \
  --output_dir outputs/glas/seed0 \
  --seed 0 --epochs 30 --batch_size 2 --amp --deterministic
```

The default optimizer is AdamW with learning rate `1e-4`, minimum learning rate
`1e-6` and weight decay `1e-5`. Input size is 464, sliding-window stride is 232,
and the default fusion is `center_weighted`. Run `python train.py --help` for the
loss weights and instance-decoding thresholds.

Whole-image validation runs after each epoch. **Select checkpoints using source
validation object F1** (`best_obj_F1.pth`). Dice and Hausdorff checkpoints are also
saved for analysis. Each run writes its arguments, full model configuration,
history and logs to its output directory.

To resume, repeat the same command with
`--resume outputs/glas/seed0/checkpoints/latest.pth`. The code restores the model,
optimizer and AMP scaler and checks the saved training/inference settings.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --ckpt outputs/glas/seed0/checkpoints/best_obj_F1.pth \
  --data_path /path/to/whole_images --split test \
  --save_dir outputs/glas/seed0/test --save_predictions
```

For GlaS, use `--split testA` and `--split testB` for the two official subsets.
Use `--split test` for CRAG and GZMH. A combined GlaS `test` directory must contain
both subsets if you evaluate the combined split using that name.

Outputs include image-wise and mean object F1, Dice, IoU and Hausdorff distance,
pixel metrics, integer instance TIFF files, foreground probability TIFF files,
and side-by-side H&E/GT/prediction previews. Metrics are averaged over images,
using the instance decoder and GlaS object-metric definitions included here.

New checkpoints use the model name `SCMNet`. The loader also accepts original
training checkpoints and always uses strict state-dict loading; it does not
silently discard missing or mismatched tensors. Trained weights are not bundled.

## Five-seed runs

The wrapper supports `glas`, `crag`, and `gzmh`, with seeds `0,1,2,3,4` by default:

```bash
DATA_ROOT=/path/to/data PRETRAINED=/path/to/vmamba_tiny_e292.pth \
GPU_ID=0 bash scripts/run.sh glas
```

Default directories under `DATA_ROOT` follow this layout:

| Dataset | Patch training directory | Whole-image directory | Test splits |
|---|---|---|---|
| GlaS | `GlaS_coverage500_stride250_patches` | `GlaS_68_17_fg_aux` | `testA`, `testB`, `test` |
| CRAG | `CRAG_coverage500_stride250_patches` | `CRAG_fg_aux` | `test` |
| GZMH | `GZMH_GLAND_coverage500_stride250_patches` | `GZMH_GLAND_fg_aux` | `test` |

Override `TRAIN_DATA` and `EVAL_DATA` to use different names. Other overrides are
`PYTHON`, `SEEDS`, `EPOCHS`, `GPU_ID` and `OUTPUT_ROOT`. GlaS/CRAG use batch size 2;
GZMH uses batch size 4. Validation and test use the same decoding thresholds.
Choose thresholds on source validation data and keep them fixed for testing.

## Correctness checks

```bash
python -m unittest discover -s tests -v
python profile_model.py --smoke --backward --output model_profile.json
```

Tests cover the state-space recurrence, parameter count, supervised gradients,
odd-size inputs, sliding windows, checkpoint compatibility, contact barriers,
data loading, metric calculation and mask export. Synthetic test inputs verify
code behavior and are not experimental segmentation results.

The source cleanup was checked against the original model and loss definitions:
with identical weights and a `1 x 3 x 32 x 48` CPU input, the foreground and all
auxiliary outputs, total loss and loss components matched exactly. See
[`docs/validation.json`](docs/validation.json). CUDA training, timing and trained
accuracy require the corresponding GPU environment, datasets and checkpoints.

## Source layout

```text
models/scmnet.py       Encoder, pyramid fusion, refinement and output heads
models/vmamba.py       Visual state-space layers
models/scan.py         CUDA dispatch and small-input CPU reference
losses/scmnet.py       Foreground and instance-structure supervision
datasets/gland.py     Joint image/instance augmentation and loading
train.py              Training and checkpoint selection
test.py               Sliding-window evaluation and mask export
postprocess.py        Contact- and affinity-guided instance recovery
utils.py              Reproducibility, image utilities and object metrics
```

## License and acknowledgements

The project is distributed under [Apache-2.0](LICENSE), with upstream notices in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and `LICENSES/`.
We acknowledge VM-UNet, VMamba, Swin-UMamba and Mamba for their source components
and pretrained encoder ecosystem.
