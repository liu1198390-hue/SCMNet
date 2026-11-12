# SCMNet

Gland instance segmentation with relation-aware state-space encoding,
pyramid fusion and high-resolution refinement. The released model has
**25,505,688 trainable parameters (25.51M)**. It predicts foreground,
local affinities, contact gaps and instance embeddings. Its boundary
decoder derives a boundary score from adjacent foreground affinities.
The manuscript's dedicated boundary head and multi-scale Q/C paths
require their corresponding model implementation and checkpoints.

## Installation

Use Python 3.10+ and install [PyTorch](https://pytorch.org/get-started/locally/)
for your CUDA environment, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

GPU training requires the CUDA selective-scan extension from
[Mamba](https://github.com/state-spaces/mamba#installation). For current releases:

```bash
MAMBA_KEEP_CUDA_BUILD=TRUE pip install mamba-ssm --no-build-isolation
```

Obtain `vmamba_tiny_e292.pth` using the
[Swin-UMamba instructions](https://github.com/JiarunLiu/Swin-UMamba).

## Data

Use RGB images and integer instance masks: background is `0`, and each gland
has a distinct positive ID. Image and mask filenames must correspond.

```text
TRAIN_ROOT/
  images/train/
  labels_instance_cc/train/
EVAL_ROOT/
  images/val/
  labels_instance_cc/val/
  images/test/
  labels_instance_cc/test/
```

Prepare training patches and whole-image evaluation data using the original
dataset or patient-level split. Datasets and pretrained weights are obtained
separately.

## Training

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python train.py \
  --train_data_path /path/to/train_patches \
  --val_data_path /path/to/whole_images \
  --pretrained /path/to/vmamba_tiny_e292.pth \
  --output_dir outputs/seed0 \
  --seed 0 --epochs 30 --batch_size 2 --amp --deterministic \
  --val_postprocess boundary
```

The default input size is 464, with AdamW and initial learning rate `1e-4`.
Use batch size 2 for GlaS and CRAG or 4 for GZMH. The validation decoder is
recorded in each new checkpoint. Select the checkpoint with the highest
whole-image validation object Dice. To resume, add
`--resume outputs/seed0/checkpoints/latest.pth` to the same command.
For repeated runs, change `--seed` and `--output_dir` for each seed.

## Testing

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --ckpt outputs/seed0/checkpoints/best_obj_Dice.pth \
  --data_path /path/to/whole_images --split test \
  --save_dir outputs/seed0/test --save_predictions \
  --postprocess boundary
```

For GlaS, use `--split testA` or `--split testB`; a combined `test` directory
can also be prepared. CRAG and GZMH use `--split test`.
The example uses 464-pixel tiles, a 232-pixel stride and center-weighted
Hann fusion with a 0.05 weight floor. The boundary decoder uses foreground,
boundary and foreground-keep thresholds of 0.50, 0.60 and 0.75. It applies
disk opening (radius 5), fills holes, removes components smaller than 100
pixels, then labels instances with eight-connectivity. Use the same
decoder for validation and test. With no `--postprocess` argument, test
uses the checkpoint's validation decoder, falling back to `graph` for
earlier checkpoints.

Evaluation saves object F1, Dice, IoU and Hausdorff metrics. With
`--save_predictions`, it also exports instance masks and prediction previews.
See `python train.py --help` and `python test.py --help` for all options.

## Files

```text
models/          SCMNet and visual state-space layers
datasets/        Image and instance-mask loading
losses/          Training objectives
train.py         Training and validation
test.py          Evaluation and prediction export
postprocess.py   Instance recovery
utils.py         Shared utilities and metrics
```

## License and acknowledgements

SCMNet is distributed under [Apache-2.0](LICENSE). The visual state-space
components are adapted from [VM-UNet](https://github.com/JCruan519/VM-UNet)
and [VMamba](https://github.com/MzeroMiko/VMamba). The VMamba MIT copyright and
license are included in `LICENSE`.
The adapted scan code adds relation-conditioned transitions; the source
retains its modification notice. Swin-UMamba supplies the pretrained encoder
format, and Mamba supplies the separately installed CUDA scan extension.
