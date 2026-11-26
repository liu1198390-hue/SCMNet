# SCMNet

Compact gland instance segmentation with SGV, CAPC and MDR. The [model](models/scmnet.py) passes three scales of boundary and contact cues to MDR and predicts foreground and boundary logits with separate heads.

## Train and evaluate

Use Python 3.10+, [PyTorch and torchvision for CUDA](https://pytorch.org/get-started/locally/), and [Mamba CUDA selective scan](https://github.com/state-spaces/mamba#installation). Download `vmamba_tiny_e292.pth` from [Swin-UMamba](https://github.com/JiarunLiu/Swin-UMamba).

Prepare matching files in `images/<split>/`, `labels_instance_cc/<split>/`, `labels_foreground/<split>/`, and `labels_boundary/<split>/`. Use integer instance IDs with zero for background. Keep validation and test images at full resolution.

```bash
python -m pip install -r requirements.txt
python prepare.py --data_root data/whole --out_root data/patches --qc_num 0

CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py @configs/compact.args \
  --data_path data/whole --train_data_path data/patches \
  --vmamba_pretrained_path vmamba_tiny_e292.pth --output_dir outputs/seed0

CUDA_VISIBLE_DEVICES=0 python test.py --data_path data/whole --split test \
  --ckpt outputs/seed0/checkpoints/best_obj_Dice.pth \
  --run_config outputs/seed0/checkpoints/run_config.json --strict_checkpoint \
  --fusion center_weighted --no_train_log --save_dir outputs/seed0/test
```

Use `--batch_size 4` for GZMH-GLAND and `--split testA` or `testB` for individual GlaS subsets. To ablate the cue paths, train separate runs with `--no-capc_use_sgv_prior`, `--no-hr_use_cue_guidance`, or both.

Reproducing the manuscript tables requires their original run configurations and checkpoints.

[Apache-2.0](LICENSE). Visual state-space components are adapted from [VM-UNet](https://github.com/JCruan519/VM-UNet) and [VMamba](https://github.com/MzeroMiko/VMamba).
