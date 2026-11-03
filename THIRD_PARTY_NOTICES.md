# Third-party notices

SCMNet includes modified visual state-space components based on these projects:

- VM-UNet: https://github.com/JCruan519/VM-UNet (Apache-2.0).
  The Apache license text is retained in the root LICENSE.
- VMamba: https://github.com/MzeroMiko/VMamba (MIT).
  Copyright (c) 2024 MzeroMiko. Its full notice is retained in
  LICENSES/VMamba-MIT.txt.
- Swin-UMamba: https://github.com/JiarunLiu/Swin-UMamba documents the pretrained
  encoder format. Pretrained weights are obtained separately and are not
  redistributed in this repository.
- Mamba: https://github.com/state-spaces/mamba supplies the optional fused
  selective-scan extension. The extension is an external dependency.

Modified files: models/vmamba.py adapts the scan transitions using local
same-instance evidence and removes an unused duplicate scan implementation.
models/layers.py contains the shared normalization, convolution and pyramid
layers used by SCMNet. models/scan.py dispatches to an installed CUDA extension
and provides a small-input, real-valued PyTorch recurrence for verification.

The SCMNet model, training, supervision and instance-decoding code has been
organized into standalone modules while preserving the original trained model's
state-dict keys and tensor shapes. The original checkpoint version string is
accepted solely for compatibility.
