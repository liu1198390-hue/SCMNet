"""Parameter-free structure evidence shared by the segmentation heads.

Kept separate from the encoder so the probability/gradient contract can be
tested on CPU without a CUDA selective-scan extension.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def calibrated_contact_evidence(logits, power=0.5, gain=2.0, window=15):
    """Combine absolute contact confidence with confidence-weighted local contrast.

    Zero logits give no permission. Confident spatially broad contacts remain
    active; a low-confidence local fluctuation is not treated as a certain cut.
    The local mean uses valid pixels only, avoiding a false rim at tile borders.
    """
    if not 0.0 < float(power) <= 1.0 or float(gain) <= 0:
        raise ValueError('Contact power must be in (0,1] and gain must be positive')
    if int(window) < 1 or int(window) % 2 == 0:
        raise ValueError('Contact evidence window must be a positive odd integer')
    values = logits.float()
    probability = torch.sigmoid(values)
    evidence = torch.exp(-float(power) * F.softplus(-values))
    background = F.avg_pool2d(evidence, int(window), stride=1,
                            padding=int(window)//2, count_include_pad=False)
    contrast = F.relu(evidence-background-1e-3) / (1.0-background).clamp_min(0.1)
    local = (float(gain)*evidence*contrast).clamp(0.0, 1.0)
    absolute = F.relu(2.0*probability-1.0)
    permission = 1.0-(1.0-absolute)*(1.0-local)
    return permission.clamp(0.0, 1.0).to(dtype=logits.dtype)


def joint_instance_logits(foreground_logits, boundary_logits):
    """Stable logit of sigmoid(foreground) * sigmoid(-boundary), in float32.

    Unlike clamping the product of half-precision probabilities before BCE, this
    retains a finite corrective gradient when a confident foreground bridge also
    has a confident boundary response.
    """
    foreground, boundary = foreground_logits.float(), boundary_logits.float()
    return -torch.logaddexp(-foreground, boundary+F.softplus(-foreground))
