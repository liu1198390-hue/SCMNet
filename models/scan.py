"""Selective scan dispatch and a small-input PyTorch reference backend.

CUDA uses the installed fused extension. CPU reference is for correctness tests;
it is not intended for whole-slide training.
"""

import torch
from torch.nn import functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _fused_scan
except (ImportError, OSError):
    try:
        from selective_scan import selective_scan_fn as _fused_scan
    except (ImportError, OSError):
        _fused_scan = None


def is_mamba_available():
    return callable(_fused_scan)


def selective_scan_reference(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    """Real-valued selective state recurrence with constant or grouped B/C."""
    dtype = u.dtype
    u, delta, A = u.float(), delta.float(), A.float()
    batch, channels, length = u.shape
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    state = u.new_zeros(batch, channels, A.shape[-1])

    def at(value, index):
        value = value.float()
        if value.ndim == 2:
            return value[None]
        if value.ndim == 3:
            return value[:, :, index][:, None, :]
        if value.ndim == 4:
            groups = value.shape[1]
            if channels % groups:
                raise ValueError("Invalid B/C group count")
            return value[:, :, :, index].repeat_interleave(channels // groups, dim=1)
        raise ValueError("Unsupported B/C layout")

    outputs = []
    for index in range(length):
        step = delta[:, :, index, None]
        state = torch.exp(step * A[None]) * state + step * at(B, index) * u[:, :, index, None]
        outputs.append((state * at(C, index)).sum(-1))
    y = torch.stack(outputs, dim=-1)
    if D is not None:
        y = y + u * D.float()[None, :, None]
    if z is not None:
        y = y * F.silu(z.float())
    y = y.to(dtype)
    return (y, state) if return_last_state else y


def selective_scan_fn(u, *args, **kwargs):
    if u.is_cuda:
        if _fused_scan is None:
            raise RuntimeError("Install a CUDA-compatible mamba-ssm/selective_scan extension")
        return _fused_scan(u, *args, **kwargs)
    return selective_scan_reference(u, *args, **kwargs)
