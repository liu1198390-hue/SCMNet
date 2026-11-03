"""Measure SCMNet parameters and optionally check small-input forward/backward."""

import argparse
import json
from pathlib import Path

import torch

from models import SCMNet
from models.scan import is_mamba_available


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("model_profile.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = SCMNet().to(args.device).eval()
    counts = {
        name: sum(p.numel() for p in getattr(model, name).parameters())
        for name in ("encoder", "decode_head", "hr_mdr")
    }
    total = sum(p.numel() for p in model.parameters())
    assert total == sum(counts.values()) == 25_505_688
    report = {
        "model": "SCMNet",
        "torch": torch.__version__,
        "parameters": total,
        "parameters_M": total / 1e6,
        "components": counts,
        "model_config": model.model_config,
    }
    if args.smoke or args.backward:
        image = torch.randn(1, 3, args.size, args.size + 16, device=args.device)
        region, aux = model(image, return_aux=True)
        assert region.shape == (1, 1, args.size, args.size + 16)
        assert aux["relation_logits"].shape == (1, 8, args.size, args.size + 16)
        assert aux["contact_logits"].shape == region.shape
        values = [region, aux["relation_logits"], aux["contact_logits"], aux["instance_embedding"]]
        assert all(torch.isfinite(v).all() for v in values)
        if args.backward:
            sum(v.square().mean() for v in values).backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        report["smoke"] = {
            "input": list(image.shape),
            "region": list(region.shape),
            "finite_outputs": True,
            "finite_backward": bool(args.backward),
            "backend": "fused_cuda"
            if args.device.startswith("cuda") and is_mamba_available()
            else "torch_reference",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
