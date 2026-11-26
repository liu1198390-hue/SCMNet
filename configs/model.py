# coding=utf-8
"""Configuration helpers for SGMNet."""
from __future__ import annotations

from typing import Dict, Tuple


def build_model_config(
    *,
    input_channels: int = 3,
    num_classes: int = 1,
    patch_size: int = 4,
    embed_dim: int = 96,
    depths: Tuple[int, ...] = (2, 2, 9, 2),
    d_state: int = 16,
    d_conv: int = 3,
    expand: int = 2,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.2,
    decoder_channels: int = 128,
    ppm_channels: int = 128,
    ppm_pool_scales: Tuple[int, ...] = (1, 2, 3, 6),
    decoder_variant: str = "gsfpn",
    use_checkpoint: bool = False,
    use_separation_head: bool = True,
    use_gland_ssm_block: bool = True,
    gland_ssm_block_stages: Tuple[int, ...] = (0, 1, 2, 3),
    gland_ssm_scales: Tuple[int, ...] = (3, 5, 7),
    sgv_compact_ratio: float = 1.0,
    sgv_scan_guidance: bool = False,
    sgv_adapter_placement: str = "all",
    use_bssf_lite: bool = False,
    repair_variant: str = "unified_hrmdr",
    residual_scale_init: float = 5e-3,
    capc_residual_scale_init: float = 1e-2,
    capc_prior_scale_init: float = 0.10,
    use_capc: bool = True,
    hr_min_gate: float = 0.05,
    hr_what_gradient_scale: float = 0.50,
    hr_morphology_dilations: Tuple[int, ...] = (1, 3, 5),
    hr_image_feature_channels: int = 32,
    hr_cue_feature_channels: int = 16,
    capc_cue_channels: int = 16,
    capc_contact_floor: float = 0.25,
    capc_use_discrepancy: bool = True,
    capc_use_contact_cue: bool = True,
    capc_use_sgv_prior: bool = True,
    hr_use_cue_guidance: bool = True,
    qc_evidence_mode: str = "legacy",
    contact_evidence_power: float = 0.5,
    contact_contrast_gain: float = 2.0,
) -> Dict[str, object]:
    repair_variant = str(repair_variant).lower().replace("_", "").replace("-", "")
    if repair_variant in {"unified", "hrmdr"}:
        repair_variant = "unifiedhrmdr"
    if repair_variant not in {"unifiedhrmdr", "none"}:
        raise ValueError(f"Unsupported repair_variant: {repair_variant}")
    if repair_variant == "unifiedhrmdr" and not use_capc:
        raise ValueError("Unified HR-MDR requires use_capc=True.")
    decoder_variant = str(decoder_variant or "gsfpn").lower().replace("_", "").replace("-", "")
    if decoder_variant not in {"gsfpn", "legacy", "uperfpn", "upernet"}:
        raise ValueError(f"Unsupported decoder_variant: {decoder_variant}")
    decoder_variant = "gsfpn"
    if sgv_adapter_placement not in {"all", "last"}:
        raise ValueError("sgv_adapter_placement must be 'all' or 'last'")
    if qc_evidence_mode not in {"legacy", "continuous", "relative", "calibrated"}:
        raise ValueError("Unsupported qc_evidence_mode")
    if not 0.0 < float(contact_evidence_power) <= 1.0:
        raise ValueError("contact_evidence_power must be in (0, 1]")
    if float(contact_contrast_gain) <= 0.0:
        raise ValueError("contact_contrast_gain must be positive")
    dims = tuple(int(embed_dim) * (2 ** index) for index in range(len(depths)))
    return {
        'capc_residual_scale_init': float(capc_residual_scale_init),
        'capc_prior_scale_init': float(capc_prior_scale_init),
        "input_channels": int(input_channels),
        "num_classes": int(num_classes),
        "patch_size": int(patch_size),
        "embed_dim": int(embed_dim),
        "depths": tuple(int(value) for value in depths),
        "dims": dims,
        "d_state": int(d_state),
        "d_conv": int(d_conv),
        "expand": int(expand),
        "drop_rate": float(drop_rate),
        "attn_drop_rate": float(attn_drop_rate),
        "drop_path_rate": float(drop_path_rate),
        "decoder_channels": int(decoder_channels),
        "ppm_channels": int(ppm_channels),
        "ppm_pool_scales": tuple(int(value) for value in ppm_pool_scales),
        "decoder_variant": decoder_variant,
        "use_checkpoint": bool(use_checkpoint),
        "use_separation_head": bool(use_separation_head),
        "use_gland_ssm_block": bool(use_gland_ssm_block),
        "gland_ssm_block_stages": tuple(int(value) for value in gland_ssm_block_stages),
        "gland_ssm_scales": tuple(int(value) for value in gland_ssm_scales),
        "sgv_compact_ratio": float(sgv_compact_ratio),
        "sgv_scan_guidance": bool(sgv_scan_guidance),
        "sgv_adapter_placement": str(sgv_adapter_placement),
        "use_bssf_lite": bool(use_bssf_lite),
        "repair_variant": repair_variant,
        "residual_scale_init": float(residual_scale_init),
        "use_capc": bool(use_capc),
        "hr_min_gate": float(hr_min_gate),
        "hr_what_gradient_scale": float(hr_what_gradient_scale),
        "hr_morphology_dilations": tuple(int(value) for value in hr_morphology_dilations),
        "hr_image_feature_channels": int(hr_image_feature_channels),
        "hr_cue_feature_channels": int(hr_cue_feature_channels),
        "capc_cue_channels": int(capc_cue_channels),
        "capc_contact_floor": float(capc_contact_floor),
        "capc_use_discrepancy": bool(capc_use_discrepancy),
        "capc_use_contact_cue": bool(capc_use_contact_cue),
        "capc_use_sgv_prior": bool(capc_use_sgv_prior),
        "hr_use_cue_guidance": bool(hr_use_cue_guidance),
        "qc_evidence_mode": str(qc_evidence_mode),
        "contact_evidence_power": float(contact_evidence_power),
        "contact_contrast_gain": float(contact_contrast_gain),
        "encoder": "VMamba-style SS2D/VSS",
        "adapter": "SGV: structure-guided pretrained VSS adapter",
        "skip_decoder": "structure-calibrated CAPC-GSFPN: SGV prior calibrates CAPC boundary evidence",
        "scale_fusion": "original GSFPN scale-selective fusion",
        "repair_head": "Unified HR-MDR" if repair_variant == "unifiedhrmdr" else "none",
    }
