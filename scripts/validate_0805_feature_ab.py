#!/usr/bin/env python3
"""Validate the controlled inputs for the 08-05 calibration classifier A/B.

The approximate-focal and calibrated-focal datasets are copies of the same
RGB recordings.  Their labels, sampling contract, and token alignment must be
identical, and frozen V-JEPA features must agree within numerical tolerance.
MANO features and their validity flags are allowed to change because they are
the input whose sensitivity to the focal calibration is being measured.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APPROX_ROOT = (
    REPO_ROOT / "data" / "kitchen_dataset" / "26.08.05_stereo_approx"
)
DEFAULT_CALIBRATED_ROOT = (
    REPO_ROOT / "data" / "kitchen_dataset" / "26.08.05_stereo_calibrated"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "output" / "skill_classifier" / "0805_calibration_comparison"
)

EXACT_METADATA_FIELDS = (
    "feature_schema_version",
    "recording",
    "action_labels",
    "num_frames",
    "num_tokens",
    "sampling_profile",
    "source_fps",
    "sample_fps",
    "token_rate_hz",
    "clip_frames",
    "tubelet_size",
    "spatial_profile",
    "label_boundary_policy",
)
EXACT_ALIGNMENT_FIELDS = (
    "labels_per_token",
    "sampled_frame_indices",
    "token_frame_indices",
    "token_center_frame_indices",
    "frame_to_token",
)
REQUIRED_FEATURE_FIELDS = (
    "vjepa_orig",
    "mano",
    "mano_valid_per_token",
)


class FeatureABValidationError(ValueError):
    """Raised when the two A/B inputs do not form a controlled experiment."""


def _normalise_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        return value.item()
    return value


def _require_fields(bundle: dict[str, Any], path: Path) -> None:
    required = set(EXACT_METADATA_FIELDS + EXACT_ALIGNMENT_FIELDS)
    required.update(REQUIRED_FEATURE_FIELDS)
    missing = sorted(required - set(bundle))
    if missing:
        raise FeatureABValidationError(
            f"{path}: feature bundle is missing required keys: {missing}"
        )


def load_feature_bundle(path: Path) -> dict[str, Any]:
    """Load one bundle using PyTorch's restricted weights-only loader."""

    if not path.is_file():
        raise FeatureABValidationError(f"missing feature bundle: {path}")
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - exact PyTorch error is version-specific
        raise FeatureABValidationError(
            f"failed to load {path} with weights_only=True: {exc}"
        ) from exc
    if not isinstance(bundle, dict):
        raise FeatureABValidationError(
            f"{path}: expected a dictionary, got {type(bundle).__name__}"
        )
    _require_fields(bundle, path)
    return bundle


def _require_exact_metadata(
    approx: dict[str, Any],
    calibrated: dict[str, Any],
    episode: str,
) -> None:
    for key in EXACT_METADATA_FIELDS:
        approx_value = _normalise_value(approx[key])
        calibrated_value = _normalise_value(calibrated[key])
        if approx_value != calibrated_value:
            raise FeatureABValidationError(
                f"episode {episode}: metadata field {key!r} differs: "
                f"approx={approx_value!r}, calibrated={calibrated_value!r}"
            )


def _as_tensor(value: Any, *, episode: str, branch: str, key: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise FeatureABValidationError(
            f"episode {episode}: {branch} field {key!r} must be a tensor, "
            f"got {type(value).__name__}"
        )
    return value.detach().cpu()


def _require_exact_alignment(
    approx: dict[str, Any],
    calibrated: dict[str, Any],
    episode: str,
) -> None:
    for key in EXACT_ALIGNMENT_FIELDS:
        left = _as_tensor(approx[key], episode=episode, branch="approx", key=key)
        right = _as_tensor(
            calibrated[key], episode=episode, branch="calibrated", key=key
        )
        if left.shape != right.shape:
            raise FeatureABValidationError(
                f"episode {episode}: alignment field {key!r} shape differs: "
                f"approx={tuple(left.shape)}, calibrated={tuple(right.shape)}"
            )
        if left.dtype != right.dtype:
            raise FeatureABValidationError(
                f"episode {episode}: alignment field {key!r} dtype differs: "
                f"approx={left.dtype}, calibrated={right.dtype}"
            )
        if not torch.equal(left, right):
            mismatch_count = int(torch.count_nonzero(left != right).item())
            raise FeatureABValidationError(
                f"episode {episode}: alignment field {key!r} differs at "
                f"{mismatch_count} element(s)"
            )


def _paired_float_tensors(
    approx_value: Any,
    calibrated_value: Any,
    *,
    episode: str,
    key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    approx = _as_tensor(
        approx_value, episode=episode, branch="approx", key=key
    )
    calibrated = _as_tensor(
        calibrated_value, episode=episode, branch="calibrated", key=key
    )
    if approx.shape != calibrated.shape:
        raise FeatureABValidationError(
            f"episode {episode}: {key!r} shape differs: "
            f"approx={tuple(approx.shape)}, calibrated={tuple(calibrated.shape)}"
        )
    if approx.numel() == 0:
        raise FeatureABValidationError(
            f"episode {episode}: {key!r} must contain at least one value"
        )
    if not (approx.is_floating_point() and calibrated.is_floating_point()):
        raise FeatureABValidationError(
            f"episode {episode}: {key!r} must use floating-point tensors"
        )
    approx64 = approx.to(torch.float64)
    calibrated64 = calibrated.to(torch.float64)
    if not (torch.isfinite(approx64).all() and torch.isfinite(calibrated64).all()):
        raise FeatureABValidationError(
            f"episode {episode}: {key!r} contains non-finite values"
        )
    return approx64, calibrated64


def _difference_stats(abs_difference: torch.Tensor) -> dict[str, Any]:
    flat = abs_difference.reshape(-1).to(torch.float64)
    if flat.numel() == 0:
        raise FeatureABValidationError("cannot summarize an empty difference tensor")
    return {
        "num_values": int(flat.numel()),
        "max_abs_difference": float(flat.max().item()),
        "mean_abs_difference": float(flat.mean().item()),
        "p95_abs_difference": float(torch.quantile(flat, 0.95).item()),
    }


def _validity_report(
    approx_value: Any,
    calibrated_value: Any,
    *,
    episode: str,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    key = "mano_valid_per_token"
    approx = _as_tensor(
        approx_value, episode=episode, branch="approx", key=key
    )
    calibrated = _as_tensor(
        calibrated_value, episode=episode, branch="calibrated", key=key
    )
    if approx.shape != calibrated.shape:
        raise FeatureABValidationError(
            f"episode {episode}: {key!r} shape differs: "
            f"approx={tuple(approx.shape)}, calibrated={tuple(calibrated.shape)}"
        )
    if approx.dtype != torch.bool or calibrated.dtype != torch.bool:
        raise FeatureABValidationError(
            f"episode {episode}: {key!r} must have torch.bool dtype"
        )
    if approx.numel() == 0:
        raise FeatureABValidationError(
            f"episode {episode}: {key!r} must contain at least one value"
        )
    equal = approx == calibrated
    both_valid = approx & calibrated
    both_invalid = ~approx & ~calibrated
    approx_only = approx & ~calibrated
    calibrated_only = ~approx & calibrated
    report = {
        "shape": list(approx.shape),
        "num_values": int(approx.numel()),
        "agreement_rate": float(equal.to(torch.float64).mean().item()),
        "disagreement_count": int(torch.count_nonzero(~equal).item()),
        "approx_valid_count": int(torch.count_nonzero(approx).item()),
        "calibrated_valid_count": int(torch.count_nonzero(calibrated).item()),
        "both_valid_count": int(torch.count_nonzero(both_valid).item()),
        "both_invalid_count": int(torch.count_nonzero(both_invalid).item()),
        "approx_only_valid_count": int(torch.count_nonzero(approx_only).item()),
        "calibrated_only_valid_count": int(
            torch.count_nonzero(calibrated_only).item()
        ),
    }
    return report, approx.reshape(-1), calibrated.reshape(-1)


def compare_episode(
    approx_path: Path,
    calibrated_path: Path,
    *,
    episode: str,
    atol: float,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Validate and summarize one episode pair."""

    approx = load_feature_bundle(approx_path)
    calibrated = load_feature_bundle(calibrated_path)
    _require_exact_metadata(approx, calibrated, episode)
    _require_exact_alignment(approx, calibrated, episode)

    vjepa_approx, vjepa_calibrated = _paired_float_tensors(
        approx["vjepa_orig"],
        calibrated["vjepa_orig"],
        episode=episode,
        key="vjepa_orig",
    )
    vjepa_difference = (vjepa_approx - vjepa_calibrated).abs()
    vjepa_stats = _difference_stats(vjepa_difference)
    if vjepa_stats["max_abs_difference"] > atol:
        raise FeatureABValidationError(
            f"episode {episode}: vjepa_orig max absolute difference "
            f"{vjepa_stats['max_abs_difference']:.9g} exceeds atol={atol:.9g}; "
            "the two branches must use identical RGB and frozen V-JEPA inputs"
        )

    mano_approx, mano_calibrated = _paired_float_tensors(
        approx["mano"], calibrated["mano"], episode=episode, key="mano"
    )
    mano_difference = (mano_approx - mano_calibrated).abs()
    mano_stats = _difference_stats(mano_difference)
    mano_stats["identical"] = bool(mano_stats["max_abs_difference"] == 0.0)

    validity, validity_approx, validity_calibrated = _validity_report(
        approx["mano_valid_per_token"],
        calibrated["mano_valid_per_token"],
        episode=episode,
    )

    labels = _as_tensor(
        approx["labels_per_token"],
        episode=episode,
        branch="approx",
        key="labels_per_token",
    )
    report = {
        "episode": episode,
        "input_paths": {
            "approx": str(approx_path.resolve()),
            "calibrated": str(calibrated_path.resolve()),
        },
        "counts": {
            "num_frames": int(approx["num_frames"]),
            "num_tokens": int(approx["num_tokens"]),
            "labelled_tokens": int(torch.count_nonzero(labels >= 0).item()),
        },
        "action_labels": list(approx["action_labels"]),
        "sampling_contract": {
            key: _normalise_value(approx[key])
            for key in (
                "sampling_profile",
                "source_fps",
                "sample_fps",
                "token_rate_hz",
                "clip_frames",
                "tubelet_size",
                "spatial_profile",
                "label_boundary_policy",
            )
        },
        "exact_checks": {
            "metadata_fields": list(EXACT_METADATA_FIELDS),
            "alignment_fields": list(EXACT_ALIGNMENT_FIELDS),
            "passed": True,
        },
        "vjepa_orig_difference": {
            "shape": list(vjepa_approx.shape),
            "atol": float(atol),
            "within_tolerance": True,
            **vjepa_stats,
        },
        "mano_difference": {
            "shape": list(mano_approx.shape),
            **mano_stats,
        },
        "mano_validity": validity,
    }
    raw = {
        "vjepa_difference": vjepa_difference.reshape(-1),
        "mano_difference": mano_difference.reshape(-1),
        "validity_approx": validity_approx,
        "validity_calibrated": validity_calibrated,
    }
    return report, raw


def _aggregate_validity(
    approx: torch.Tensor, calibrated: torch.Tensor
) -> dict[str, Any]:
    report, _, _ = _validity_report(
        approx,
        calibrated,
        episode="aggregate",
    )
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 08-05 calibration feature A/B validation",
        "",
        "**Status: PASS**",
        "",
        (
            "Labels, sampling metadata, token alignment, and frozen V-JEPA "
            "features satisfy the controlled-input checks. MANO differences "
            "are diagnostic and are not required to be non-zero."
        ),
        "",
        f"- Approx root: `{report['roots']['approx']}`",
        f"- Calibrated root: `{report['roots']['calibrated']}`",
        f"- V-JEPA absolute tolerance: `{report['vjepa_atol']}`",
        "",
        "| Episode | Frames | Tokens | V-JEPA max | V-JEPA mean | V-JEPA p95 | MANO max | MANO mean | MANO p95 | Validity agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for episode in report["episodes"]:
        vjepa = episode["vjepa_orig_difference"]
        mano = episode["mano_difference"]
        validity = episode["mano_validity"]
        lines.append(
            "| {episode} | {frames} | {tokens} | {vmax:.8g} | {vmean:.8g} | "
            "{vp95:.8g} | {mmax:.8g} | {mmean:.8g} | {mp95:.8g} | "
            "{agreement:.6f} |".format(
                episode=episode["episode"],
                frames=episode["counts"]["num_frames"],
                tokens=episode["counts"]["num_tokens"],
                vmax=vjepa["max_abs_difference"],
                vmean=vjepa["mean_abs_difference"],
                vp95=vjepa["p95_abs_difference"],
                mmax=mano["max_abs_difference"],
                mmean=mano["mean_abs_difference"],
                mp95=mano["p95_abs_difference"],
                agreement=validity["agreement_rate"],
            )
        )

    aggregate = report["aggregate"]
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- V-JEPA max/mean/p95 absolute difference: "
            f"`{aggregate['vjepa_orig_difference']['max_abs_difference']:.8g}` / "
            f"`{aggregate['vjepa_orig_difference']['mean_abs_difference']:.8g}` / "
            f"`{aggregate['vjepa_orig_difference']['p95_abs_difference']:.8g}`",
            f"- MANO max/mean/p95 absolute difference: "
            f"`{aggregate['mano_difference']['max_abs_difference']:.8g}` / "
            f"`{aggregate['mano_difference']['mean_abs_difference']:.8g}` / "
            f"`{aggregate['mano_difference']['p95_abs_difference']:.8g}`",
            f"- MANO validity agreement: "
            f"`{aggregate['mano_validity']['agreement_rate']:.6f}` "
            f"({aggregate['mano_validity']['disagreement_count']} disagreements)",
            "",
            "## Inputs",
            "",
        ]
    )
    for episode in report["episodes"]:
        lines.extend(
            [
                f"### Episode {episode['episode']}",
                "",
                f"- Approx: `{episode['input_paths']['approx']}`",
                f"- Calibrated: `{episode['input_paths']['calibrated']}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def validate_feature_ab(
    approx_root: Path,
    calibrated_root: Path,
    *,
    episodes: Sequence[str] = ("1", "2"),
    atol: float = 1.0e-6,
    output_dir: Path | None = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Validate all episode pairs and optionally write JSON/Markdown reports."""

    if not math.isfinite(atol) or atol < 0:
        raise FeatureABValidationError("atol must be finite and non-negative")
    episode_names = [str(episode) for episode in episodes]
    if not episode_names or any(not name.strip() for name in episode_names):
        raise FeatureABValidationError("at least one non-empty episode is required")
    if len(set(episode_names)) != len(episode_names):
        raise FeatureABValidationError("episode names must be unique")

    approx_root = Path(approx_root)
    calibrated_root = Path(calibrated_root)
    episode_reports: list[dict[str, Any]] = []
    raw_reports: list[dict[str, torch.Tensor]] = []
    for episode in episode_names:
        episode_report, raw = compare_episode(
            approx_root / episode / "features.pt",
            calibrated_root / episode / "features.pt",
            episode=episode,
            atol=atol,
        )
        episode_reports.append(episode_report)
        raw_reports.append(raw)

    all_vjepa = torch.cat([raw["vjepa_difference"] for raw in raw_reports])
    all_mano = torch.cat([raw["mano_difference"] for raw in raw_reports])
    all_validity_approx = torch.cat(
        [raw["validity_approx"] for raw in raw_reports]
    )
    all_validity_calibrated = torch.cat(
        [raw["validity_calibrated"] for raw in raw_reports]
    )
    aggregate_mano = _difference_stats(all_mano)
    aggregate_mano["identical"] = bool(
        aggregate_mano["max_abs_difference"] == 0.0
    )
    report = {
        "schema_version": 1,
        "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "roots": {
            "approx": str(approx_root.resolve()),
            "calibrated": str(calibrated_root.resolve()),
        },
        "vjepa_atol": float(atol),
        "episodes": episode_reports,
        "aggregate": {
            "num_episodes": len(episode_reports),
            "vjepa_orig_difference": {
                "within_tolerance": True,
                **_difference_stats(all_vjepa),
            },
            "mano_difference": aggregate_mano,
            "mano_validity": _aggregate_validity(
                all_validity_approx, all_validity_calibrated
            ),
        },
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "feature_ab_validation.json"
        markdown_path = output_dir / "feature_ab_validation.md"
        json_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approx-root", type=Path, default=DEFAULT_APPROX_ROOT)
    parser.add_argument(
        "--calibrated-root", type=Path, default=DEFAULT_CALIBRATED_ROOT
    )
    parser.add_argument("--episodes", nargs="+", default=["1", "2"])
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = validate_feature_ab(
            args.approx_root,
            args.calibrated_root,
            episodes=args.episodes,
            atol=args.atol,
            output_dir=args.output_dir,
        )
    except FeatureABValidationError as exc:
        parser.exit(2, f"feature A/B validation failed: {exc}\n")
    print(
        "Feature A/B validation passed: "
        f"{len(report['episodes'])} episode(s); reports in {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
