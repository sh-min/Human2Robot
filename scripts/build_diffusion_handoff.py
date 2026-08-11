#!/usr/bin/env python3
"""Build a readable, frame-aligned skill-conditioned policy handoff.

The exporter is intentionally safe to run before the expensive perception
pipeline finishes.  Every episode always receives its source video, GT labels,
per-frame skill IDs, and one-hot conditions.  Inpainted observations and
calibrated robot actions are added automatically when their upstream artifacts
exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pkl_to_lerobot.schema import final_pose_to_state_action, states_to_actions
from utils.labels import ACTION_LABELS


def natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", value)]


def video_info(path: Path) -> tuple[int, float]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,nb_frames,avg_frame_rate",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    count = stream.get("nb_read_frames") or stream.get("nb_frames")
    if count in (None, "N/A"):
        raise ValueError(f"Cannot determine video frame count: {path}")
    numerator, denominator = stream["avg_frame_rate"].split("/")
    fps = float(numerator) / float(denominator)
    return int(count), fps


def replace_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            shutil.copy2(source, destination)
            return
    if mode == "symlink":
        destination.symlink_to(source.resolve())
        return
    shutil.copy2(source, destination)


def replace_aligned_video(
    source: Path,
    destination: Path,
    source_indices: np.ndarray,
    fps: float,
    mode: str,
) -> None:
    """Place a video whose frame order exactly matches the exported arrays."""
    source_count, _ = video_info(source)
    indices = np.asarray(source_indices, dtype=np.int64)
    if len(indices) == 0 or int(indices[-1]) >= source_count:
        raise ValueError(
            f"Cannot align {source}: frames={source_count}, "
            f"requested={indices[[0, -1]].tolist() if len(indices) else []}"
        )
    if (
        np.array_equal(indices, np.arange(source_count))
        and source.suffix.lower() == destination.suffix.lower()
    ):
        replace_file(source, destination, mode)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    select = "+".join(f"eq(n\\,{int(index)})" for index in indices)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(source), "-an",
            "-vf", f"select={select},setpts=N/({fps}*TB)",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-r", str(fps), str(destination),
        ],
        check=True,
    )


def find_source_video(episode: Path) -> Path:
    preferred = [
        episode / f"{episode.name}.mp4",
        episode / f"{episode.name}.MOV",
        episode / f"{episode.name}.mov",
    ]
    for candidate in preferred:
        if candidate.is_file():
            return candidate
    candidates = sorted(
        path for path in episode.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".mkv"}
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one source video in {episode}, found {len(candidates)}"
        )
    return candidates[0]


def first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def processed_artifacts(
    episode: Path,
) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    base = episode / "inpainting_processed" / episode.name / "0"
    background = first_existing([
        base / "inpaint_processor" / "video_human_inpaint.mp4",
        base / "inpaint_processor" / "video_human_inpaint.mkv",
    ])
    composite = first_existing([
        base / "video_overlay_rby1_xhand.mp4",
        base / "video_overlay_rby1_xhand.mkv",
    ])
    visualization = first_existing([
        base / "pipeline_components_rby1_xhand.mp4",
    ])
    render_metadata = first_existing([
        base / "overlay_processor_arm" / "render_metadata.json",
    ])
    return background, composite, visualization, render_metadata


def full_smoothing_verified(path: Path | None) -> bool:
    if path is None:
        return False
    metadata = json.loads(path.read_text())
    required = (
        "smoothed_finger_qpos",
        "smoothed_wrist_position",
        "smoothed_wrist_orientation",
        "arm_ik_uses_smoothed_wrist_pose",
    )
    return all(bool(metadata.get(key)) for key in required)


def expand_labels(gt: dict, label_to_id: dict[str, int]) -> tuple[np.ndarray, list[str]]:
    num_frames = int(gt["num_frames"])
    label_ids = np.full(num_frames, -1, dtype=np.int64)
    label_names = [""] * num_frames
    coverage = np.zeros(num_frames, dtype=np.int16)

    for segment in gt["segments"]:
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        label = str(segment["label"])
        if label not in label_to_id:
            raise ValueError(f"Unknown skill label: {label}")
        if start < 0 or end >= num_frames or start > end:
            raise ValueError(f"Invalid segment {start}-{end} for {num_frames} frames")
        label_ids[start:end + 1] = label_to_id[label]
        label_names[start:end + 1] = [label] * (end - start + 1)
        coverage[start:end + 1] += 1

    missing = int(np.sum(coverage == 0))
    overlap = int(np.sum(coverage > 1))
    if missing or overlap:
        raise ValueError(f"Invalid GT coverage: missing={missing}, overlap={overlap}")
    return label_ids, label_names


def load_aligned_actions(
    episode: Path,
    num_video_frames: int,
    action_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool] | None:
    pose_path = episode / "rgb_hawor" / "final_pose.pkl"
    if not pose_path.is_file():
        return None
    with pose_path.open("rb") as handle:
        final_pose = pickle.load(handle)
    if final_pose.get("coordinate_frame") != "rby1_base":
        return None

    states, _, valid = final_pose_to_state_action(
        final_pose, action_mode=action_mode
    )
    length = min(len(states), len(valid), num_video_frames)
    states = states[:length]
    valid_indices = np.flatnonzero(np.asarray(valid[:length], dtype=bool))
    if len(valid_indices) == 0:
        raise ValueError(f"No valid retargeted frames: {pose_path}")
    states = states[valid_indices]
    actions = states_to_actions(states, action_mode)
    pose_sources = final_pose.get("source", {})
    pkl_sources = [
        str(pose_sources[key])
        for key in ("right_pkl", "left_pkl")
        if pose_sources.get(key)
    ]
    smoothed_source = bool(pkl_sources) and all(
        Path(source).stem.endswith("_smooth") for source in pkl_sources
    )
    return states, actions, valid_indices, smoothed_source


def write_frame_table(
    path: Path,
    source_indices: np.ndarray,
    fps: float,
    label_ids: np.ndarray,
    label_names: list[str],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "aligned_frame_index",
            "source_frame_index",
            "timestamp_sec",
            "skill_id",
            "skill_name",
        ])
        for aligned_index, source_index in enumerate(source_indices):
            writer.writerow([
                aligned_index,
                int(source_index),
                f"{float(source_index) / fps:.6f}",
                int(label_ids[aligned_index]),
                label_names[aligned_index],
            ])


def write_readme(
    output_root: Path,
    scene_count: int,
    ready_count: int,
    labels: list[str],
    action_mode: str,
) -> None:
    pending = scene_count - ready_count
    text = f"""# Skill-conditioned Diffusion Policy handoff

이 폴더는 `26.07.27` 데이터셋을 Skill Classifier 및 LeRobot Diffusion
Policy 학습 담당자에게 전달하기 위한 패키지입니다.

## 현재 상태

- 전체 scene: **{scene_count}**
- policy 학습 준비 완료: **{ready_count}**
- 인페인팅/robot action 생성 대기: **{pending}**
- FPS 기준: 각 원본 영상 및 `metadata.json` 참고
- action 표현: **{action_mode}**, 양손 38차원
- one-hot 순서: `{labels}`

`manifest.csv`의 `ready_for_policy_training`이 `true`인 scene만 영상과
action이 모두 준비된 상태입니다. 현재 원본 영상과 라벨만 있는 scene을
학습 데이터로 오인하지 마세요.

## 전달 완성 조건

각 scene은 아래 네 파일이 모두 있어야 전달 완료입니다.

1. `robot_composite.mp4`: 사람 손·팔을 지운 배경에 로봇 손·팔을 합성한
   **Diffusion Policy 필수 영상 입력**
2. `robot_actions.npy`: 합성 영상의 각 프레임과 정렬된 38차원 action
3. `skill_onehot.npy`: 같은 프레임과 정렬된 6차원 skill condition
4. `source_frame_indices.npy`: 원본 영상과의 대응 관계

`source_rgb.*`나 사람만 제거된 `inpainted_rgb.mp4`만으로는 policy 전달
완료로 간주하지 않습니다. 이 스크립트 역시 `robot_composite.mp4`가
없으면 `ready_for_policy_training=false`로 기록합니다.

## 폴더 구조

```text
.
├── README.md
├── label_map.json
├── manifest.csv
├── manifest.json
├── validation_report.json
└── scenes/
    └── <episode_id>/
        ├── source_rgb.<mp4|mov>
        ├── gt_labels.json
        ├── frame_table.csv
        ├── skill_ids.npy
        ├── skill_onehot.npy
        ├── source_frame_indices.npy
        ├── metadata.json
        ├── inpainted_rgb.mp4             # 배경 확인용
        ├── robot_composite.mp4           # policy 필수 입력
        ├── pipeline_visualization.mp4     # 전처리/합성 육안 검사용
        ├── render_metadata.json           # 전체 스무딩 적용 증명
        ├── robot_states.npy              # policy state
        └── robot_actions.npy             # policy supervision
```

## 배열 정의

- `skill_ids.npy`: `(T,)`, `label_map.json`의 정수 ID
- `skill_onehot.npy`: `(T, {len(labels)})`, skill condition
- `source_frame_indices.npy`: `(T,)`, 원본 영상의 프레임 번호
- `robot_states.npy`: `(T, 38)`, 현재 robot state
- `robot_actions.npy`: `(T, 38)`, 다음 state를 예측 목표로 사용하는 action

38차원 순서는 다음과 같습니다.

```text
right: finger_joint[12], wrist_position_xyz[3], wrist_quaternion_xyzw[4]
left:  finger_joint[12], wrist_position_xyz[3], wrist_quaternion_xyzw[4]
```

현재 `26.07.27`의 30개 scene은 모두 왼손 시연입니다. 따라서 오른손
19차원은 0으로 채워지고, 왼손 19차원에 추정된 관절·손목 pose가
저장됩니다.

action이 있는 scene은 invalid hand-tracking 프레임을 제거한 뒤 영상,
action, skill condition을 같은 `source_frame_indices.npy`로 정렬합니다.

`validation_report.json`은 모든 scene의 MP4 프레임 수, 배열 shape,
NaN/Inf, one-hot 유효성, `action[t] = state[t+1]` 관계를 검사한
결과입니다. `error_count`가 0인지 확인한 후 학습에 사용하세요.

## 권장 학습 입력

- 영상 observation: `robot_composite.mp4` (**필수**)
- skill condition: `skill_onehot.npy`
- supervision: `robot_actions.npy`

`inpainted_rgb.mp4`는 사람 손/팔을 제거한 배경 확인용이고,
`robot_composite.mp4`는 그 배경 위에 로봇 embodiment를 렌더링한 policy
입력입니다.

`render_metadata.json`의 `smoothed_finger_qpos`,
`smoothed_wrist_position`, `smoothed_wrist_orientation`,
`arm_ik_uses_smoothed_wrist_pose`가 모두 `true`인지 검증했습니다.
`pipeline_visualization.mp4`에서는 원본, 손+팔 마스크, 인페인팅 배경,
로봇 렌더링 및 최종 합성을 한 화면에서 확인할 수 있습니다.
`robot_actions.npy` 역시 `final_pose.pkl`이 참조하는
`qpos_xhand_*_smooth.pkl`에서 생성됐으며, 각 `metadata.json`의
`action_smoothing_verified=true`로 확인할 수 있습니다.

훈련 시 GT one-hot을 사용하고, 추론 시에는 Skill Classifier의 예측
one-hot 또는 확률 벡터를 같은 condition 입력에 넣어야 합니다.

## 패키지 갱신

상위 파이프라인 결과가 추가된 뒤 아래 명령을 다시 실행하면 같은 폴더가
갱신됩니다.

```bash
cd /home/robin/shMin/skill2policy
PYTHONPATH=src conda run -n vjepa2-312 --no-capture-output \\
  python scripts/build_diffusion_handoff.py \\
  --data_root data/cube_dataset/26.07.27 \\
  --out_dir {output_root} \\
  --require_full_smoothing \\
  --strict_ready
```
"""
    (output_root / "README.md").write_text(text)


def validate_handoff(
    output_root: Path,
    manifest: list[dict],
    skill_dimension: int,
    action_mode: str,
) -> None:
    """Validate aligned videos, actions, states, and skill conditions."""
    report = {
        "scene_count": len(manifest),
        "total_aligned_frames": 0,
        "ready_count": 0,
        "error_count": 0,
        "errors": [],
        "scenes": [],
    }

    for item in manifest:
        episode_id = item["episode_id"]
        scene_dir = output_root / "scenes" / episode_id
        expected = int(item["aligned_num_frames"])
        states = np.load(scene_dir / "robot_states.npy")
        actions = np.load(scene_dir / "robot_actions.npy")
        skill_ids = np.load(scene_dir / "skill_ids.npy")
        onehot = np.load(scene_dir / "skill_onehot.npy")
        indices = np.load(scene_dir / "source_frame_indices.npy")

        checks = {
            "ready_for_policy_training": bool(item["ready_for_policy_training"]),
            "full_smoothing_verified": bool(item["full_smoothing_verified"]),
            "states_shape": states.shape == (expected, 38),
            "actions_shape": actions.shape == (expected, 38),
            "skill_ids_shape": skill_ids.shape == (expected,),
            "onehot_shape": onehot.shape == (expected, skill_dimension),
            "indices_shape": indices.shape == (expected,),
            "finite": bool(
                np.isfinite(states).all()
                and np.isfinite(actions).all()
                and np.isfinite(onehot).all()
            ),
        }
        checks["onehot_valid"] = bool(
            checks["onehot_shape"]
            and np.allclose(onehot.sum(axis=1), 1.0)
            and np.array_equal(onehot.argmax(axis=1), skill_ids)
        )

        if checks["states_shape"] and checks["actions_shape"]:
            if action_mode == "absolute":
                expected_actions = np.concatenate(
                    [states[1:], states[-1:]], axis=0
                )
            else:
                expected_actions = np.concatenate(
                    [states[1:] - states[:-1], np.zeros_like(states[-1:])],
                    axis=0,
                )
            checks["action_state_relation"] = bool(
                np.allclose(actions, expected_actions, atol=1e-6)
            )
        else:
            checks["action_state_relation"] = False

        videos = {}
        for name in (
            "inpainted_rgb.mp4",
            "robot_composite.mp4",
            "pipeline_visualization.mp4",
        ):
            path = scene_dir / name
            try:
                frames, fps = video_info(path)
            except Exception as exc:
                frames, fps = -1, -1.0
                videos[name] = {"error": str(exc)}
            else:
                videos[name] = {"frames": frames, "fps": fps}
            checks[f"{name}_frames"] = frames == expected

        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            report["errors"].append({
                "episode_id": episode_id,
                "failed_checks": failed,
            })
        else:
            report["ready_count"] += 1
        report["total_aligned_frames"] += expected
        report["scenes"].append({
            "episode_id": episode_id,
            "frames": expected,
            "checks": checks,
            "videos": videos,
        })

    report["error_count"] = len(report["errors"])
    (output_root / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        f"Validation: scenes={report['scene_count']}, "
        f"frames={report['total_aligned_frames']}, "
        f"errors={report['error_count']}"
    )
    if report["error_count"]:
        raise ValueError(
            f"Handoff validation failed: {report['error_count']} scene(s)"
        )


def build(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.out_dir).resolve()
    scenes_root = output_root / "scenes"
    scenes_root.mkdir(parents=True, exist_ok=True)

    labels = list(ACTION_LABELS)
    label_to_id = {label: index for index, label in enumerate(labels)}
    (output_root / "label_map.json").write_text(json.dumps({
        "labels": labels,
        "label_to_id": label_to_id,
        "onehot_dimension": len(labels),
        "transition_label": "Trans",
    }, indent=2) + "\n")

    episodes = sorted(
        (path for path in data_root.iterdir()
         if path.is_dir() and (path / "gt_labels.json").is_file()),
        key=lambda path: natural_key(path.name),
    )
    manifest: list[dict] = []

    for episode in episodes:
        gt_path = episode / "gt_labels.json"
        gt = json.loads(gt_path.read_text())
        source_video = find_source_video(episode)
        video_frames, fps = video_info(source_video)
        if video_frames != int(gt["num_frames"]):
            raise ValueError(
                f"{episode.name}: video={video_frames}, GT={gt['num_frames']}"
            )

        all_label_ids, all_label_names = expand_labels(gt, label_to_id)
        aligned = load_aligned_actions(
            episode, video_frames, args.action_mode
        )
        if aligned is None:
            states = actions = None
            action_smoothing_verified = False
            source_indices = np.arange(video_frames, dtype=np.int64)
        else:
            states, actions, source_indices, action_smoothing_verified = aligned

        label_ids = all_label_ids[source_indices]
        label_names = [all_label_names[int(index)] for index in source_indices]
        onehot = np.eye(len(labels), dtype=np.float32)[label_ids]

        scene_dir = scenes_root / episode.name
        scene_dir.mkdir(parents=True, exist_ok=True)
        for pattern in (
            "inpainted_rgb.*",
            "robot_composite.*",
            "pipeline_visualization.*",
            "render_metadata.json",
            "robot_states.npy",
            "robot_actions.npy",
        ):
            for stale in scene_dir.glob(pattern):
                stale.unlink()
        replace_file(
            source_video,
            scene_dir / f"source_rgb{source_video.suffix.lower()}",
            args.link_mode,
        )
        shutil.copy2(gt_path, scene_dir / "gt_labels.json")
        np.save(scene_dir / "skill_ids.npy", label_ids)
        np.save(scene_dir / "skill_onehot.npy", onehot)
        np.save(scene_dir / "source_frame_indices.npy", source_indices)
        write_frame_table(
            scene_dir / "frame_table.csv",
            source_indices,
            fps,
            label_ids,
            label_names,
        )

        background, composite, visualization, render_metadata = (
            processed_artifacts(episode)
        )
        render_smoothing_verified = full_smoothing_verified(render_metadata)
        smoothing_verified = (
            render_smoothing_verified and action_smoothing_verified
        )
        if background is not None:
            replace_aligned_video(
                background,
                scene_dir / "inpainted_rgb.mp4",
                source_indices,
                fps,
                args.link_mode,
            )
        if composite is not None:
            replace_aligned_video(
                composite,
                scene_dir / "robot_composite.mp4",
                source_indices,
                fps,
                args.link_mode,
            )
        if visualization is not None:
            replace_aligned_video(
                visualization,
                scene_dir / "pipeline_visualization.mp4",
                source_indices,
                fps,
                args.link_mode,
            )
        if render_metadata is not None:
            shutil.copy2(render_metadata, scene_dir / "render_metadata.json")
        if states is not None and actions is not None:
            np.save(scene_dir / "robot_states.npy", states.astype(np.float32))
            np.save(scene_dir / "robot_actions.npy", actions.astype(np.float32))

        missing = []
        if background is None:
            missing.append("inpainted_rgb")
        if composite is None:
            missing.append("robot_composite")
        if actions is None:
            missing.append("calibrated_robot_actions")
        if visualization is None:
            missing.append("pipeline_visualization")
        if args.require_full_smoothing and not smoothing_verified:
            missing.append("verified_full_smoothing")
        ready = not missing
        try:
            source_episode = str(episode.relative_to(ROOT))
        except ValueError:
            source_episode = episode.name
        metadata = {
            "episode_id": episode.name,
            "source_num_frames": video_frames,
            "aligned_num_frames": int(len(source_indices)),
            "fps": fps,
            "action_mode": args.action_mode,
            "action_dimension": 38,
            "skill_dimension": len(labels),
            "render_smoothing_verified": render_smoothing_verified,
            "action_smoothing_verified": action_smoothing_verified,
            "full_smoothing_verified": smoothing_verified,
            "ready_for_policy_training": ready,
            "missing": missing,
            "source_episode": source_episode,
        }
        (scene_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        manifest.append(metadata)
        print(
            f"[{'READY' if ready else 'PENDING'}] {episode.name}: "
            f"T={len(source_indices)}, missing={','.join(missing) or '-'}"
        )

    with (output_root / "manifest.csv").open("w", newline="") as handle:
        fieldnames = [
            "episode_id", "source_num_frames", "aligned_num_frames", "fps",
            "action_mode", "action_dimension", "skill_dimension",
            "render_smoothing_verified", "action_smoothing_verified",
            "full_smoothing_verified",
            "ready_for_policy_training", "missing", "source_episode",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in manifest:
            row = dict(item)
            row["ready_for_policy_training"] = (
                "true" if item["ready_for_policy_training"] else "false"
            )
            row["full_smoothing_verified"] = (
                "true" if item["full_smoothing_verified"] else "false"
            )
            row["render_smoothing_verified"] = (
                "true" if item["render_smoothing_verified"] else "false"
            )
            row["action_smoothing_verified"] = (
                "true" if item["action_smoothing_verified"] else "false"
            )
            row["missing"] = ";".join(item["missing"])
            writer.writerow(row)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    ready_count = sum(bool(item["ready_for_policy_training"]) for item in manifest)
    write_readme(
        output_root,
        len(manifest),
        ready_count,
        labels,
        args.action_mode,
    )
    validate_handoff(output_root, manifest, len(labels), args.action_mode)
    print(f"\nHandoff: {output_root}")
    print(f"Scenes: {len(manifest)}, ready={ready_count}, pending={len(manifest)-ready_count}")
    if args.strict_ready and ready_count != len(manifest):
        raise SystemExit("Not every scene is ready for policy training")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--link_mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="How videos are placed in the handoff (hardlink falls back to copy).",
    )
    parser.add_argument(
        "--action_mode",
        choices=("absolute", "delta"),
        default="absolute",
    )
    parser.add_argument(
        "--strict_ready",
        action="store_true",
        help="Fail unless every scene has inpainted video, composite, and actions.",
    )
    parser.add_argument(
        "--require_full_smoothing",
        action="store_true",
        help="Require render metadata proving finger, wrist position, wrist "
             "orientation, and arm-IK smoothing for every ready scene.",
    )
    build(parser.parse_args())


if __name__ == "__main__":
    main()
