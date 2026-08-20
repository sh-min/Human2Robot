from pathlib import Path

from src.data_preprocess.extract_stereo_realsense_db3 import (
    CameraRecording,
    FrameRecord,
    filter_stereo_pairs_by_depth,
)


def _records(*timestamps: float) -> tuple[FrameRecord, ...]:
    return tuple(
        FrameRecord(
            source_index=index,
            bag_timestamp_ns=index,
            timestamp_s=timestamp,
            frame_number=index,
            metadata={},
        )
        for index, timestamp in enumerate(timestamps)
    )


def _camera(
    rgb_timestamps: tuple[float, ...], depth_timestamps: tuple[float, ...]
) -> CameraRecording:
    return CameraRecording(
        path=Path("unused.db3"),
        device={},
        color_info={},
        depth_info={},
        color_tf_raw="",
        depth_tf_raw="",
        depth_units_m=0.001,
        rgb=_records(*rgb_timestamps),
        depth=_records(*depth_timestamps),
    )


def test_filter_stereo_pairs_skips_missing_depth_frame() -> None:
    camera_1 = _camera((0.000, 0.033, 0.066), (0.000, 0.066))
    camera_2 = _camera((0.000, 0.033, 0.066), (0.000, 0.033, 0.066))

    pairs, depth_1, depth_2, rejected = filter_stereo_pairs_by_depth(
        camera_1,
        camera_2,
        [(0, 0), (1, 1), (2, 2)],
        max_delta_ms=20.0,
    )

    assert pairs == [(0, 0), (2, 2)]
    assert depth_1 == [0, 1]
    assert depth_2 == [0, 2]
    assert len(rejected) == 1
    assert "camera_1_depth_delta" in rejected[0]["reasons"]


def test_filter_stereo_pairs_never_reuses_depth_frame() -> None:
    camera_1 = _camera((0.000, 0.005, 0.020), (0.000, 0.020))
    camera_2 = _camera((0.000, 0.005, 0.020), (0.000, 0.005, 0.020))

    pairs, depth_1, depth_2, rejected = filter_stereo_pairs_by_depth(
        camera_1,
        camera_2,
        [(0, 0), (1, 1), (2, 2)],
        max_delta_ms=10.0,
    )

    assert pairs == [(0, 0), (2, 2)]
    assert depth_1 == [0, 1]
    assert depth_2 == [0, 2]
    assert rejected[0]["reasons"] == ["camera_1_depth_non_unique"]
