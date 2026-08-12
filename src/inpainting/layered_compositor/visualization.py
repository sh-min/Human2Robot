"""Visualization helpers kept separate from compositing semantics."""

from __future__ import annotations

import cv2
import numpy as np


def checkerboard(height: int, width: int, cell: int = 24) -> np.ndarray:
    yy, xx = np.indices((height, width))
    value = np.where(((xx // cell) + (yy // cell)) % 2 == 0, 24, 38)
    return np.repeat(value[..., None], 3, axis=2).astype(np.uint8)


def isolated_layer(content: np.ndarray, mask: np.ndarray,
                   plate: np.ndarray,
                   edge_bgr: tuple[int, int, int]) -> np.ndarray:
    """Show one sparse layer on a checkerboard transparency plate."""

    mask_u8 = np.asarray(mask, dtype=np.uint8)
    active = mask_u8.astype(bool)
    result = plate.copy()
    result[active] = np.asarray(content, dtype=np.uint8)[active]
    edge = cv2.morphologyEx(
        mask_u8, cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ).astype(bool)
    result[edge] = edge_bgr
    return result


def context_layer(background: np.ndarray, content: np.ndarray,
                  mask: np.ndarray,
                  edge_bgr: tuple[int, int, int]) -> np.ndarray:
    """Show one layer over a dim scene so its position stays recognizable."""

    mask_u8 = np.asarray(mask, dtype=np.uint8)
    active = mask_u8.astype(bool)
    result = np.clip(
        np.asarray(background, dtype=np.float32) * 0.20, 0, 255
    ).astype(np.uint8)
    result[active] = np.asarray(content, dtype=np.uint8)[active]
    edge = cv2.morphologyEx(
        mask_u8, cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ).astype(bool)
    result[edge] = edge_bgr
    return result


def label(frame: np.ndarray, text: str) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (8, 8), (8 + 18 * len(text), 45), (0, 0, 0), -1)
    cv2.putText(result, text, (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 2, cv2.LINE_AA)
    return result


def grid_3x2(frames: list[np.ndarray], width: int, height: int) -> np.ndarray:
    tile_w = width // 3
    tile_h = height // 2
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for index, frame in enumerate(frames[:6]):
        row, col = divmod(index, 3)
        resized = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        x0, y0 = col * tile_w, row * tile_h
        canvas[y0:y0 + tile_h, x0:x0 + tile_w] = resized
    return canvas
