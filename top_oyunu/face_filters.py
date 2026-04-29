"""
Tespit edilen yuze sapka, gozluk, biyik gibi eglenceli sekiller cizer.
Sadece yuz kutusunun (x1, y1, x2, y2) bilgisini kullanir; ek bir model
yuklemeden orantilari basit oranlarla kestirip cv2 cizimleri ile uygular.
F tusu ile farkli filtre setleri arasinda gecilir.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


Box = Tuple[int, int, int, int]


FILTER_NAMES: List[str] = [
    "Yok",
    "Sihirbaz",
    "Hipster",
    "Palyaco",
    "Kraliyet",
    "Gangster",
    "Suuper Kahraman",
]
NUM_FILTERS = len(FILTER_NAMES)


def filter_name(index: int) -> str:
    return FILTER_NAMES[index % NUM_FILTERS]


# ---------- Yardimci cizim parcalari ---------------------------------------


def _draw_round_glasses(frame, x1, y1, x2, y2, color=(20, 20, 20)) -> None:
    fw = x2 - x1
    fh = y2 - y1
    eye_y = y1 + int(fh * 0.42)
    left_x = x1 + int(fw * 0.30)
    right_x = x1 + int(fw * 0.70)
    r = max(8, int(fw * 0.13))
    cv2.circle(frame, (left_x, eye_y), r, color, 3)
    cv2.circle(frame, (right_x, eye_y), r, color, 3)
    cv2.line(frame, (left_x + r, eye_y), (right_x - r, eye_y), color, 3)
    # sapanin uclari
    cv2.line(frame, (x1 + 2, eye_y), (left_x - r, eye_y), color, 2)
    cv2.line(frame, (right_x + r, eye_y), (x2 - 2, eye_y), color, 2)


def _draw_sun_glasses(frame, x1, y1, x2, y2) -> None:
    fw = x2 - x1
    fh = y2 - y1
    eye_y = y1 + int(fh * 0.42)
    left_x = x1 + int(fw * 0.30)
    right_x = x1 + int(fw * 0.70)
    rx = max(10, int(fw * 0.16))
    ry = max(6, int(fh * 0.07))
    cv2.ellipse(frame, (left_x, eye_y), (rx, ry), 0, 0, 360, (20, 20, 20), -1)
    cv2.ellipse(frame, (right_x, eye_y), (rx, ry), 0, 0, 360, (20, 20, 20), -1)
    cv2.line(frame, (left_x + rx, eye_y), (right_x - rx, eye_y), (20, 20, 20), 3)
    # parlaklik
    cv2.line(
        frame,
        (left_x - rx // 2, eye_y - ry // 2),
        (left_x, eye_y - ry // 2 + 1),
        (200, 200, 200),
        2,
    )


def _draw_hero_mask(frame, x1, y1, x2, y2) -> None:
    fw = x2 - x1
    fh = y2 - y1
    eye_y = y1 + int(fh * 0.42)
    half_h = max(10, int(fh * 0.16))
    pts = np.array(
        [
            [x1 + int(fw * 0.05), eye_y - half_h],
            [x2 - int(fw * 0.05), eye_y - half_h],
            [x2 - int(fw * 0.05), eye_y + half_h],
            [x1 + int(fw * 0.55), eye_y + int(half_h * 0.4)],
            [x1 + int(fw * 0.45), eye_y + int(half_h * 0.4)],
            [x1 + int(fw * 0.05), eye_y + half_h],
        ],
        dtype=np.int32,
    )
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (40, 30, 200))
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, dst=frame)
    cv2.polylines(frame, [pts], True, (0, 0, 0), 2)
    # goz delikleri
    rx = max(8, int(fw * 0.10))
    ry = max(5, int(fh * 0.06))
    left_x = x1 + int(fw * 0.32)
    right_x = x1 + int(fw * 0.68)
    cv2.ellipse(frame, (left_x, eye_y), (rx, ry), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(frame, (right_x, eye_y), (rx, ry), 0, 0, 360, (255, 255, 255), -1)


def _draw_mustache(frame, x1, y1, x2, y2) -> None:
    fw = x2 - x1
    fh = y2 - y1
    cx = (x1 + x2) // 2
    cy = y1 + int(fh * 0.72)
    half = max(10, int(fw * 0.22))
    h = max(4, int(fh * 0.06))
    cv2.ellipse(frame, (cx - half // 2, cy), (half // 2 + 4, h + 2), 0, 0, 180, (20, 20, 20), -1)
    cv2.ellipse(frame, (cx + half // 2, cy), (half // 2 + 4, h + 2), 0, 0, 180, (20, 20, 20), -1)


def _draw_wizard_hat(frame, x1, y1, x2, y2) -> None:
    fw = x2 - x1
    cx = (x1 + x2) // 2
    base_y = y1 + 4
    base_w = int(fw * 0.95)
    tip_y = max(2, y1 - int(fw * 1.1))
    pts = np.array(
        [
            [cx - base_w // 2, base_y],
            [cx + base_w // 2, base_y],
            [cx, tip_y],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [pts], (140, 50, 200))
    cv2.polylines(frame, [pts], True, (40, 10, 70), 2)
    cv2.rectangle(
        frame,
        (cx - base_w // 2, base_y - 8),
        (cx + base_w // 2, base_y),
        (20, 20, 20),
        -1,
    )
    # yildizlar
    for sx, sy in [
        (cx - base_w // 4, base_y - int(fw * 0.45)),
        (cx + base_w // 5, base_y - int(fw * 0.75)),
    ]:
        if sy > 0:
            cv2.putText(
                frame,
                "*",
                (sx - 8, sy + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 230, 255),
                2,
                cv2.LINE_AA,
            )


def _draw_top_hat(frame, x1, y1, x2, y2) -> None:
    fw = x2 - x1
    cx = (x1 + x2) // 2
    base_y = y1 + 4
    brim_w = int(fw * 1.05)
    top_w = int(fw * 0.7)
    top_h = max(20, int(fw * 0.85))
    top_y = max(2, base_y - top_h - 8)
    cv2.rectangle(frame, (cx - brim_w // 2, base_y - 8), (cx + brim_w // 2, base_y), (15, 15, 15), -1)
    cv2.rectangle(frame, (cx - top_w // 2, top_y), (cx + top_w // 2, base_y - 8), (15, 15, 15), -1)
    cv2.rectangle(
        frame,
        (cx - top_w // 2, base_y - 24),
        (cx + top_w // 2, base_y - 8),
        (160, 30, 30),
        -1,
    )


def _draw_crown(frame, x1, y1, x2, y2) -> None:
    fw = x2 - x1
    cx = (x1 + x2) // 2
    base_y = y1 + 2
    base_w = int(fw * 0.9)
    h = max(20, int(fw * 0.4))
    left = cx - base_w // 2
    right = cx + base_w // 2
    pts = np.array(
        [
            [left, base_y],
            [left, base_y - h // 2],
            [left + base_w // 4, base_y - h // 4],
            [cx - base_w // 8, max(2, base_y - h)],
            [cx, base_y - h // 2],
            [cx + base_w // 8, max(2, base_y - h)],
            [cx + base_w // 4, base_y - h // 4],
            [right, base_y - h // 2],
            [right, base_y],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [pts], (0, 200, 240))
    cv2.polylines(frame, [pts], True, (0, 130, 180), 2)
    for gx in [cx - base_w // 4, cx, cx + base_w // 4]:
        cv2.circle(frame, (gx, base_y - h // 3), 5, (60, 30, 220), -1)
        cv2.circle(frame, (gx, base_y - h // 3), 5, (255, 255, 255), 1)


def _draw_clown_nose(frame, x1, y1, x2, y2) -> None:
    fw = x2 - x1
    fh = y2 - y1
    cx = (x1 + x2) // 2
    cy = y1 + int(fh * 0.62)
    r = max(8, int(fw * 0.10))
    cv2.circle(frame, (cx, cy), r, (60, 60, 230), -1)
    cv2.circle(frame, (cx, cy), r, (255, 255, 255), 2)
    cv2.circle(frame, (cx - r // 3, cy - r // 3), max(2, r // 4), (255, 255, 255), -1)


def _draw_blush(frame, x1, y1, x2, y2) -> None:
    fw = x2 - x1
    fh = y2 - y1
    cy = y1 + int(fh * 0.62)
    rL = (x1 + int(fw * 0.18), cy)
    rR = (x1 + int(fw * 0.82), cy)
    overlay = frame.copy()
    cv2.circle(overlay, rL, max(6, int(fw * 0.08)), (140, 140, 255), -1)
    cv2.circle(overlay, rR, max(6, int(fw * 0.08)), (140, 140, 255), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, dst=frame)


def _draw_party_hat(frame, x1, y1, x2, y2) -> None:
    fw = x2 - x1
    cx = (x1 + x2) // 2
    base_y = y1 + 4
    base_w = int(fw * 0.55)
    tip_y = max(2, y1 - int(fw * 0.7))
    pts = np.array(
        [
            [cx - base_w // 2, base_y],
            [cx + base_w // 2, base_y],
            [cx, tip_y],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [pts], (60, 200, 255))
    cv2.polylines(frame, [pts], True, (0, 120, 180), 2)
    # ponponcuk
    if tip_y > 6:
        cv2.circle(frame, (cx, tip_y - 4), 6, (255, 255, 255), -1)


# ---------- Ana API --------------------------------------------------------


def apply_filter(frame: np.ndarray, face_boxes: List[Box], filter_index: int) -> None:
    """frame uzerinde verilen filtreyi tum yuzlere uygular."""
    if filter_index <= 0:
        return
    name = filter_name(filter_index)
    for (x1, y1, x2, y2) in face_boxes:
        if name == "Sihirbaz":
            _draw_wizard_hat(frame, x1, y1, x2, y2)
            _draw_round_glasses(frame, x1, y1, x2, y2, color=(40, 10, 70))
        elif name == "Hipster":
            _draw_party_hat(frame, x1, y1, x2, y2)
            _draw_round_glasses(frame, x1, y1, x2, y2)
            _draw_mustache(frame, x1, y1, x2, y2)
        elif name == "Palyaco":
            _draw_clown_nose(frame, x1, y1, x2, y2)
            _draw_blush(frame, x1, y1, x2, y2)
            _draw_party_hat(frame, x1, y1, x2, y2)
        elif name == "Kraliyet":
            _draw_crown(frame, x1, y1, x2, y2)
        elif name == "Gangster":
            _draw_top_hat(frame, x1, y1, x2, y2)
            _draw_sun_glasses(frame, x1, y1, x2, y2)
            _draw_mustache(frame, x1, y1, x2, y2)
        elif name == "Suuper Kahraman":
            _draw_hero_mask(frame, x1, y1, x2, y2)
