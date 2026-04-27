"""
Top oyunu icin sade yuz (kafa) yakalayici.
OpenCV Haar cascade kullanir; sadece (x1, y1, x2, y2) kutulari dondurur.
"""

import os
import shutil
import tempfile
from typing import List, Tuple

import cv2


Box = Tuple[int, int, int, int]


def _load_cascade(filename: str) -> cv2.CascadeClassifier:
    """Cascade XML'ini yukler. Windows'ta non-ASCII yollari icin temp'e kopyalar."""
    src = os.path.join(cv2.data.haarcascades, filename)
    if not os.path.isfile(src):
        raise RuntimeError(f"Cascade bulunamadi: {src}")

    try:
        src.encode("ascii")
        cascade = cv2.CascadeClassifier(src)
        if not cascade.empty():
            return cascade
    except UnicodeEncodeError:
        pass

    tmp_dir = os.path.join(tempfile.gettempdir(), "topoyunu_cv2")
    os.makedirs(tmp_dir, exist_ok=True)
    dst = os.path.join(tmp_dir, filename)
    if not os.path.isfile(dst):
        shutil.copyfile(src, dst)

    cascade = cv2.CascadeClassifier(dst)
    if cascade.empty():
        raise RuntimeError(f"Haar cascade yuklenemedi: {dst}")
    return cascade


class FaceDetector:
    def __init__(self):
        self._cascade = _load_cascade("haarcascade_frontalface_default.xml")

    def detect(self, frame_bgr) -> List[Box]:
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        min_side = max(40, int(min(h, w) * 0.08))

        rects = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(min_side, min_side),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        faces: List[Box] = []
        for (x, y, bw, bh) in rects:
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(w - 1, int(x + bw))
            y2 = min(h - 1, int(y + bh))
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2, y2))
        return faces
