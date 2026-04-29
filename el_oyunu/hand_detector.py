"""
HSV ten-rengi ile el tespiti (saf OpenCV, mediapipe gerektirmez).

Yaklasim:
- Kareyi HSV'ye cevir, ten rengi araliklarinda maske olustur.
- Morfolojik temizlik + en buyuk kontur = "el".
- Konturun konveks gobden ve defektlerden parmak sayisi tahmin edilir.
- En ust nokta parmak ucu olarak alinir (tek parmakla isaret etme).
- "Kalibrasyon" sirasinda kullanici elini referans dikdortgenine koyar,
  ortalama HSV'ye gore aralik dinamik olarak guncellenir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


Box = Tuple[int, int, int, int]


@dataclass
class HandResult:
    contour: np.ndarray
    bbox: Box
    center: Tuple[int, int]      # konturun agirlik merkezi
    fingertip: Tuple[int, int]   # en ustteki nokta (isaret parmagi tahmini)
    finger_count: int            # 0-5
    area: float
    mask: np.ndarray             # tek kanalli ten/el maskesi


class HandDetector:
    def __init__(self) -> None:
        # Varsayilan ten araliklari (iki ayri aralik kirmizi-tonlarinin
        # H ekseninde bolundugu icin). Kalibrasyon ile guncellenebilir.
        self._lower1 = np.array([0, 35, 60], dtype=np.uint8)
        self._upper1 = np.array([20, 180, 255], dtype=np.uint8)
        self._lower2 = np.array([160, 35, 60], dtype=np.uint8)
        self._upper2 = np.array([180, 180, 255], dtype=np.uint8)

        self._min_area = 3500
        self._calibrated = False

    # ------------------------------------------------------------------
    @property
    def calibrated(self) -> bool:
        return self._calibrated

    def calibrate_from_roi(self, frame_bgr: np.ndarray, roi: Box) -> bool:
        """ROI icindeki piksellerden HSV aralik tahmini yap."""
        x1, y1, x2, y2 = roi
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_bgr.shape[1], x2)
        y2 = min(frame_bgr.shape[0], y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return False
        patch = frame_bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0].reshape(-1)
        s = hsv[:, :, 1].reshape(-1)
        v = hsv[:, :, 2].reshape(-1)

        # robust olabilmek icin yuzdelik dilimler
        h_lo, h_hi = np.percentile(h, [5, 95])
        s_lo, s_hi = np.percentile(s, [5, 95])
        v_lo, v_hi = np.percentile(v, [5, 95])

        # H ekseni icin biraz tampon
        h_lo = max(0, int(h_lo) - 8)
        h_hi = min(180, int(h_hi) + 8)
        s_lo = max(0, int(s_lo) - 25)
        s_hi = min(255, int(s_hi) + 25)
        v_lo = max(0, int(v_lo) - 30)
        v_hi = min(255, int(v_hi) + 30)

        # Ten genelde dusuk H'de (0-25). Eger orta degerler oradaysa
        # tek aralik yeterli; degilse ikinci araligi soyut tutariz.
        self._lower1 = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
        self._upper1 = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
        # Ikinci aralik (kirmizinin diger ucu) sadece h_lo cok dusukse anlamli
        if h_lo <= 8:
            self._lower2 = np.array([170, s_lo, v_lo], dtype=np.uint8)
            self._upper2 = np.array([180, s_hi, v_hi], dtype=np.uint8)
        else:
            self._lower2 = np.array([0, 255, 255], dtype=np.uint8)
            self._upper2 = np.array([0, 255, 255], dtype=np.uint8)
        self._calibrated = True
        return True

    def reset_calibration(self) -> None:
        self.__init__()

    # ------------------------------------------------------------------
    def detect(
        self,
        frame_bgr: np.ndarray,
        ignore_boxes: Optional[List[Box]] = None,
    ) -> Optional[HandResult]:
        mask = self._skin_mask(frame_bgr)
        if ignore_boxes:
            for (x1, y1, x2, y2) in ignore_boxes:
                pad = 14
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(mask.shape[1], x2 + pad)
                y2 = min(mask.shape[0], y2 + pad)
                mask[y1:y2, x1:x2] = 0

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        if area < self._min_area:
            return None

        m = cv2.moments(cnt)
        if m["m00"] == 0:
            return None
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])

        x, y, w, h = cv2.boundingRect(cnt)
        bbox: Box = (x, y, x + w, y + h)

        # parmak ucu = konturun en ust noktasi
        top_idx = int(cnt[:, :, 1].argmin())
        tip = (int(cnt[top_idx, 0, 0]), int(cnt[top_idx, 0, 1]))

        fingers = self._count_fingers(cnt, area)

        return HandResult(
            contour=cnt,
            bbox=bbox,
            center=(cx, cy),
            fingertip=tip,
            finger_count=fingers,
            area=area,
            mask=mask,
        )

    # ------------------------------------------------------------------
    def _skin_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, self._lower1, self._upper1)
        m2 = cv2.inRange(hsv, self._lower2, self._upper2)
        mask = cv2.bitwise_or(m1, m2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        return mask

    def _count_fingers(self, cnt: np.ndarray, area: float) -> int:
        # Konveks gobek alanina kiyasla doluluk: kapali yumruk -> yuksek doluluk
        hull_pts = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull_pts)
        solidity = area / hull_area if hull_area > 0 else 1.0

        try:
            hull_idx = cv2.convexHull(cnt, returnPoints=False)
            if hull_idx is None or len(hull_idx) < 4:
                return 0 if solidity > 0.85 else 1
            defects = cv2.convexityDefects(cnt, hull_idx)
        except cv2.error:
            return 0 if solidity > 0.85 else 1

        if defects is None:
            return 0 if solidity > 0.85 else 1

        valid = 0
        # parmak araliklari icin asgari derinlik (alanla olceklendirilmis)
        min_depth = max(8000.0, area * 0.05) * 0.5

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            if d < min_depth:
                continue
            start = cnt[s, 0]
            end = cnt[e, 0]
            far = cnt[f, 0]
            a = float(np.linalg.norm(end - start))
            b = float(np.linalg.norm(far - start))
            c = float(np.linalg.norm(end - far))
            if b * c == 0:
                continue
            cos_ang = (b * b + c * c - a * a) / (2.0 * b * c)
            cos_ang = max(-1.0, min(1.0, cos_ang))
            angle = np.degrees(np.arccos(cos_ang))
            if angle < 95.0:
                valid += 1

        if valid == 0:
            # 0 defekt: ya yumruk ya da tek parmak
            return 0 if solidity > 0.85 else 1
        return min(5, valid + 1)

    # ------------------------------------------------------------------
    @staticmethod
    def draw_overlay(frame_bgr: np.ndarray, hand: HandResult) -> None:
        cv2.drawContours(frame_bgr, [hand.contour], -1, (60, 220, 120), 2)
        x1, y1, x2, y2 = hand.bbox
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (180, 255, 180), 1)
        cx, cy = hand.center
        cv2.circle(frame_bgr, (cx, cy), 6, (40, 40, 240), -1)
        tx, ty = hand.fingertip
        cv2.circle(frame_bgr, (tx, ty), 8, (40, 220, 255), -1)
        cv2.circle(frame_bgr, (tx, ty), 12, (40, 220, 255), 2)
