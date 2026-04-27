import os
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple

import cv2


Box = Tuple[int, int, int, int]


def _load_cascade_safely(filename: str) -> cv2.CascadeClassifier:
    """OpenCV FileStorage, Windows'ta non-ASCII yollarda hata veriyor.
    Cascade XML'ini ASCII-guvenli bir temp yola kopyalayip oradan yukluyoruz.
    """
    src = os.path.join(cv2.data.haarcascades, filename)
    if not os.path.isfile(src):
        raise RuntimeError(f"Cascade bulunamadi: {src}")

    try:
        src.encode("ascii")
        ascii_safe = True
    except UnicodeEncodeError:
        ascii_safe = False

    if ascii_safe:
        cascade = cv2.CascadeClassifier(src)
        if not cascade.empty():
            return cascade

    tmp_dir = os.path.join(tempfile.gettempdir(), "facededection_cv2")
    os.makedirs(tmp_dir, exist_ok=True)
    dst = os.path.join(tmp_dir, filename)
    if not os.path.isfile(dst):
        shutil.copyfile(src, dst)

    cascade = cv2.CascadeClassifier(dst)
    if cascade.empty():
        raise RuntimeError(f"Haar cascade yuklenemedi: {dst}")
    return cascade


class FaceDetector:
    """OpenCV Haar cascade tabanli yuz yakalayici + duygu (gulumseme) tahmini.
    detect()    -> sadece kutular (geri uyumluluk)
    analyze()   -> [(box, {'smile': bool, 'eyes_open': bool}), ...]
    """

    def __init__(self, min_confidence: float = 0.6, model_selection: int = 0):
        self._min_confidence = float(min_confidence)
        self._cascade = _load_cascade_safely("haarcascade_frontalface_default.xml")
        self._smile_cascade = _load_cascade_safely("haarcascade_smile.xml")
        self._eye_cascade = _load_cascade_safely("haarcascade_eye.xml")

    def _detect_boxes(self, frame_bgr) -> List[Box]:
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        min_neighbors = 5 if self._min_confidence >= 0.6 else 3
        min_side = max(40, int(min(h, w) * 0.08))

        rects = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=min_neighbors,
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

    def detect(self, frame_bgr) -> List[Box]:
        # Eski API: soldan saga sirali kutular
        faces = self._detect_boxes(frame_bgr)
        faces.sort(key=lambda b: b[0])
        return faces

    def analyze(self, frame_bgr) -> List[Tuple[Box, Dict[str, bool]]]:
        """Kareyi alir, her yuz icin (box, {'smile':..., 'eyes_open':...}) doner.
        Sirali degildir; tracker stabil siralamayi yapacak.
        """
        gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_full = cv2.equalizeHist(gray_full)

        out: List[Tuple[Box, Dict[str, bool]]] = []
        for box in self._detect_boxes(frame_bgr):
            x1, y1, x2, y2 = box
            fw = x2 - x1
            fh = y2 - y1
            if fw < 20 or fh < 20:
                continue

            face_gray = gray_full[y1:y2, x1:x2]

            # --- gulumseme: yuzun alt yarisinda ara ---
            mouth_y1 = int(fh * 0.55)
            mouth_roi = face_gray[mouth_y1:fh, :]
            smile = False
            if mouth_roi.size > 0:
                smiles = self._smile_cascade.detectMultiScale(
                    mouth_roi,
                    scaleFactor=1.7,
                    minNeighbors=22,
                    minSize=(int(fw * 0.30), int(fh * 0.10)),
                )
                smile = len(smiles) > 0

            # --- gozler: yuzun ust yarisinda ara ---
            eyes_y2 = int(fh * 0.60)
            eyes_roi = face_gray[0:eyes_y2, :]
            eyes_count = 0
            if eyes_roi.size > 0:
                eyes = self._eye_cascade.detectMultiScale(
                    eyes_roi,
                    scaleFactor=1.1,
                    minNeighbors=10,
                    minSize=(int(fw * 0.10), int(fh * 0.08)),
                )
                eyes_count = len(eyes)

            out.append(
                (
                    box,
                    {
                        "smile": bool(smile),
                        "eyes_open": eyes_count >= 1,
                    },
                )
            )
        return out

    def close(self):
        self._cascade = None
        self._smile_cascade = None
        self._eye_cascade = None


# ---------------------------------------------------------------------------
# Yuz takipcisi: kutulara stabil ID atar, isimlerin yuze yapismasini saglar.
# ---------------------------------------------------------------------------
class FaceTracker:
    """IoU + merkez-mesafe tabanli basit yuz takipcisi.
    - Her yeni yuze artan bir ID verir; ayni yuz hareket edince ID korunur.
    - Bir yuz uzun sure gorunmezse iz silinir.
    - Duygu icin yumusatma (frame-bazli oy) tutar.
    """

    def __init__(
        self,
        iou_threshold: float = 0.25,
        center_dist_ratio: float = 0.6,
        max_missing: int = 25,
    ):
        self._next_id = 1
        self._tracks: Dict[int, dict] = {}
        self._iou_threshold = iou_threshold
        self._center_dist_ratio = center_dist_ratio
        self._max_missing = max_missing

    @staticmethod
    def _iou(a: Box, b: Box) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _center(b: Box) -> Tuple[float, float]:
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    def update(
        self,
        observations: List[Tuple[Box, Dict[str, bool]]],
    ) -> List[Tuple[int, Box, str]]:
        """Yeni gozlemleri mevcut izlerle eslestirir.
        Doner: [(id, box, emotion), ...] -- gozlem sirasinda.
        emotion: 'guluyor' | 'notr' | 'uzgun'
        """
        det_boxes = [o[0] for o in observations]
        n_det = len(det_boxes)

        track_ids = list(self._tracks.keys())
        # Skor tablosu: her (track, det) cifti icin esleme skoru
        candidates: List[Tuple[float, int, int]] = []
        for ti in track_ids:
            t_box = self._tracks[ti]["bbox"]
            t_w = max(1, t_box[2] - t_box[0])
            tcx, tcy = self._center(t_box)
            for di, d_box in enumerate(det_boxes):
                iou = self._iou(t_box, d_box)
                if iou >= self._iou_threshold:
                    candidates.append((iou + 1.0, ti, di))  # IoU eslesmesi onceliklidir
                    continue
                dcx, dcy = self._center(d_box)
                dist = ((tcx - dcx) ** 2 + (tcy - dcy) ** 2) ** 0.5
                if dist < t_w * self._center_dist_ratio:
                    score = 1.0 - dist / (t_w * self._center_dist_ratio)
                    if score > 0.05:
                        candidates.append((score, ti, di))

        candidates.sort(reverse=True)

        used_tracks: set = set()
        used_dets: set = set()
        det_to_track: Dict[int, int] = {}
        for _score, ti, di in candidates:
            if ti in used_tracks or di in used_dets:
                continue
            det_to_track[di] = ti
            used_tracks.add(ti)
            used_dets.add(di)

        results: List[Tuple[int, Box, str]] = []
        for di, (box, feats) in enumerate(observations):
            if di in det_to_track:
                tid = det_to_track[di]
                tr = self._tracks[tid]
                tr["bbox"] = box
                tr["missing"] = 0
            else:
                tid = self._next_id
                self._next_id += 1
                tr = {
                    "bbox": box,
                    "missing": 0,
                    "smile_hits": 0,
                    "no_smile_hits": 0,
                    "eyes_closed_hits": 0,
                    "emotion": "notr",
                }
                self._tracks[tid] = tr

            self._update_emotion(tr, feats)
            results.append((tid, box, tr["emotion"]))

        # Eslesmeyen izlerin missing sayisini artir, gerekirse sil
        for tid in list(self._tracks.keys()):
            if tid in used_tracks:
                continue
            if any(d == tid for d in det_to_track.values()):
                continue
            self._tracks[tid]["missing"] += 1
            if self._tracks[tid]["missing"] > self._max_missing:
                del self._tracks[tid]

        return results

    @staticmethod
    def _update_emotion(track: dict, feats: Dict[str, bool]) -> None:
        # Smoothing: tekil karelerdeki gurultuyu yumusat
        if feats.get("smile"):
            track["smile_hits"] = min(8, track["smile_hits"] + 2)
            track["no_smile_hits"] = max(0, track["no_smile_hits"] - 1)
        else:
            track["smile_hits"] = max(0, track["smile_hits"] - 1)
            track["no_smile_hits"] = min(15, track["no_smile_hits"] + 1)

        if feats.get("eyes_open"):
            track["eyes_closed_hits"] = max(0, track["eyes_closed_hits"] - 2)
        else:
            track["eyes_closed_hits"] = min(15, track["eyes_closed_hits"] + 1)

        if track["smile_hits"] >= 3:
            track["emotion"] = "guluyor"
        elif track["eyes_closed_hits"] >= 8 and track["smile_hits"] == 0:
            # gozler uzun suredir kapali/sikilmis + gulumseme yok -> uzgun/aglyor tahmini
            track["emotion"] = "uzgun"
        elif track["no_smile_hits"] >= 4:
            track["emotion"] = "notr"

    def alive_ids(self) -> List[int]:
        return [tid for tid, tr in self._tracks.items() if tr["missing"] == 0]

    def all_ids(self) -> List[int]:
        return sorted(self._tracks.keys())

    def remove(self, tid: int) -> None:
        self._tracks.pop(tid, None)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
