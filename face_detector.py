import os
import shutil
import tempfile

import cv2


def _load_cascade_safely(filename: str) -> cv2.CascadeClassifier:
    """OpenCV FileStorage, Windows'ta non-ASCII yollarda hata veriyor.
    Cascade XML'ini ASCII-guvenli bir temp yola kopyalayip oradan yukluyoruz.
    """
    src = os.path.join(cv2.data.haarcascades, filename)
    if not os.path.isfile(src):
        raise RuntimeError(f"Cascade bulunamadi: {src}")

    # OpenCV FileStorage, Windows'ta ASCII disi yollarda basarisiz oluyor.
    # Yol tamamen ASCII ise dogrudan yukle, degilse temp'e kopyala.
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
    """OpenCV Haar cascade tabanli yuz yakalayici.
    Kareyi alir, (x1, y1, x2, y2) kutulari listesi dondurur.
    Mediapipe gerektirmez; opencv-python ile gelir.
    """

    def __init__(self, min_confidence: float = 0.6, model_selection: int = 0):
        # min_confidence ve model_selection, API uyumlulugu icin tutuluyor.
        # Haar'da dogrudan kullanilmaz; min_neighbors'a dolayli yansitiyoruz.
        self._min_confidence = float(min_confidence)
        self._cascade = _load_cascade_safely("haarcascade_frontalface_default.xml")

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # min_confidence yuksekse daha sert esikler kullan
        min_neighbors = 5 if self._min_confidence >= 0.6 else 3
        min_side = max(40, int(min(h, w) * 0.08))

        rects = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=min_neighbors,
            minSize=(min_side, min_side),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        faces = []
        for (x, y, bw, bh) in rects:
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(w - 1, int(x + bw))
            y2 = min(h - 1, int(y + bh))
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2, y2))

        # Soldan saga sirala -> kararli indeks
        faces.sort(key=lambda b: b[0])
        return faces

    def close(self):
        # Haar cascade icin kapatilacak kaynak yok; API uyumlulugu.
        self._cascade = None
