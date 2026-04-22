import cv2
import mediapipe as mp


class FaceDetector:
    """MediaPipe tabanli yuz yakalayici. Kareyi alir, kutular listesi dondurur."""

    def __init__(self, min_confidence: float = 0.6, model_selection: int = 0):
        self._mp_face = mp.solutions.face_detection
        self._detector = self._mp_face.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_confidence,
        )

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._detector.process(rgb)

        faces = []
        if result.detections:
            for det in result.detections:
                bbox = det.location_data.relative_bounding_box
                x1 = max(0, int(bbox.xmin * w))
                y1 = max(0, int(bbox.ymin * h))
                x2 = min(w - 1, x1 + int(bbox.width * w))
                y2 = min(h - 1, y1 + int(bbox.height * h))
                if x2 > x1 and y2 > y1:
                    faces.append((x1, y1, x2, y2))

        # Soldan saga sirala -> kararli indeks
        faces.sort(key=lambda b: b[0])
        return faces

    def close(self):
        try:
            self._detector.close()
        except Exception:
            pass
