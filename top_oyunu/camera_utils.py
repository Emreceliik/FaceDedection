import cv2


def list_available_cameras(max_index: int = 5):
    """0..max_index-1 arasindaki kamera indekslerini dener, calisanlari dondurur."""
    available = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # Windows icin DSHOW daha hizli
        if cap is not None and cap.isOpened():
            ok, _ = cap.read()
            if ok:
                available.append(idx)
            cap.release()
    return available
