"""
Top Oyunu - Sadelestirilmis surum.
Kamerada bir top vardir; kafanla ve ellerinle topa vurarak sektirebilirsin.
- Yer cekimi vardir, top dususe gecer.
- Yuz tespiti ile kafa carpismasi yapilir.
- Hareket maskesi ile el carpismasi yapilir (kameraya el sallayinca top sekiyor).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

import cv2
from PIL import Image, ImageTk

from ball_game import BallGame
from camera_utils import list_available_cameras
from face_detector import FaceDetector


FRAME_W = 720
FRAME_H = 540


class TopOyunuApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Top Oyunu")
        self.root.geometry("900x620")
        self.root.minsize(800, 560)

        self.cap: cv2.VideoCapture | None = None
        self.detector = FaceDetector()
        self.game = BallGame(width=FRAME_W, height=FRAME_H)
        self.running = False

        self._build_ui()
        self._populate_cameras()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Key>", self._on_key)

    # ---------- UI ----------
    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Kamera:").pack(side=tk.LEFT, padx=(0, 6))

        self.cam_var = tk.StringVar()
        self.cam_combo = ttk.Combobox(
            top, textvariable=self.cam_var, width=18, state="readonly"
        )
        self.cam_combo.pack(side=tk.LEFT)

        ttk.Button(top, text="Yenile", command=self._populate_cameras).pack(
            side=tk.LEFT, padx=6
        )

        self.start_btn = ttk.Button(top, text="Baslat", command=self._toggle_run)
        self.start_btn.pack(side=tk.LEFT, padx=6)

        ttk.Button(top, text="Yeni Oyun (Space)", command=self.game.reset).pack(
            side=tk.LEFT, padx=6
        )

        self.status_var = tk.StringVar(value="Hazir.")
        ttk.Label(top, textvariable=self.status_var, foreground="#555").pack(
            side=tk.LEFT, padx=12
        )

        video_frame = ttk.Frame(self.root, relief="sunken", borderwidth=1)
        video_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.video_label = ttk.Label(video_frame, anchor="center")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        tip = (
            "Kafanla topu kafala, ellerini sallayarak topa vur. "
            "Top yere dusunce skor sifirlanir. "
            "Tuslar: Space = yeni oyun, Q = cikis."
        )
        ttk.Label(self.root, text=tip, foreground="#666").pack(pady=(0, 6))

    # ---------- Kamera ----------
    def _populate_cameras(self) -> None:
        self.status_var.set("Kameralar taraniyor...")
        self.root.update_idletasks()
        cams = list_available_cameras(max_index=5)
        if not cams:
            self.cam_combo["values"] = []
            self.cam_var.set("")
            self.status_var.set("Kamera bulunamadi.")
            return
        values = [f"Kamera {i}" for i in cams]
        self.cam_combo["values"] = values
        self.cam_var.set(values[0])
        self.status_var.set(f"{len(cams)} kamera bulundu.")

    def _selected_camera_index(self) -> int | None:
        v = self.cam_var.get()
        if not v:
            return None
        try:
            return int(v.split()[-1])
        except ValueError:
            return None

    def _toggle_run(self) -> None:
        if self.running:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        idx = self._selected_camera_index()
        if idx is None:
            messagebox.showwarning("Uyari", "Once bir kamera sec.")
            return
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("Hata", f"Kamera {idx} acilamadi.")
            self.cap = None
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.running = True
        self.start_btn.config(text="Durdur")
        self.status_var.set(f"Kamera {idx} acik. Topu sektir!")
        self.game.reset()
        self._update_frame()

    def _stop(self) -> None:
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_btn.config(text="Baslat")
        self.status_var.set("Durduruldu.")
        self.video_label.config(image="")

    # ---------- Dongu ----------
    def _update_frame(self) -> None:
        if not self.running or self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.status_var.set("Kare okunamadi.")
            self.root.after(50, self._update_frame)
            return

        frame = cv2.flip(frame, 1)  # ayna goruntu
        face_boxes = self.detector.detect(frame)
        self.game.step(frame, face_boxes)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        w = max(self.video_label.winfo_width(), 320)
        h = max(self.video_label.winfo_height(), 240)
        img.thumbnail((w, h), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(img)
        self.video_label.config(image=self._tk_img)

        self.root.after(15, self._update_frame)

    # ---------- Klavye / Kapanis ----------
    def _on_key(self, event: tk.Event) -> None:
        ch = event.keysym.lower()
        if ch == "q":
            self._on_close()
        elif ch == "space":
            self.game.reset()

    def _on_close(self) -> None:
        self._stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    TopOyunuApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
