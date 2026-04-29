"""
Top Oyunu - Sadelestirilmis surum + eglenceli eklentiler.
Kamerada bir top vardir; kafanla ve ellerinle topa vurarak sektirebilirsin.
- Yer cekimi vardir, top dususe gecer.
- Yuz tespiti ile kafa carpismasi yapilir.
- Hareket maskesi ile el carpismasi yapilir (kameraya el sallayinca top sekiyor).
- Ust uste vuruslar combo carpani verir, belirli skorda LEVEL UP olur.
- Topun arkasinda renkli iz, vurulunca parcacik patlamalari.
- F tusuyla yuze sapka / gozluk / biyik gibi eglenceli filtreler eklenir.
- Ses efektleri (Windows winsound) acik/kapali yapilabilir.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

import cv2
from PIL import Image, ImageTk

from ball_game import BallGame
from camera_utils import list_available_cameras
from face_detector import FaceDetector
from face_filters import NUM_FILTERS, apply_filter, filter_name
from sounds import SoundFx


FRAME_W = 720
FRAME_H = 540


class TopOyunuApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Top Oyunu - Eglenceli Surum")
        self.root.geometry("960x680")
        self.root.minsize(820, 600)

        self.cap: cv2.VideoCapture | None = None
        self.detector = FaceDetector()
        self.sound = SoundFx(enabled=True)
        self.game = BallGame(width=FRAME_W, height=FRAME_H, sound=self.sound)
        self.running = False
        self.filter_index = 0
        self.sound_enabled = tk.BooleanVar(value=True)

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

        ttk.Button(top, text="Yeni Oyun (Space)", command=self._new_game).pack(
            side=tk.LEFT, padx=6
        )

        self.filter_btn = ttk.Button(
            top,
            text=f"Filtre: {filter_name(self.filter_index)} (F)",
            command=self._next_filter,
        )
        self.filter_btn.pack(side=tk.LEFT, padx=6)

        ttk.Checkbutton(
            top,
            text="Ses",
            variable=self.sound_enabled,
            command=self._toggle_sound,
        ).pack(side=tk.LEFT, padx=6)

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
            "Ust uste vuruslarda combo carpani buyur. "
            "Tuslar: Space = yeni oyun, F = filtre degistir, S = sesi ac/kapa, Q = cikis."
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

    def _new_game(self) -> None:
        self.game.reset()
        self.status_var.set("Yeni oyun!")

    def _next_filter(self) -> None:
        self.filter_index = (self.filter_index + 1) % NUM_FILTERS
        name = filter_name(self.filter_index)
        self.filter_btn.config(text=f"Filtre: {name} (F)")
        self.status_var.set(f"Filtre: {name}")

    def _toggle_sound(self) -> None:
        self.sound.set_enabled(self.sound_enabled.get())
        self.status_var.set(
            "Ses acik." if self.sound_enabled.get() else "Ses kapali."
        )

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

        if self.filter_index > 0:
            apply_filter(frame, face_boxes, self.filter_index)

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
            self._new_game()
        elif ch == "f":
            self._next_filter()
        elif ch == "s":
            self.sound_enabled.set(not self.sound_enabled.get())
            self._toggle_sound()

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
