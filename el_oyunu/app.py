"""
El Oyunu - Tkinter ana uygulamasi.

Modlar:
  1) Cizim       (parmak ucunla havada cizim)
  2) Balon Patlatma  (elinle balonlari patlat)
  3) Parmak Sayma    (kac parmak gosterdigini soyler)

El tespiti saf OpenCV ile (HSV ten rengi + kontur). Ten algisi farkli
isiklarda saglikli olsun diye "Kalibre Et" dugmesi var:
- Dugmeye basinca ekranin ortasindaki yesil dikdortgenden HSV
  ortalamasi alinir ve aralik dinamik olarak guncellenir.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

import cv2
from PIL import Image, ImageTk

from balloon_game import BalloonGame
from camera_utils import list_available_cameras
from counter_mode import CounterMode
from hand_detector import HandDetector
from paint_game import PaintGame


FRAME_W = 720
FRAME_H = 540

MODE_PAINT = "Cizim"
MODE_BALLOON = "Balon Patlatma"
MODE_COUNTER = "Parmak Sayma"
MODES = (MODE_PAINT, MODE_BALLOON, MODE_COUNTER)


class ElOyunuApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("El Oyunu - Hand Gesture Playground")
        self.root.geometry("980x720")
        self.root.minsize(820, 600)

        self.cap: cv2.VideoCapture | None = None
        self.detector = HandDetector()
        self.paint = PaintGame(width=FRAME_W, height=FRAME_H)
        self.balloon = BalloonGame(width=FRAME_W, height=FRAME_H)
        self.counter = CounterMode()

        self.running = False
        self.mode_var = tk.StringVar(value=MODE_PAINT)
        self.show_mask = tk.BooleanVar(value=False)
        self.show_overlay = tk.BooleanVar(value=True)
        self.calibrate_request = False

        self._build_ui()
        self._populate_cameras()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Key>", self._on_key)

    # ---------------- UI ----------------
    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Kamera:").pack(side=tk.LEFT, padx=(0, 6))
        self.cam_var = tk.StringVar()
        self.cam_combo = ttk.Combobox(
            top, textvariable=self.cam_var, width=14, state="readonly"
        )
        self.cam_combo.pack(side=tk.LEFT)
        ttk.Button(top, text="Yenile", command=self._populate_cameras).pack(
            side=tk.LEFT, padx=6
        )

        self.start_btn = ttk.Button(top, text="Baslat", command=self._toggle_run)
        self.start_btn.pack(side=tk.LEFT, padx=6)

        ttk.Label(top, text="Mod:").pack(side=tk.LEFT, padx=(12, 4))
        self.mode_combo = ttk.Combobox(
            top,
            textvariable=self.mode_var,
            values=MODES,
            width=18,
            state="readonly",
        )
        self.mode_combo.pack(side=tk.LEFT)
        self.mode_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_mode_change())

        ttk.Button(top, text="Sifirla (Space)", command=self._reset_current).pack(
            side=tk.LEFT, padx=6
        )

        ttk.Button(top, text="Kalibre Et (C)", command=self._request_calibration).pack(
            side=tk.LEFT, padx=6
        )

        ttk.Checkbutton(
            top, text="Maske", variable=self.show_mask
        ).pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(
            top, text="El Cizimi", variable=self.show_overlay
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
            "Cizim: 1 parmak = ciz, 2 parmak = tasi, >=4 parmak = temizle, yumruk = silgi. "
            "Balon: elinle balonlari patlat. "
            "Tuslar: Space=sifirla, M=mod degistir, C=kalibre et, Q=cikis."
        )
        ttk.Label(self.root, text=tip, foreground="#666").pack(pady=(0, 6))

    def _on_mode_change(self) -> None:
        self._reset_current()
        self.status_var.set(f"Mod: {self.mode_var.get()}")

    # ---------------- Kamera ----------------
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

    def _reset_current(self) -> None:
        mode = self.mode_var.get()
        if mode == MODE_PAINT:
            self.paint.reset()
        elif mode == MODE_BALLOON:
            self.balloon.reset()
        elif mode == MODE_COUNTER:
            self.counter.reset()

    def _request_calibration(self) -> None:
        if not self.running:
            messagebox.showinfo(
                "Kalibrasyon",
                "Once Baslat'a bas, sonra elini ekrandaki yesil kutuya getirip C'ye bas.",
            )
            return
        self.calibrate_request = True
        self.status_var.set("Elini yesil kutunun icine getir, kalibrasyon yapiliyor...")

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
        self.status_var.set(f"Kamera {idx} acik. Mod: {self.mode_var.get()}")
        self._reset_current()
        self._update_frame()

    def _stop(self) -> None:
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_btn.config(text="Baslat")
        self.status_var.set("Durduruldu.")
        self.video_label.config(image="")

    # ---------------- Dongu ----------------
    def _update_frame(self) -> None:
        if not self.running or self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.status_var.set("Kare okunamadi.")
            self.root.after(50, self._update_frame)
            return

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # kalibrasyon kutusu
        cal_box = self._calibration_box(w, h)
        if self.calibrate_request:
            ok = self.detector.calibrate_from_roi(frame, cal_box)
            self.calibrate_request = False
            if ok:
                self.status_var.set("Ten rengi kalibre edildi.")
            else:
                self.status_var.set("Kalibrasyon basarisiz.")

        hand = self.detector.detect(frame)

        mode = self.mode_var.get()
        if mode == MODE_PAINT:
            self.paint.step(frame, hand)
        elif mode == MODE_BALLOON:
            self.balloon.step(frame, hand)
        elif mode == MODE_COUNTER:
            self.counter.step(frame, hand)

        if self.show_overlay.get() and hand is not None and mode != MODE_COUNTER:
            HandDetector.draw_overlay(frame, hand)

        if self.show_mask.get() and hand is not None:
            mask_bgr = cv2.cvtColor(hand.mask, cv2.COLOR_GRAY2BGR)
            mask_small = cv2.resize(mask_bgr, (160, 120))
            frame[h - 130:h - 10, w - 170:w - 10] = mask_small
            cv2.rectangle(
                frame,
                (w - 170, h - 130),
                (w - 10, h - 10),
                (255, 255, 255),
                1,
            )

        # kalibrasyon kutusunu ciz (kullaniciya ipucu)
        if not self.detector.calibrated:
            self._draw_calibration_hint(frame, cal_box)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        lw = max(self.video_label.winfo_width(), 320)
        lh = max(self.video_label.winfo_height(), 240)
        img.thumbnail((lw, lh), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(img)
        self.video_label.config(image=self._tk_img)

        self.root.after(15, self._update_frame)

    def _calibration_box(self, w: int, h: int) -> tuple[int, int, int, int]:
        cw, ch = 140, 140
        cx = w // 2
        cy = h // 2 + 40
        return (cx - cw // 2, cy - ch // 2, cx + cw // 2, cy + ch // 2)

    def _draw_calibration_hint(
        self, frame_bgr, box: tuple[int, int, int, int]
    ) -> None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (60, 220, 60), 2)
        cv2.putText(
            frame_bgr,
            "Eli buraya getir + C'ye bas",
            (x1 - 20, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (60, 220, 60),
            2,
            cv2.LINE_AA,
        )

    # ---------------- Klavye / Kapanis ----------------
    def _on_key(self, event: tk.Event) -> None:
        ch = event.keysym.lower()
        if ch == "q":
            self._on_close()
        elif ch == "space":
            self._reset_current()
        elif ch == "c":
            self._request_calibration()
        elif ch == "m":
            i = MODES.index(self.mode_var.get())
            self.mode_var.set(MODES[(i + 1) % len(MODES)])
            self._on_mode_change()

    def _on_close(self) -> None:
        self._stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    ElOyunuApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
