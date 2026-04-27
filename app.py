"""
Cocuklara gostermek icin: kamerada yuz yakalayip ustune isim + duygu yazar.
- Sol ust: kamera secimi + Baslat/Durdur
- Sag panel: yakalanan yuzler; isim Ekle / Duzenle / Sil
- Her yuze stabil bir ID atanir (tracker), isim ID'ye bagli kalir.
- Yuzun gulumseyip gulumsemedigine ve gozlerinin kapalı olup olmadigina
  bakarak "Guluyor", "Notr" veya "Uzgun" etiketi gosterilir.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from typing import Dict, List, Tuple

import cv2
from PIL import Image, ImageTk

from ball_game import BallGame
from camera_utils import list_available_cameras
from face_detector import FaceDetector, FaceTracker


BOX_COLOR = (0, 255, 0)         # BGR - isimsiz
BOX_COLOR_NAMED = (0, 200, 255) # BGR - isimli
TEXT_COLOR = (255, 255, 255)
FRAME_W = 720
FRAME_H = 540

EMOTION_LABEL = {
    "guluyor": "Guluyor :)",
    "notr": "Notr",
    "uzgun": "Uzgun :(",
}


class FaceLabelApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Yuz Yakala ve Isimlendir")
        self.root.geometry("1080x620")
        self.root.minsize(900, 560)

        self.cap: cv2.VideoCapture | None = None
        self.detector = FaceDetector(min_confidence=0.6)
        self.tracker = FaceTracker(iou_threshold=0.25, max_missing=25)
        self.ball_game = BallGame(width=FRAME_W, height=FRAME_H)
        self.game_mode = False
        self.running = False

        # Su anki karedeki yuzler: (track_id, box, emotion)
        self.current_faces: List[Tuple[int, Tuple[int, int, int, int], str]] = []
        # Isimler artik track ID'sine bagli
        self.name_by_id: Dict[int, str] = {}
        # Listbox satir indeksi -> track_id eslemesi
        self._listbox_ids: List[int] = []

        self._build_ui()
        self._populate_cameras()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

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

        self.game_btn = ttk.Button(
            top, text="Top Oyunu: Kapali", command=self._toggle_game
        )
        self.game_btn.pack(side=tk.LEFT, padx=6)

        self.status_var = tk.StringVar(value="Hazir.")
        ttk.Label(top, textvariable=self.status_var, foreground="#555").pack(
            side=tk.LEFT, padx=12
        )

        main = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        main.pack(fill=tk.BOTH, expand=True)

        # Video alani
        video_frame = ttk.Frame(main, relief="sunken", borderwidth=1)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.video_label = ttk.Label(video_frame, anchor="center")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Sag panel
        side = ttk.Frame(main, padding=(10, 0, 0, 0))
        side.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(side, text="Yakalanan Yuzler", font=("Segoe UI", 11, "bold")).pack(
            anchor="w"
        )
        ttk.Label(
            side,
            text="(Her yuze stabil ID verilir; isim o yuze yapisir.)",
            foreground="#666",
        ).pack(anchor="w", pady=(0, 6))

        self.listbox = tk.Listbox(side, width=32, height=18, activestyle="dotbox")
        self.listbox.pack(fill=tk.Y, expand=False)
        self.listbox.bind("<Double-Button-1>", lambda _e: self._edit_name())

        btns = ttk.Frame(side)
        btns.pack(fill=tk.X, pady=8)
        ttk.Button(btns, text="Isim Ekle/Ata", command=self._add_name).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(btns, text="Duzenle", command=self._edit_name).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(btns, text="Sil", command=self._delete_name).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(btns, text="Tumunu Sifirla", command=self._reset_names).pack(
            fill=tk.X, pady=2
        )

        tip = (
            "Ipucu: Listede bir satira cift tiklayarak\n"
            "ismi duzenleyebilirsin. 1-9 tuslari listedeki\n"
            "ilgili siradaki yuze hizli isim atar."
        )
        ttk.Label(side, text=tip, foreground="#666", justify="left").pack(
            anchor="w", pady=(6, 0)
        )

        self.root.bind("<Key>", self._on_key)

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

    def _toggle_game(self) -> None:
        self.game_mode = not self.game_mode
        if self.game_mode:
            self.ball_game.reset()
            self.game_btn.config(text="Top Oyunu: Acik")
            self.status_var.set("Top oyunu acik. Kafa ve ellerinle topu sektir!")
        else:
            self.game_btn.config(text="Top Oyunu: Kapali")
            self.status_var.set("Top oyunu kapatildi.")

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
        self.status_var.set(f"Kamera {idx} acik.")
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

        frame = cv2.flip(frame, 1)  # ayna
        observations = self.detector.analyze(frame)
        tracked = self.tracker.update(observations)
        self.current_faces = tracked
        face_boxes = [box for _tid, box, _emo in tracked]

        if self.game_mode:
            # Oyun modu: yuz kutularini cizmiyoruz; sadece top oynaniyor
            self.ball_game.step(frame, face_boxes)
        else:
            for tid, (x1, y1, x2, y2), emotion in tracked:
                name = self.name_by_id.get(tid)
                base = name if name else f"Yuz #{tid}"
                emo_text = EMOTION_LABEL.get(emotion, "")
                label = f"{base} | {emo_text}" if emo_text else base
                color = BOX_COLOR_NAMED if name else BOX_COLOR

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                ty = max(0, y1 - th - 8)
                cv2.rectangle(
                    frame, (x1, ty), (x1 + tw + 10, ty + th + 8), color, -1
                )
                cv2.putText(
                    frame,
                    label,
                    (x1 + 5, ty + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    TEXT_COLOR,
                    2,
                    cv2.LINE_AA,
                )

        self._refresh_listbox()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        w = max(self.video_label.winfo_width(), 320)
        h = max(self.video_label.winfo_height(), 240)
        img.thumbnail((w, h), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(img)
        self.video_label.config(image=self._tk_img)

        self.root.after(15, self._update_frame)

    # ---------- Liste / Isim ----------
    def _refresh_listbox(self) -> None:
        # Listbox'i ID sirasi ile yeniliyoruz; secimi koruyoruz
        prev_selected_id: int | None = None
        sel = self.listbox.curselection()
        if sel and sel[0] < len(self._listbox_ids):
            prev_selected_id = self._listbox_ids[sel[0]]

        self.listbox.delete(0, tk.END)
        # Sadece su an gorunen yuzleri ID'ye gore listele
        ordered = sorted(self.current_faces, key=lambda t: t[0])
        self._listbox_ids = [tid for tid, _box, _emo in ordered]

        for tid, _box, emotion in ordered:
            name = self.name_by_id.get(tid, "-")
            emo_text = EMOTION_LABEL.get(emotion, "")
            self.listbox.insert(tk.END, f"#{tid}  {name}   [{emo_text}]")

        if prev_selected_id is not None and prev_selected_id in self._listbox_ids:
            row = self._listbox_ids.index(prev_selected_id)
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(row)

    def _selected_track_id(self) -> int | None:
        sel = self.listbox.curselection()
        if not sel:
            return None
        row = sel[0]
        if row < 0 or row >= len(self._listbox_ids):
            return None
        return self._listbox_ids[row]

    def _ask_name(self, initial: str = "") -> str | None:
        name = simpledialog.askstring(
            "Isim", "Isim gir:", initialvalue=initial, parent=self.root
        )
        if name is None:
            return None
        return name.strip()

    def _add_name(self) -> None:
        if not self.current_faces:
            messagebox.showinfo("Bilgi", "Once kamerada bir yuz yakalanmali.")
            return
        tid = self._selected_track_id()
        if tid is None:
            # Secim yoksa: ismi olmayan ilk yuze ata; yoksa kullaniciya secmesini soyle
            unnamed = [
                t for t, _b, _e in sorted(self.current_faces, key=lambda x: x[0])
                if t not in self.name_by_id
            ]
            if not unnamed:
                messagebox.showinfo(
                    "Bilgi",
                    "Listeden hangi yuze isim atayacagini sec.\n"
                    "(Tum yuzlerin zaten ismi var.)",
                )
                return
            tid = unnamed[0]
        name = self._ask_name(self.name_by_id.get(tid, ""))
        if name:
            self.name_by_id[tid] = name

    def _edit_name(self) -> None:
        tid = self._selected_track_id()
        if tid is None:
            messagebox.showinfo("Bilgi", "Listeden bir yuz sec.")
            return
        name = self._ask_name(self.name_by_id.get(tid, ""))
        if name is not None:
            if name == "":
                self.name_by_id.pop(tid, None)
            else:
                self.name_by_id[tid] = name

    def _delete_name(self) -> None:
        tid = self._selected_track_id()
        if tid is None:
            messagebox.showinfo("Bilgi", "Listeden bir yuz sec.")
            return
        self.name_by_id.pop(tid, None)

    def _reset_names(self) -> None:
        if messagebox.askyesno("Onay", "Tum isimler silinsin mi?"):
            self.name_by_id.clear()
            self.tracker.reset()

    # ---------- Klavye ----------
    def _on_key(self, event: tk.Event) -> None:
        ch = event.keysym.lower()
        if ch == "q":
            self._on_close()
            return
        if ch == "g":
            self._toggle_game()
            return
        if ch == "space" and self.game_mode:
            self.ball_game.reset()
            return
        if ch == "r":
            self._reset_names()
            return
        if len(event.char) == 1 and event.char.isdigit() and event.char != "0":
            row = int(event.char) - 1
            if 0 <= row < len(self._listbox_ids):
                tid = self._listbox_ids[row]
                self.listbox.selection_clear(0, tk.END)
                self.listbox.selection_set(row)
                name = self._ask_name(self.name_by_id.get(tid, ""))
                if name:
                    self.name_by_id[tid] = name

    # ---------- Kapanis ----------
    def _on_close(self) -> None:
        self._stop()
        try:
            self.detector.close()
        except Exception:
            pass
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    FaceLabelApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
