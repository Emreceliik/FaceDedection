"""
Cocuklara gostermek icin: kamerada yuz yakalayip ustune isim etiketi yazar.
- Sol ust: kamera secimi + Baslat/Durdur
- Sag panel: yakalanan yuzler; isim Ekle / Duzenle / Sil
Not: Otomatik yuz tanima yoktur; yuzler soldan saga sirali indekslenir
      ve o indekse isim atarsin.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from typing import Dict, List, Tuple

import cv2
from PIL import Image, ImageTk

from camera_utils import list_available_cameras
from face_detector import FaceDetector


BOX_COLOR = (0, 255, 0)       # BGR
BOX_COLOR_NAMED = (0, 200, 255)
TEXT_COLOR = (255, 255, 255)
FRAME_W = 720
FRAME_H = 540


class FaceLabelApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Yuz Yakala ve Isimlendir")
        self.root.geometry("1080x620")
        self.root.minsize(900, 560)

        self.cap: cv2.VideoCapture | None = None
        self.detector = FaceDetector(min_confidence=0.6)
        self.running = False
        self.current_faces: List[Tuple[int, int, int, int]] = []
        self.name_by_index: Dict[int, str] = {}

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

        # Sag panel: isim listesi
        side = ttk.Frame(main, padding=(10, 0, 0, 0))
        side.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(side, text="Yakalanan Yuzler", font=("Segoe UI", 11, "bold")).pack(
            anchor="w"
        )
        ttk.Label(
            side,
            text="(Soldan saga 1, 2, 3 ... olarak numaralandirilir)",
            foreground="#666",
        ).pack(anchor="w", pady=(0, 6))

        self.listbox = tk.Listbox(side, width=28, height=18, activestyle="dotbox")
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
            "Ipucu: Listede bir satira cift tiklayarak da\n"
            "ismi duzenleyebilirsin. 1-9 tuslari hizli atama."
        )
        ttk.Label(side, text=tip, foreground="#666", justify="left").pack(
            anchor="w", pady=(6, 0)
        )

        # Klavye kisayollari
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
        faces = self.detector.detect(frame)
        self.current_faces = faces

        for idx, (x1, y1, x2, y2) in enumerate(faces):
            name = self.name_by_index.get(idx)
            label = name if name else f"Yuz {idx + 1}"
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

        # Video alanina sigdir
        w = max(self.video_label.winfo_width(), 320)
        h = max(self.video_label.winfo_height(), 240)
        img.thumbnail((w, h), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(img)
        self.video_label.config(image=self._tk_img)

        self.root.after(15, self._update_frame)

    # ---------- Liste / Isim ----------
    def _refresh_listbox(self) -> None:
        self.listbox.delete(0, tk.END)
        for i in range(len(self.current_faces)):
            name = self.name_by_index.get(i, "-")
            self.listbox.insert(tk.END, f"{i + 1}. {name}")

    def _selected_face_index(self) -> int | None:
        sel = self.listbox.curselection()
        if not sel:
            return None
        return sel[0]

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
        i = self._selected_face_index()
        if i is None:
            i = 0  # secim yoksa ilk yuze ata
        name = self._ask_name(self.name_by_index.get(i, ""))
        if name:
            self.name_by_index[i] = name

    def _edit_name(self) -> None:
        i = self._selected_face_index()
        if i is None:
            messagebox.showinfo("Bilgi", "Listeden bir yuz sec.")
            return
        name = self._ask_name(self.name_by_index.get(i, ""))
        if name is not None:
            if name == "":
                self.name_by_index.pop(i, None)
            else:
                self.name_by_index[i] = name

    def _delete_name(self) -> None:
        i = self._selected_face_index()
        if i is None:
            messagebox.showinfo("Bilgi", "Listeden bir yuz sec.")
            return
        self.name_by_index.pop(i, None)

    def _reset_names(self) -> None:
        if messagebox.askyesno("Onay", "Tum isimler silinsin mi?"):
            self.name_by_index.clear()

    # ---------- Klavye ----------
    def _on_key(self, event: tk.Event) -> None:
        ch = event.keysym.lower()
        if ch == "q":
            self._on_close()
            return
        if ch == "r":
            self._reset_names()
            return
        if len(event.char) == 1 and event.char.isdigit() and event.char != "0":
            face_idx = int(event.char) - 1
            if face_idx < len(self.current_faces):
                self.listbox.selection_clear(0, tk.END)
                self.listbox.selection_set(face_idx)
                name = self._ask_name(self.name_by_index.get(face_idx, ""))
                if name:
                    self.name_by_index[face_idx] = name

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
