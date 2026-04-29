"""
Basit ses efektleri.
Windows'ta yerlesik 'winsound' modulunu kullanir; ek bagimlilik gerekmez.
Ses calmasi tkinter dongusunu kilitlemesin diye kisa thread'lerde calisir.
"""

from __future__ import annotations

import threading
from typing import Iterable, Tuple

try:
    import winsound  # type: ignore
    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False


Note = Tuple[int, int]  # (frekans Hz, sure ms)


def _play_notes(notes: Iterable[Note]) -> None:
    if not _HAS_WINSOUND:
        return
    for freq, dur in notes:
        try:
            winsound.Beep(int(freq), int(dur))
        except Exception:
            return


class SoundFx:
    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled and _HAS_WINSOUND)

    def set_enabled(self, value: bool) -> None:
        self.enabled = bool(value and _HAS_WINSOUND)

    def _play(self, notes: Iterable[Note]) -> None:
        if not self.enabled:
            return
        notes = tuple(notes)
        threading.Thread(target=_play_notes, args=(notes,), daemon=True).start()

    def hit(self, combo: int = 0) -> None:
        base = 700 + min(combo, 10) * 60
        self._play([(base, 35)])

    def combo_break(self) -> None:
        self._play([(500, 40), (350, 60)])

    def level_up(self) -> None:
        self._play([(660, 70), (880, 70), (1100, 90), (1320, 130)])

    def fall(self) -> None:
        self._play([(420, 90), (300, 120), (200, 180)])

    def start(self) -> None:
        self._play([(660, 70), (990, 90)])
