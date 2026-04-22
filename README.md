# Yuz Yakala ve Isimlendir (Cocuklar icin)

Kamerada yuz yakalar, uzerine kutu cizer ve verdigin isim etiketini
ekranda gosterir. Tkinter tabanli kucuk bir arayuz vardir:
- Kamera secimi (Yenile + Baslat/Durdur)
- Sag panelden yakalanan yuzler: **Ekle / Duzenle / Sil / Sifirla**
- Klavye kisayollari: `1-9` hizli isim atama, `R` sifirla, `Q` cikis

> Not: Otomatik yuz **tanima** yoktur. Yuzler ekranda soldan saga
> `1, 2, 3 ...` diye siralanir ve o indekse isim atarsin. Yuz
> cercevede yer degistirirse etiket baska bir yuze gecebilir -
> demo icin bu yeterlidir.

## Kurulum

PowerShell'de proje klasorunde:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Eger `Activate.ps1` calismazsa (PowerShell izin hatasi):

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## Calistirma

```powershell
python app.py
```

1. Ustteki acilir menuden kamerayi sec (yoksa **Yenile**'ye bas).
2. **Baslat**'a tikla.
3. Bir yuz goruldugunde sag panelde `1. -`, `2. -` gibi cikar.
4. Listeden yuzu sec -> **Isim Ekle/Ata** (veya cift tikla).
5. Ustteki etiket artik ismi gosterir.

## Dosyalar

- `app.py` - Tkinter arayuzu ve ana dongu
- `face_detector.py` - MediaPipe yuz yakalama
- `camera_utils.py` - Kullanilabilir kameralari listeleme
- `requirements.txt` - Bagimliliklar

## Olasi Gelistirmeler

- Gercek yuz **tanima** (ornegin `face_recognition` kutuphanesi) ile
  Ahmet'i yuzunden otomatik bul.
- Isimleri JSON'a kaydedip sonraki acilista yukle.
- Tum govdeyi yakalama (person detection) modu.
