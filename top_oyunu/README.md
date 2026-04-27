# Top Oyunu

Kamera onunde sanal bir top sektirme oyunu. Kafanla **kafalayabilir**,
ellerini sallayarak topa vurabilirsin. Top dususte yer cekimi etkisindedir;
yere duserse skor sifirlanir, en iyi skor saklanir.

## Nasil calisir?

- **Kafa carpismasi**: OpenCV Haar cascade ile yuz kutusu bulunur. Top
  bu kutunun ust kenarina dustugunde sekme uygulanir; kenardan kafalarsan
  top yan tarafa, ortadan kafalarsan dik yukari gider.
- **El carpismasi**: Ardisik iki kareyi karsilastirip "hareket eden
  pikseller" cikarilir (frame differencing). Yuz bolgesi bu maskeden
  silinir, geriye kalan hareket "el" sayilir. Topun etrafinda yeterli
  hareket varsa top, hareketin geldigi yonun tersine itilir.
- **Fizik**: Yer cekimi, hava surtunmesi, hiz limiti ve duvar sekmeleri
  basit Newton mantigi ile uygulanir.

## Kurulum (Windows / PowerShell)

Proje klasorunde:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`Activate.ps1` izin hatasi verirse bir kerelik:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## Calistirma

```powershell
python app.py
```

1. Acilan pencerede ustteki menuden kamerayi sec (yoksa **Yenile**).
2. **Baslat**'a tikla; ekranda mavi-beyaz top dusmeye basla.
3. Kafanla / ellerinle topa vurarak sektirmeyi dene. Sol ust kosede
   skor ve en iyi skor gorunur.
4. Top yere duserse otomatik yeniden baslar. **Space** ile elle de
   yeniden baslatabilirsin.

## Klavye

- `Space` -> Yeni oyun
- `Q` -> Cikis

## Dosyalar

- `app.py` - Tkinter arayuz ve ana dongu
- `ball_game.py` - Top fizigi ve carpisma mantigi
- `face_detector.py` - Yuz (kafa) tespiti
- `camera_utils.py` - Kullanilabilir kameralari listeleme
- `requirements.txt` - Bagimliliklar

## Olasi Gelistirmeler (ogrenci alistirmalari)

- Top yerine farkli "topcuk" gorselleri (resim) kullanmak.
- Skor 10'a varinca seviye atlamak ve yer cekimini artirmak.
- Mediapipe ile gercek el iskelet noktalarini kullanip parmak ucu
  carpismasi yapmak.
- Iki kisilik mod: iki yuz tespit edilirse top araya dusunce hangi
  oyuncuya puan eklenecegini belirlemek.
- Skoru ekrana grafik olarak (zaman icinde) cizmek.
