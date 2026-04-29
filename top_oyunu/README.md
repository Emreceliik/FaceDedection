# Top Oyunu (Eglenceli Surum)

Kamera onunde sanal bir top sektirme oyunu. Kafanla **kafalayabilir**,
ellerini sallayarak topa vurabilirsin. Top dususte yer cekimi etkisindedir;
yere duserse skor sifirlanir, en iyi skor saklanir.

Eglenceli eklentiler:

- **Combo carpani**: Ust uste vuruslar puani 2x, 3x, 4x katlar.
- **Seviye sistemi**: Her 10 puanda LEVEL UP, top hizlanir ve rengi degisir.
- **Parcacik patlamalari** ve topun arkasinda **renkli iz**.
- **Yuz filtreleri**: F tusu ile sapka, gozluk, biyik, taç, palyaco
  burnu, super kahraman maskesi gibi farkli filtreler.
- **Ses efektleri** (Windows yerlesik `winsound` ile, ek bagimlilik yok).

## Nasil calisir?

- **Kafa carpismasi**: OpenCV Haar cascade ile yuz kutusu bulunur. Top
  bu kutunun ust kenarina dustugunde sekme uygulanir; kenardan kafalarsan
  top yan tarafa, ortadan kafalarsan dik yukari gider.
- **El carpismasi**: Ardisik iki kareyi karsilastirip "hareket eden
  pikseller" cikarilir (frame differencing). Yuz bolgesi bu maskeden
  silinir, geriye kalan hareket "el" sayilir. Topun etrafinda yeterli
  hareket varsa top, hareketin geldigi yonun tersine itilir.
- **Fizik**: Yer cekimi, hava surtunmesi, hiz limiti ve duvar sekmeleri
  basit Newton mantigi ile uygulanir. Seviye yukseldikce yer cekimi
  ve hiz limiti %8 artar.
- **Filtreler**: Yuz kutusunun oranlarina gore goz / agiz / sac
  bolgeleri tahmin edilip cv2 cizimleri ile uzerine sekiller eklenir.
- **Sesler**: `winsound.Beep` ile kisa frekans dizileri calinir; her
  ses kendi thread'inde calistigi icin oyun donmez.

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
2. **Baslat**'a tikla; ekranda renkli top dusmeye basla.
3. Kafanla / ellerinle topa vurarak sektirmeyi dene. Sol ust kosede
   skor, seviye, combo ve en iyi skor gorunur.
4. Top yere duserse otomatik yeniden baslar.

## Klavye / Butonlar

- `Space` -> Yeni oyun
- `F`     -> Yuz filtresi degistir (Sihirbaz / Hipster / Palyaco /
             Kraliyet / Gangster / Suuper Kahraman / Yok)
- `S`     -> Sesi ac / kapa
- `Q`     -> Cikis

## Dosyalar

- `app.py`           - Tkinter arayuz ve ana dongu
- `ball_game.py`     - Top fizigi, combo, seviye, parcacik, iz
- `face_detector.py` - Yuz (kafa) tespiti (Haar cascade)
- `face_filters.py`  - Sapka / gozluk / biyik gibi yuz filtreleri
- `sounds.py`        - Basit ses efektleri (winsound)
- `camera_utils.py`  - Kullanilabilir kameralari listeleme
- `requirements.txt` - Bagimliliklar

## Olasi Gelistirmeler (ogrenci alistirmalari)

- Foto kabini: gulumsedikce otomatik foto cek, klasore kaydet.
- Cift kisilik mod: iki yuz tespit edilirse ekran ikiye bolunur,
  hangi tarafta top dustuyse o oyuncu can kaybetsin.
- Mediapipe ile gercek el iskelet noktalari kullanip parmak ucu
  carpismasi ekle.
- Skoru zaman icinde grafige cizen mini panel.
- Kendi filtrenizi yapin: `face_filters.py` icine yeni bir cizim
  fonksiyonu ekleyip `apply_filter` icine baglayin.
