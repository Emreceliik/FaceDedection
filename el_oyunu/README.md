# El Oyunu (Hand Gesture Playground)

Webcam onunde el hareketleriyle oynanan, **saf OpenCV** tabanli mini bir
uygulama. MediaPipe gerekmez. Kardes proje `top_oyunu`'nun yapisini
takip eder.

## Modlar

1. **Cizim** - Parmak ucunla havada cizim yap.
   - 1 parmak  -> ciz
   - 2 parmak  -> tasi (cizmez)
   - 4-5 parmak -> tum tuvali temizle
   - Yumruk    -> silgi
   - Renk paleti ust seritte; parmagini renk uzerinde yarim saniye tut, secilir.

2. **Balon Patlatma** - Asagidan yukselen balonlari elinle hizla patlat.
   - Combo carpani: ust uste vurus puani buyutur.
   - Balon ust serite ulasirsa can gider (3 can).
   - Skor 12'sinde seviye atlar, balonlar sik ve hizli olur.

3. **Parmak Sayma** - Gosterdigin parmak sayisini buyuk fontla yansitir.
   - Konturu, konveks gobegi ve parmak ucunu cizer.

## Kurulum

```bash
cd el_oyunu
pip install -r requirements.txt
python app.py
```

## Kullanim ipuclari

- Acilinca bir kamera sec ve **Baslat**'a bas.
- Algilama zayifsa ekranin ortasindaki **yesil kutuya** elini koy ve **C**
  tusuna bas. Boylece HSV ten araligin senin cilt tonuna ve isigina
  gore kalibre edilir.
- Tuslar: `Space` = sifirla, `M` = mod degistir, `C` = kalibrasyon,
  `Q` = cikis.
- Sag alttaki **Maske** kutusunu acarak detektorun ne gordugunu canli
  izleyebilirsin (siyah-beyaz pencere).

## Dosyalar

- `app.py` - Tkinter arayuz + ana dongu, mod gecisi.
- `hand_detector.py` - HSV ten + kontur + konveks defekt ile el ve parmak.
- `paint_game.py` - Cizim modu (parmak ucu fircasi, palet, silgi).
- `balloon_game.py` - Balon patlatma oyunu.
- `counter_mode.py` - Parmak sayma modu.
- `camera_utils.py` - Kullanilabilir kameralari listeler.

## Sinirlamalar

- Ten rengi tabanli oldugu icin arka planda ten tonunda objeler varsa
  yanilabilir. Ozellikle ahsap masa, krem duvar, ten tonunda kiyafet
  zorlayicidir. Bu durumda **Kalibre Et** kullanmak cok yardimci olur.
- Parmak sayma 5'e kadar yaklasik dogrulukla calisir; kamera kalitesine
  ve aydinlatmaya gore degisir. Daha iyi sonuc icin elini kameraya yakin
  ve duz arka plana karsi tut.
