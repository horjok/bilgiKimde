"""
Teshis: kisi_ayrilirsa() neden bos donuyor + bus_factor "0 kisi" meselesi.
Sadece teshis - ana dosyaya (yatkinlik_motoru.py) dokunmaz, duzeltme yapmaz.
Model/embedding YUKLENMEZ; sadece JSON okur, aninda calisir.
"""

import json
from pathlib import Path

# Veri yolunu script'in kendi konumuna gore coz -> CWD'den bagimsiz, saglam.
# (data/ klasoru capstone'un icinde; '../data' DEGIL.)
DATA = Path(__file__).parent / "data"
PROFIL_YOLU = DATA / "profiller_zengin.json"
HAVUZ_YOLU = DATA / "yetkinlik_havuzu.json"


def yukle(yol):
    with open(yol, encoding="utf-8") as f:
        return json.load(f)


def kac_kisi_biliyor(profiller, yad):
    """yad yetkinligini bilen kisilerin isim listesi."""
    return [p["kisi"] for p in profiller
            if any(y["ad"] == yad for y in p["yetkinlikler"])]


def main():
    print(f"Profil dosyasi : {PROFIL_YOLU}")
    print(f"Dosya var mi   : {PROFIL_YOLU.exists()}")

    profiller = yukle(PROFIL_YOLU)
    print(f"Toplam kisi    : {len(profiller)}\n")

    # 2) Ilk kisinin adi ve yetkinlikleri
    ilk = profiller[0]
    ilk_yetkinlikler = [y["ad"] for y in ilk["yetkinlikler"]]
    print(f"ILK KISI: {ilk['kisi']}")
    print(f"  yetkinlik sayisi: {len(ilk_yetkinlikler)}")
    print(f"  yetkinlikleri   : {', '.join(ilk_yetkinlikler)}\n")

    # 3) Ilk kisinin her yetkinligini KAC kisi biliyor (isimlerle) + kisi haric kalan
    print("--- Ilk kisinin yetkinliklerini kim biliyor? ---")
    for yad in ilk_yetkinlikler:
        bilenler = kac_kisi_biliyor(profiller, yad)
        baskalari = [ad for ad in bilenler if ad != ilk["kisi"]]
        print(f"[{yad}] toplam {len(bilenler)} kisi | {ilk['kisi']} HARIC {len(baskalari)} kisi kaliyor")
        print(f"    -> {', '.join(bilenler)}")

    # 4) Havuz genelinde dagilim: 0 / tam 1 / 2+ kisi
    havuz = yukle(HAVUZ_YOLU)
    tum_yetkinlikler = [ad for kat in havuz.values() for ad in kat]
    sifir, bir, iki_arti = 0, 0, 0
    bir_kisilikler = []
    for yad in tum_yetkinlikler:
        n = len(kac_kisi_biliyor(profiller, yad))
        if n == 0:
            sifir += 1
        elif n == 1:
            bir += 1
            bir_kisilikler.append(yad)
        else:
            iki_arti += 1

    print("\n--- Havuz genelinde dagilim (toplam {} yetkinlik) ---".format(len(tum_yetkinlikler)))
    print(f"0 kisi biliyor  : {sifir}  (gap - kimsede yok, bus factor riski DEGIL)")
    print(f"1 kisi biliyor  : {bir}  (GERCEK bus factor riski)")
    print(f"2+ kisi biliyor : {iki_arti}")
    print(f"\n1 kisilik (gercek riskli) yetkinlikler:")
    print(f"  {', '.join(bir_kisilikler) or '-'}")


if __name__ == "__main__":
    main()
