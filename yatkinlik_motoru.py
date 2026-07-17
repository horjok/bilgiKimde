"""
Yatkinlik Motoru - Faz 3 (embedding + yatkinlik hesabi)

Ust kisim: yetkinlik havuzundaki 51 yetkinligi cok dilli bir cumle-embedding
modeliyle vektore cevirir (+ kisa kosinus testi).
Alt kisim: profillerdeki kisiler icin hedef bir yetkinlige yatkinlik hesaplar.
"""

import json

# NOT: sentence_transformers/torch importu bilerek TEMBEL (fonksiyon icinde) yapilir.
# Boylece bu modulu import eden torch'suz surecler ( or. Streamlit arayuzu) torch'u
# hic yuklemez; torch yalnizca embedding hesaplayan fonksiyon/surecte devreye girer.
MODEL_ADI = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Gorece yollar: script'in calistirildigi dizine gore cozulur (capstone/ icinden calistir).
HAVUZ_YOLU = "data/yetkinlik_havuzu.json"
PROFIL_YOLU = "data/profiller_zengin.json"

# Seviye -> agirlik. Daha ileri seviye, o bilgi uzerinden yatkinligi guclendirir.
SEVIYE_AGIRLIK = {"baslangic": 0.25, "orta": 0.50, "ileri": 0.75, "uzman": 1.0}

# Modul seviyesi embedding sozlugu: main() doldurur, yatkinlik() kullanir.
embed = {}


# ----------------------------- Embedding kismi -----------------------------

def yetkinlikleri_yukle(yol):
    """JSON'daki tum kategorileri tek duz yetkinlik listesinde birlestirir."""
    with open(yol, encoding="utf-8") as f:
        havuz = json.load(f)
    return [ad for kategori in havuz.values() for ad in kategori]


def embedding_sozlugu(model, yetkinlikler):
    """Her yetkinligi { ad: vektor } biciminde bir sozluge embed eder."""
    vektorler = model.encode(yetkinlikler, convert_to_tensor=True)
    return {ad: vektor for ad, vektor in zip(yetkinlikler, vektorler)}


def benzerlik(embed, a, b):
    """Iki yetkinlik arasindaki kosinus benzerligini dondurur."""
    from sentence_transformers import util
    return float(util.cos_sim(embed[a], embed[b]))


def embeddingleri_yukle():
    """Modeli yukler, modul seviyesi 'embed' sozlugunu doldurur, embed'i dondurur."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_ADI)
    embed.update(embedding_sozlugu(model, yetkinlikleri_yukle(HAVUZ_YOLU)))
    return embed


# ----------------------------- Yatkinlik kismi -----------------------------

def profilleri_yukle(yol):
    """Kisi profillerini (37 kisi) okur."""
    with open(yol, encoding="utf-8") as f:
        return json.load(f)


def yatkinlik(kisi, hedef_yetkinlik):
    """Kisinin hedefe yatkinligi. Sahipse (1.0, "zaten sahip"); degilse bildigi her
    yetkinlik y icin cos_sim(y, hedef)*agirlik[y.seviye] MAKSIMUMU alinir, en yuksegi
    veren y "kopru" olur. Doner (skor, kopru_yetkinlik)."""
    sahip = {y["ad"]: y["seviye"] for y in kisi["yetkinlikler"]}
    if hedef_yetkinlik in sahip:
        return (1.0, "zaten sahip")

    from sentence_transformers import util
    en_iyi_skor = -1.0
    en_iyi_kopru = None
    for ad, seviye in sahip.items():
        skor = float(util.cos_sim(embed[ad], embed[hedef_yetkinlik])) * SEVIYE_AGIRLIK[seviye]
        if skor > en_iyi_skor:
            en_iyi_skor = skor
            en_iyi_kopru = ad
    return (en_iyi_skor, en_iyi_kopru)


def en_yatkin_kisiler(profiller, hedef_yetkinlik, n=5):
    """Hedefe en yatkin n kisiyi dondurur (zaten sahip olanlar haric)."""
    sonuc = []
    for kisi in profiller:
        skor, kopru = yatkinlik(kisi, hedef_yetkinlik)
        if kopru == "zaten sahip":
            continue
        sonuc.append((kisi["kisi"], skor, kopru))
    sonuc.sort(key=lambda x: x[1], reverse=True)
    return sonuc[:n]


# --------------------------- Risk analizi kismi ----------------------------

def bus_factor():
    """Her yetkinlik icin sahip kisileri bulur; sahip sayisina gore ARTAN
    sirali doner (0 kisili dahil): [(yad, sayi, [(isim, seviye), ...]), ...]."""
    yetkinlikler = yetkinlikleri_yukle(HAVUZ_YOLU)
    profiller = profilleri_yukle(PROFIL_YOLU)
    sonuc = []
    for yad in yetkinlikler:
        sahipler = [(p["kisi"], y["seviye"]) for p in profiller
                    for y in p["yetkinlikler"] if y["ad"] == yad]
        sonuc.append((yad, len(sahipler), sahipler))
    sonuc.sort(key=lambda x: x[1])  # artan: en az kisili yetkinlik en riskli
    return sonuc


def kim_ogrenmeli(hedef_yetkinlik):
    """Ogrenmesi onerilen en iyi adayi (henuz sahip olmayan) doner:
    (isim, skor, kopru) veya None. yatkinlik() mantigini kullanir."""
    profiller = profilleri_yukle(PROFIL_YOLU)
    adaylar = en_yatkin_kisiler(profiller, hedef_yetkinlik, 1)
    return adaylar[0] if adaylar else None


def kisi_ayrilirsa(kisi_adi):
    """Kisi cikarilirsa etkilenen yetkinlikler. Doner (kisi yoksa None):
    (kaybolan[cikinca 0 kisi kalan], riske_giren[1 kisi kalan])."""
    profiller = profilleri_yukle(PROFIL_YOLU)
    kisi = next((p for p in profiller if p["kisi"] == kisi_adi), None)
    if kisi is None:
        return None
    kaybolan, riske_giren = [], []
    for y in kisi["yetkinlikler"]:
        yad = y["ad"]
        kalan = sum(1 for p in profiller if p["kisi"] != kisi_adi
                    and any(yy["ad"] == yad for yy in p["yetkinlikler"]))
        if kalan == 0:
            kaybolan.append(yad)
        elif kalan == 1:
            riske_giren.append(yad)
    return kaybolan, riske_giren


# --------------------------------- main ------------------------------------

def main():
    # 1-3) Modeli yukle + embed sozlugunu doldur (import edilebilir fonksiyon).
    print(f"Model yukleniyor: {MODEL_ADI} (ilk calistirmada indirilir)")
    embeddingleri_yukle()
    print(f"{len(embed)} yetkinlik vektore cevrildi")

    # 4) TEST (embedding): birkac ciftin kosinus benzerligi.
    ciftler = [("Docker", "Kubernetes"), ("Docker", "UI/UX Temelleri"),
               ("React", "Vue"), ("SQL", "PostgreSQL"), ("Python", "Teknik Liderlik")]
    print("\n--- Embedding testi ---")
    for a, b in ciftler:
        skor = benzerlik(embed, a, b)
        print(f"{a} <-> {b}: {skor:.2f}")

    # 5) TEST (yatkinlik): bir hedefe en yatkin 5 kisi.
    profiller = profilleri_yukle(PROFIL_YOLU)
    print(f"\n{len(profiller)} kisi yuklendi")

    hedef = "Kubernetes"
    print(f"\n--- '{hedef}' yetkinligine en yatkin 5 kisi (zaten sahip olanlar haric) ---")
    for i, (isim, skor, kopru) in enumerate(en_yatkin_kisiler(profiller, hedef, 5), 1):
        print(f"{i}. {isim} - {skor:.2f} (kopru: {kopru})")

    # 6) ANALIZ 1: bus factor - "0 kisi" (kapsam disi) ile gercek risk ayrildi.
    bf = bus_factor()
    kapsam_disi = [yad for yad, sayi, _ in bf if sayi == 0]
    print(f"\n--- KAPSAM DISI (kimse bilmiyor) - {len(kapsam_disi)} yetkinlik ---")
    print("  " + (", ".join(kapsam_disi[:5]) + (" ..." if len(kapsam_disi) > 5 else "") or "(yok)"))
    print("\n--- Bus factor risk siralamasi (1+ kisi, en az kisili ustte) ---")
    for yad, sayi, sahipler in [x for x in bf if x[1] >= 1][:10]:
        etiket = "RISKLI (tek kisi)" if sayi == 1 else "DIKKAT" if sayi == 2 else "guvende"
        print(f"{sayi} kisi - {yad} [{etiket}]: {', '.join(i for i, _ in sahipler)}")
    tek_kisilikler = [(yad, s[0][0]) for yad, sayi, s in bf if sayi == 1]
    print(f"\n--- ANA RISK: tam 1 kisinin bildigi yetkinlikler ({len(tek_kisilikler)}) ---")
    for yad, kisi in tek_kisilikler:
        print(f"{yad} - sadece [{kisi}] biliyor")

    # 7) ANALIZ 2: en riskli yetkinlik icin ogrenme onerisi.
    en_riskli = bf[0][0]
    print(f"\n--- '{en_riskli}' icin ogrenmesi onerilen kisi ---")
    oneri = kim_ogrenmeli(en_riskli)
    if oneri:
        isim, skor, kopru = oneri
        print(f"{isim} - {skor:.2f} (kopru: {kopru})")
    else:
        print("Uygun aday yok.")

    # 8) ANALIZ 3: kisi ayrilirsa - tek sahipli bir yetkinligin sahibini otomatik sec.
    if tek_kisilikler:
        sec_yad, sec_kisi = tek_kisilikler[0]
        print(f"\n--- '{sec_kisi}' ayrilirsa (secim: '{sec_yad}' yetkinliginin tek sahibi) ---")
        kaybolan, riske_giren = kisi_ayrilirsa(sec_kisi)
        print(f"KAYBOLUYOR ({len(kaybolan)}): {', '.join(kaybolan) or '-'}")
        print(f"RISKE GIRIYOR ({len(riske_giren)}): {', '.join(riske_giren) or '-'}")
    else:
        print("\n(Tam 1 kisilik yetkinlik yok; test atlaniyor.)")


if __name__ == "__main__":
    main()
