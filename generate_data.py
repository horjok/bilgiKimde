"""Faz 1: Sentetik veri üreteci (mock — LLM yok)."""

import json
import random
from pathlib import Path

from pydantic import BaseModel

HAVUZ_YOLU = Path("data/yetkinlik_havuzu.json")
CIKTI_YOLU = Path("data/profiller.json")

SEVIYELER = ["baslangic", "orta", "ileri", "uzman"]
SAHTE_ISIMLER = [
    "Ada Yılmaz", "Deniz Kaya", "Ece Demir", "Kerem Şahin",
    "Mert Aydın", "Selin Arslan", "Berk Çelik", "Naz Koç",
]


class Yetkinlik(BaseModel):
    ad: str
    seviye: str
    kanit: str


class Profil(BaseModel):
    kisi: str
    yetkinlikler: list[Yetkinlik]


def havuzu_oku() -> list[str]:
    """Havuzdaki tüm yetkinlikleri düz bir listeye indir."""
    veri = json.loads(HAVUZ_YOLU.read_text(encoding="utf-8"))
    tum_yetkinlikler: list[str] = []
    for liste in veri.values():
        tum_yetkinlikler.extend(liste)
    return tum_yetkinlikler


def generate_profile(havuz: list[str]) -> Profil:
    """MOCK: havuzdan rastgele 4-6 yetkinlik seçip sahte profil üretir."""
    adet = random.randint(4, 6)
    secilenler = random.sample(havuz, adet)
    yetkinlikler = [
        Yetkinlik(
            ad=ad,
            seviye=random.choice(SEVIYELER),
            kanit="örnek kanıt",
        )
        for ad in secilenler
    ]
    return Profil(kisi=random.choice(SAHTE_ISIMLER), yetkinlikler=yetkinlikler)


def main() -> None:
    havuz = havuzu_oku()
    profiller = [generate_profile(havuz) for _ in range(3)]

    veri = [p.model_dump() for p in profiller]
    CIKTI_YOLU.write_text(
        json.dumps(veri, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(veri, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
