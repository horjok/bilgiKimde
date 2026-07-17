"""
Embedding worker - TEMIZ bir alt surecte calisir (Streamlit/torch segfault'unu izole eder).
yatkinlik_motoru.embeddingleri_yukle() ile 51 embedding'i hesaplar, .npz'ye yazar.
Kullanim:  python embed_worker.py <cikti_yolu.npz>
"""
import sys

import numpy as np

from yatkinlik_motoru import embeddingleri_yukle


def main(cikti_yolu):
    embed = embeddingleri_yukle()  # { ad: torch tensor }
    adlar = list(embed.keys())
    vektorler = np.stack([embed[ad].cpu().numpy() for ad in adlar]).astype("float32")
    np.savez(cikti_yolu, adlar=np.array(adlar), vektorler=vektorler)
    print(f"{len(adlar)} embedding yazildi -> {cikti_yolu}")


if __name__ == "__main__":
    main(sys.argv[1])
