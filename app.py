"""
Yetkinlik & Risk Analiz Panosu ("Bilgi Kimde?") - Streamlit arayuzu.

Motor mantigi yatkinlik_motoru.py'den import edilir. Embedding hesabi (torch)
TEMIZ bir alt surecte yapilir; Streamlit surecinde HIC torch islemi calismaz -
cunku bu ortamda torch, Streamlit sunucu surecinde segfault veriyor. Kosinus
benzerligi numpy ile hesaplanir (util.cos_sim ile sayisal olarak ozdes).

Calistirma: capstone/ dizininden ->  streamlit run app.py
"""

import os
import sys

# Alt surec / import guvenligi icin thread paralelligini kis (zararsiz).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import streamlit as st

import yatkinlik_motoru as ym
from yatkinlik_motoru import (
    bus_factor,
    kisi_ayrilirsa,
    yetkinlikleri_yukle,
    profilleri_yukle,
    HAVUZ_YOLU,
    PROFIL_YOLU,
)


@st.cache_resource
def embeddingleri_getir():
    """51 embedding'i TEMIZ bir alt surecte (gercek model) hesaplar ve numpy
    sozluk dondurur. Streamlit surecinde torch islemi olmaz -> segfault yok."""
    import subprocess
    import tempfile

    kapsam = os.path.dirname(os.path.abspath(__file__))
    npz = os.path.join(tempfile.gettempdir(), "yetkinlik_embed.npz")
    subprocess.run([sys.executable, "embed_worker.py", npz], check=True, cwd=kapsam)
    veri = np.load(npz)
    return {str(ad): vek for ad, vek in zip(veri["adlar"], veri["vektorler"])}


@st.cache_data
def havuz_yetkinlikleri():
    return yetkinlikleri_yukle(HAVUZ_YOLU)


@st.cache_data
def profiller():
    return profilleri_yukle(PROFIL_YOLU)


def _kosinus(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def en_yatkin(profil_listesi, hedef, embed, n=5):
    """yatkinlik_motoru.en_yatkin_kisiler ile AYNI mantik, numpy kosinusuyle:
    kisinin bildigi her yetkinlik icin cos * seviye_agirligi -> maks; kopru = argmaks.
    Zaten sahip olanlar haric. util.cos_sim ile sayisal olarak ozdes."""
    sonuc = []
    for kisi in profil_listesi:
        sahip = {y["ad"]: y["seviye"] for y in kisi["yetkinlikler"]}
        if hedef in sahip:
            continue
        en_iyi, kopru = -1.0, None
        for ad, sev in sahip.items():
            s = _kosinus(embed[ad], embed[hedef]) * ym.SEVIYE_AGIRLIK[sev]
            if s > en_iyi:
                en_iyi, kopru = s, ad
        sonuc.append((kisi["kisi"], en_iyi, kopru))
    sonuc.sort(key=lambda x: x[1], reverse=True)
    return sonuc[:n]


st.set_page_config(page_title="Bilgi Kimde?", page_icon="🛡️", layout="wide")

# --------------------- Hafif kurumsal tema (tek style blogu) ---------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2.4rem; padding-bottom: 2rem; max-width: 1180px; }
    [data-testid="stMetric"] {
        background-color: #1e293b; border: 1px solid #334155;
        border-radius: 12px; padding: 16px 20px;
    }
    [data-testid="stMetricLabel"] p { color: #94a3b8; font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] { font-size: 1.05rem; font-weight: 600; padding: 10px 20px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------- Baslik alani ----------------------------------
st.markdown(
    "<h1 style='margin-bottom:0'>Bilgi Kimde?</h1>"
    "<p style='color:#94a3b8; font-size:1.05rem; margin-top:2px'>"
    "Ekip bilgi riski ve öğrenme yatkınlığı analizi</p>",
    unsafe_allow_html=True,
)

# ------------------------------------ Sidebar ------------------------------------
with st.sidebar:
    st.title("🛡️ Bilgi Kimde?")
    st.caption("Ekipteki bilgi riskini ve öğrenme yatkınlığını analiz eder.")
    with st.expander("Nasıl çalışır?"):
        st.write(
            "Her yetkinlik çok dilli bir embedding modeliyle vektöre çevrilir. "
            "Bir kişinin mevcut becerilerine kosinüs yakınlığı, o kişinin bir "
            "yetkinliğe **öğrenme yatkınlığını** verir. Bir yetkinliği kaç kişinin "
            "bildiği ise **bus factor** riskini (tek kişiye bağımlılık) gösterir."
        )

# --------------------------- Model / embedding yukleme ---------------------------
with st.spinner("Embedding'ler hesaplanıyor (ilk seferde model indirilebilir)..."):
    EMBED = embeddingleri_getir()

# Ortak veri (bus_factor bir kez).
bf = bus_factor()
tek_kisilik = [(yad, sahipler[0][0]) for yad, sayi, sahipler in bf if sayi == 1]
kapsam_disi = [yad for yad, sayi, _ in bf if sayi == 0]
kisiler = [p["kisi"] for p in profiller()]

# ------------------------------ Ust ozet metrikler -------------------------------
st.divider()
m1, m2, m3 = st.columns(3)
m1.metric("👥 Çalışan Sayısı", len(kisiler), delta="aktif profil", delta_color="off")
m2.metric("🔴 Risk Noktası", len(tek_kisilik), delta="tek kişiye bağlı", delta_color="off")
m3.metric("⚪ Kapsam Dışı Yetkinlik", len(kapsam_disi), delta="kimse bilmiyor", delta_color="off")
st.divider()

sekme1, sekme2, sekme3 = st.tabs(["⚠️ Risk Panosu", "🎯 Kim Yatkın?", "🚪 Ya Ayrılırsa?"])

# ----------------------------- SEKME 1: Risk Panosu ------------------------------
with sekme1:
    st.markdown("##### Tek kişiye bağlı, kaybolma riski taşıyan yetkinlikler")
    st.caption("Bu kişiler ayrılırsa ilgili bilgi tamamen kaybolur — asıl risk noktaları.")
    if tek_kisilik:
        kolonlar = st.columns(3)
        for i, (yad, kisi) in enumerate(tek_kisilik):
            with kolonlar[i % 3].container(border=True):
                st.markdown(f"**{yad}**")
                st.caption(f"sadece {kisi} biliyor")
    else:
        st.success("Tek kişiye bağlı yetkinlik yok — ekip bu açıdan yedekli.")
    st.divider()
    with st.expander(f"Kimse bilmiyor — kapsam dışı ({len(kapsam_disi)} yetkinlik)"):
        st.write(", ".join(kapsam_disi) if kapsam_disi else "Yok")

# ----------------------------- SEKME 2: Kim Yatkın? ------------------------------
with sekme2:
    st.markdown("##### Seçilen yetkinliği en kolay öğrenebilecek (henüz bilmeyen) kişiler")
    hedef = st.selectbox("Yetkinlik seç", havuz_yetkinlikleri())
    st.caption(f"'{hedef}' yetkinliğine en yatkın (henüz sahip olmayan) 5 kişi:")
    st.divider()
    sonuc = en_yatkin(profiller(), hedef, EMBED, 5)
    if sonuc:
        for i, (isim, skor, kopru) in enumerate(sonuc, 1):
            st.markdown(f"**{i}. {isim}**  ·  köprü yetkinlik: *{kopru}*  ·  skor: **{skor:.2f}**")
            st.progress(min(max(skor, 0.0), 1.0))
    else:
        st.info("Bu yetkinliğe yatkın (henüz sahip olmayan) kişi bulunamadı.")

# ---------------------------- SEKME 3: Ya Ayrılırsa? -----------------------------
with sekme3:
    st.markdown("##### Bir kişi ayrılırsa hangi yetkinlikler kaybolur veya riske girer")
    kisi = st.selectbox("Kişi seç", kisiler)
    profil = next(p for p in profiller() if p["kisi"] == kisi)
    st.caption(f"**{kisi}** toplam {len(profil['yetkinlikler'])} yetkinlik biliyor.")
    st.divider()
    kaybolan, riske_giren = kisi_ayrilirsa(kisi)
    if not kaybolan and not riske_giren:
        st.success(f"{kisi} ayrılırsa kritik etki yok — bu kişinin bilgisi yedekli.")
    else:
        if kaybolan:
            st.error("Tamamen kaybolan yetkinlikler (bu kişi çıkınca 0 kişi kalır):")
            for yad in kaybolan:
                st.error(f"• {yad}")
        if riske_giren:
            st.warning("Riske giren yetkinlikler (bus factor 1'e düşer):")
            for yad in riske_giren:
                st.warning(f"• {yad}")
