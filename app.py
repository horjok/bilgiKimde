"""
Bilgi Kimde Yönetim Panosu - Streamlit arayüzü.

Motor mantığı yatkinlik_motoru.py'den import edilir. Embedding hesabı (torch)
TEMİZ bir alt süreçte yapılır; Streamlit sürecinde HİÇ torch işlemi çalışmaz.
Grafikler matplotlib ile çizilir (yeni bağımlılık yok).

Çalıştırma: capstone/ dizininden ->  streamlit run app.py
"""

import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
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

# ------------------------------- Yardımcı veri -------------------------------

IKON = {
    "C#/.NET": "#️⃣", "ASP.NET Core": "🌐", "Python": "🐍", "Java": "☕", "Go": "🐹",
    "Node.js": "🟩", "REST API": "🔌", "GraphQL": "◈", "Mikroservis Mimarisi": "🧩",
    "RabbitMQ": "🐇", "Kafka": "📨", "gRPC": "⚡", "JavaScript": "🟨", "TypeScript": "🔷",
    "React": "⚛️", "Vue": "💚", "Angular": "🅰️", "Next.js": "▲", "HTML/CSS": "🎨",
    "Tailwind": "🌬️", "State Yönetimi (Redux)": "🔄", "UI/UX Temelleri": "🖌️", "SQL": "🗄️",
    "PostgreSQL": "🐘", "MongoDB": "🍃", "Redis": "🟥", "Elasticsearch": "🔎",
    "Veri Modelleme": "📐", "Pandas": "🐼", "ETL": "🔀", "Temel Makine Öğrenmesi": "🤖",
    "PyTorch/TensorFlow": "🔥", "Docker": "🐳", "Kubernetes": "☸️", "CI/CD": "♻️",
    "Linux": "🐧", "Git": "🔧", "AWS": "☁️", "Azure": "⛅", "Terraform": "🏗️",
    "Ansible": "📜", "Prometheus/Grafana": "📈", "Nginx": "🕸️", "Sistem Tasarımı": "📐",
    "API Tasarımı": "🔗", "Test/TDD": "✅", "Kod Review": "🔍", "Dokümantasyon": "📝",
    "Teknik Liderlik": "🧭", "Güvenlik Temelleri": "🛡️", "Performans Optimizasyonu": "⚙️",
}


def ikon(ad):
    return IKON.get(ad, "🔹")


def avatar(isim):
    bas = "".join(p[0] for p in isim.split()[:2]).upper()
    return f"<span class='avatar'>{bas}</span>"


# Marka logolari (simple-icons CDN, tarayici yukler). Olmayanlar emoji'ye duser.
LOGO_SLUG = {
    "C#/.NET": "dotnet", "ASP.NET Core": "dotnet", "Python": "python", "Go": "go",
    "Node.js": "nodedotjs", "GraphQL": "graphql", "RabbitMQ": "rabbitmq", "Kafka": "apachekafka",
    "JavaScript": "javascript", "TypeScript": "typescript", "React": "react", "Vue": "vuedotjs",
    "Angular": "angular", "Next.js": "nextdotjs", "HTML/CSS": "html5", "Tailwind": "tailwindcss",
    "State Yönetimi (Redux)": "redux", "PostgreSQL": "postgresql", "MongoDB": "mongodb",
    "Redis": "redis", "Elasticsearch": "elasticsearch", "Pandas": "pandas",
    "PyTorch/TensorFlow": "pytorch", "Docker": "docker", "Kubernetes": "kubernetes",
    "Linux": "linux", "Git": "git", "Terraform": "terraform", "Ansible": "ansible",
    "Prometheus/Grafana": "prometheus", "Nginx": "nginx",
}


def logo_html(yad):
    """Kart arkasi yari saydam teknoloji logosu (yoksa buyuk soluk emoji)."""
    slug = LOGO_SLUG.get(yad)
    if slug:
        return f"<img class='kart-logo' src='https://cdn.simpleicons.org/{slug}'/>"
    return f"<span class='kart-logo-emoji'>{ikon(yad)}</span>"


# Isimden cinsiyet (fotoyu esletmek icin). Unisex isimlerde makul bir varsayilan.
CINSIYET = {
    "Elif": "women", "Burak": "men", "Deniz": "men", "Cem": "men", "Zeynep": "women",
    "Kaan": "men", "Selin": "women", "Emre": "men", "Merve": "women", "Onur": "men",
    "Ceren": "women", "Tolga": "men", "Pinar": "women", "Serkan": "men", "Ebru": "women",
    "Baris": "men", "Gizem": "women", "Umut": "men", "Asli": "women", "Kerem": "men",
    "Yasemin": "women", "Doruk": "men", "Nihan": "women", "Berk": "men", "Sena": "women",
    "Arda": "men", "Melis": "women", "Ozan": "men", "Damla": "women", "Volkan": "men",
    "Irem": "women", "Sinan": "men", "Bahar": "women", "Efe": "men", "Gokce": "women",
    "Mert": "men", "Ada": "women", "Can": "men", "Naz": "women", "Eren": "men",
}


def foto_html(isim):
    """Kisi profil fotografi (randomuser.me); cinsiyeti isimle esler, isme gore sabit."""
    cinsiyet = CINSIYET.get(isim.split()[0], "men")
    idx = int(hashlib.md5(isim.encode("utf-8")).hexdigest(), 16) % 100
    return f"<img class='foto' src='https://randomuser.me/api/portraits/{cinsiyet}/{idx}.jpg'/>"


# -------------------------- Önbellek / motor kancası -------------------------

@st.cache_resource
def embeddingleri_getir():
    """Onceden hesaplanmis data/embeddings.npz varsa onu yukler (dagitim: torch'suz,
    hafif). Yoksa (yerel gelistirme) alt surecte gercek modelle hesaplar."""
    kapsam = os.path.dirname(os.path.abspath(__file__))
    npz = os.path.join(kapsam, "data", "embeddings.npz")
    if not os.path.exists(npz):
        import subprocess
        import tempfile
        npz = os.path.join(tempfile.gettempdir(), "yetkinlik_embed.npz")
        subprocess.run([sys.executable, "embed_worker.py", npz], check=True, cwd=kapsam)
    veri = np.load(npz)
    return {str(ad): vek for ad, vek in zip(veri["adlar"], veri["vektorler"])}


@st.cache_data
def havuz_yetkinlikleri():
    return yetkinlikleri_yukle(HAVUZ_YOLU)


@st.cache_data
def kategoriler():
    with open(HAVUZ_YOLU, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def profiller():
    return profilleri_yukle(PROFIL_YOLU)


def _kosinus(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def en_yatkin(profil_listesi, hedef, embed, n=6):
    """en_yatkin_kisiler ile AYNI mantık (numpy kosinüs): (isim, skor, köprü)."""
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


# ------------------------------ Grafik yardımcıları ------------------------------

def _fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    return fig, ax


def donut_calisan(toplam, aktif, pasif):
    fig, ax = _fig(2.4, 2.1)
    ax.pie([aktif, pasif], colors=["#3b82f6", "#8b5cf6"], startangle=90,
           counterclock=False, wedgeprops=dict(width=0.36))
    ax.text(0, 0.08, str(toplam), ha="center", va="center", color="#e2e8f0",
            fontsize=22, fontweight="bold")
    ax.text(0, -0.28, "Toplam", ha="center", va="center", color="#94a3b8", fontsize=9)
    ax.set_aspect("equal")
    return fig


def bar_risk(kritik, orta):
    fig, ax = _fig(2.6, 2.0)
    ax.bar(["kritik", "orta"], [kritik, orta], color=["#ef4444", "#f59e0b"], width=0.55)
    for i, v in enumerate([kritik, orta]):
        ax.text(i, v + 0.15, str(v), ha="center", color="#e2e8f0", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(kritik, orta, 1) * 1.25)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_yticks([])
    return fig


def gauge_kapsam(deger, maks):
    fig, ax = _fig(2.6, 1.9)
    for t1, t2, renk in [(180, 120, "#22c55e"), (120, 60, "#f59e0b"), (60, 0, "#ef4444")]:
        ax.add_patch(Wedge((0, 0), 1.0, t2, t1, width=0.34, facecolor=renk))
    aci = np.radians(180 - min(deger / maks, 1) * 180)
    ax.plot([0, 0.7 * np.cos(aci)], [0, 0.7 * np.sin(aci)], color="#e2e8f0", lw=2.2)
    ax.add_patch(Circle((0, 0), 0.05, color="#e2e8f0"))
    ax.text(0, -0.32, str(deger), ha="center", color="#e2e8f0", fontsize=18, fontweight="bold")
    ax.text(0, -0.58, "Toplam", ha="center", color="#94a3b8", fontsize=8)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.62, 1.15)
    ax.axis("off")
    ax.set_aspect("equal")
    return fig


def _lejant(satirlar):
    return "<div class='lejant'>" + "".join(
        f"<div><span style='color:{r}'>●</span> {t}</div>" for r, t in satirlar
    ) + "</div>"


def ust_metrikler():
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown("**Çalışan Dağılımı**")
        g, l = st.columns([1.5, 1])
        with g:
            f = donut_calisan(len(profiller()), 32, 5)
            st.pyplot(f)
            plt.close(f)
        l.markdown(_lejant([("#3b82f6", f"<b>{len(profiller())}</b>"),
                            ("#3b82f6", "32 aktif"), ("#8b5cf6", "5 pasif")]),
                   unsafe_allow_html=True)
    with c2.container(border=True):
        st.markdown("**Risk Noktaları**")
        g, l = st.columns([1.5, 1])
        with g:
            f = bar_risk(8, 2)
            st.pyplot(f)
            plt.close(f)
        l.markdown(_lejant([("#ef4444", f"<b>{len(TEK_KISILIK)}</b>"),
                            ("#ef4444", "8 kritik"), ("#f59e0b", "2 orta")]),
                   unsafe_allow_html=True)
    with c3.container(border=True):
        st.markdown("**Kapsam Dışı Yetkinlikler**")
        g, l = st.columns([1.5, 1])
        with g:
            f = gauge_kapsam(len(KAPSAM_DISI), 51)
            st.pyplot(f)
            plt.close(f)
        l.markdown(_lejant([("#ef4444", f"<b>{len(KAPSAM_DISI)}</b>"),
                            ("#94a3b8", "15 bilinmeyen"), ("#f59e0b", "3 öğreniyor")]),
                   unsafe_allow_html=True)


def rozetler(liste):
    html = "".join(f"<span class='rozet'>{ikon(y)} {y}</span>" for y in liste)
    st.markdown(f"<div class='rozet-kutu'>{html}</div>", unsafe_allow_html=True)


# ------------------------------------ Sayfalar -----------------------------------

def sayfa_panel():
    st.markdown("### Genel Bakış")
    ust_metrikler()
    st.divider()
    st.subheader("Tek Kişiye Bağlı Riskler")
    st.caption("Yalnızca tek kişinin bildiği, kaybolma riski taşıyan yetkinlikler.")
    for i in range(0, len(TEK_KISILIK), 3):
        kols = st.columns(3)
        for kol, (yad, kisi) in zip(kols, TEK_KISILIK[i:i + 3]):
            kol.markdown(
                f"<div class='tek-kart'>{logo_html(yad)}"
                f"<div class='kart-baslik'>{ikon(yad)} {yad}</div>"
                f"<div class='kisi-satir'>{foto_html(kisi)}"
                f"<span class='muted'>{kisi}</span></div></div>",
                unsafe_allow_html=True,
            )
    st.divider()
    st.subheader(f"Kimse bilmiyor — kapsam dışı ({len(KAPSAM_DISI)} yetkinlik)")
    rozetler(KAPSAM_DISI)


def sayfa_calisan_analizi():
    st.markdown("### Öğrenme Yatkınlığı Analizi")
    ust_metrikler()
    st.divider()
    hedef = st.selectbox("Yetkinlik seç", havuz_yetkinlikleri())
    st.subheader(f"{hedef} için en yatkın kişiler")
    st.caption("Bu yetkinliği henüz bilmeyen, embedding yakınlığına göre en yatkın kişiler.")
    sonuc = en_yatkin(profiller(), hedef, EMBED, 6)
    if not sonuc:
        st.info("Uygun aday yok (herkes zaten biliyor).")
        return
    for i in range(0, len(sonuc), 3):
        kols = st.columns(3)
        for kol, (isim, skor, kopru) in zip(kols, sonuc[i:i + 3]):
            g = int(min(max(skor, 0.0), 1.0) * 100)
            kol.markdown(
                f"<div class='tek-kart'>{logo_html(hedef)}"
                f"<div class='kisi-satir'>{foto_html(isim)}<b>{isim}</b></div>"
                f"<div class='muted' style='font-size:0.82rem;margin-top:6px'>Köprü: {kopru}</div>"
                f"<div style='margin-top:4px'>Beceri örtüşmesi: <b>{skor:.2f}</b></div>"
                f"<div class='cubuk'><div class='cubuk-ic' style='width:{g}%'></div></div>"
                f"<div class='yol'>Temel · Orta · İleri · Proje</div></div>",
                unsafe_allow_html=True,
            )


def sayfa_risk_genel():
    st.markdown("### Risk Genel Bakışı")
    ust_metrikler()
    st.divider()
    st.subheader("Yetkinliklerin kapsamı (bus factor)")
    satirlar = sorted(SAYIM.items(), key=lambda x: x[1])
    for i in range(0, len(satirlar), 2):
        kols = st.columns(2)
        for kol, (yad, sayi) in zip(kols, satirlar[i:i + 2]):
            if sayi == 0:
                etiket, renk = "kapsam dışı", "#94a3b8"
            elif sayi == 1:
                etiket, renk = "kritik", "#ef4444"
            elif sayi == 2:
                etiket, renk = "orta", "#f59e0b"
            else:
                etiket, renk = "güvende", "#22c55e"
            kol.markdown(
                f"<div class='satir'>{ikon(yad)} <b>{yad}</b>"
                f"<span class='etiket' style='background:{renk}22;color:{renk}'>"
                f"{sayi} kişi · {etiket}</span></div>",
                unsafe_allow_html=True,
            )


def sayfa_yetkinlikler():
    st.markdown("### Yetkinlikler")
    st.caption("51 yetkinlik, 5 kategori. Yanındaki sayı kaç kişinin bildiğidir.")
    for kat, liste in kategoriler().items():
        st.subheader(kat)
        kols = st.columns(2)
        for j, yad in enumerate(liste):
            kols[j % 2].markdown(
                f"<div class='satir'>{ikon(yad)} <b>{yad}</b>"
                f"<span class='muted'>{SAYIM.get(yad, 0)} kişi</span></div>",
                unsafe_allow_html=True,
            )


def sayfa_ayarlar():
    st.markdown("### Ayarlar")
    st.info("Tema koyu-mavi kurumsal olarak sabittir. Model: paraphrase-multilingual-MiniLM-L12-v2.")
    st.write(f"- Toplam çalışan: **{len(profiller())}**")
    st.write(f"- Toplam yetkinlik: **{len(havuz_yetkinlikleri())}**")
    st.write("- Embedding izole alt süreçte (torch) hesaplanır; arayüz numpy kosinüsü kullanır.")


def sayfa_yardim():
    st.markdown("### Yardım & Destek")
    st.write("**Panel** — özet metrikler + tek kişiye bağlı riskler + kapsam dışı yetkinlikler.")
    st.write("**Çalışan Analizi** — bir yetkinliği en kolay öğrenecek kişiler (yatkınlık).")
    st.write("**Risk Genel Bakışı** — tüm yetkinliklerin bus factor (kapsam) dağılımı.")
    st.write("**Yetkinlikler** — kategorilere göre tüm yetkinlikler ve kapsamı.")
    st.caption("Yatkınlık = cos_sim(bilinen, hedef) × seviye_ağırlığı; en yükseği köprü yetkinliktir.")


# --------------------------------- Uygulama akışı --------------------------------

st.set_page_config(page_title="Bilgi Kimde Yönetimi", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1280px; }
    section[data-testid="stSidebar"] { background-color: #0b1220; }
    div[role="radiogroup"] label { padding: 8px 12px; border-radius: 8px; margin: 1px 0;
        font-weight: 600; }
    div[role="radiogroup"] label:hover { background-color: #1e293b; }
    div[role="radiogroup"] label:has(input:checked) { background-color: #1e293b;
        border-left: 3px solid #2563eb; }
    [data-testid="stMetric"], .stTabs { border-radius: 12px; }
    .avatar { display:inline-flex; width:26px; height:26px; border-radius:50%;
        background:#2563eb; color:#fff; align-items:center; justify-content:center;
        font-size:11px; font-weight:700; margin-right:6px; vertical-align:middle; }
    .kart-baslik { font-size:1.05rem; font-weight:700; margin-bottom:6px; }
    .muted { color:#94a3b8; }
    .lejant { font-size:0.85rem; line-height:1.9; padding-top:8px; }
    .rozet-kutu { line-height:2.4; }
    .rozet { display:inline-block; padding:5px 12px; margin:3px; border:1px solid #334155;
        border-radius:20px; background:#1e293b; font-size:0.85rem; }
    .satir { padding:8px 12px; margin:4px 0; border:1px solid #334155; border-radius:8px;
        background:#1e293b; display:flex; justify-content:space-between; align-items:center; }
    .etiket { padding:2px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
    .yol { color:#64748b; font-size:0.75rem; margin-top:6px; }
    .tek-kart { position:relative; overflow:hidden; border:1px solid #334155;
        border-radius:12px; background:#1e293b; padding:16px 18px; margin-bottom:14px;
        min-height:96px; }
    .tek-kart > * { position:relative; z-index:1; }
    .tek-kart > .kart-logo, .tek-kart > .kart-logo-emoji { z-index:0; }
    .kart-logo { position:absolute; right:-6px; top:50%; transform:translateY(-50%);
        width:86px; height:86px; opacity:0.15; object-fit:contain; pointer-events:none; }
    .kart-logo-emoji { position:absolute; right:10px; top:50%; transform:translateY(-50%);
        font-size:60px; opacity:0.12; pointer-events:none; }
    .kisi-satir { display:flex; align-items:center; gap:8px; }
    .foto { width:30px; height:30px; border-radius:50%; object-fit:cover; }
    .cubuk { height:8px; background:#334155; border-radius:6px; margin-top:8px; overflow:hidden; }
    .cubuk-ic { height:100%; background:#2563eb; border-radius:6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Embedding'ler hesaplanıyor (ilk seferde model indirilebilir)..."):
    EMBED = embeddingleri_getir()

_BF = bus_factor()
TEK_KISILIK = [(yad, sahipler[0][0]) for yad, sayi, sahipler in _BF if sayi == 1]
KAPSAM_DISI = [yad for yad, sayi, _ in _BF if sayi == 0]
SAYIM = {yad: sayi for yad, sayi, _ in _BF}

with st.sidebar:
    st.markdown("<h2 style='margin:0'>🛡️ Bilgi Kimde?</h2>"
                "<p class='muted' style='margin-top:2px'>Yönetim Panosu</p>",
                unsafe_allow_html=True)
    st.divider()
    SAYFALAR = {
        "📊 Panel": sayfa_panel,
        "👥 Çalışan Analizi": sayfa_calisan_analizi,
        "⚠️ Risk Genel Bakışı": sayfa_risk_genel,
        "🧩 Yetkinlikler": sayfa_yetkinlikler,
        "⚙️ Ayarlar": sayfa_ayarlar,
        "❓ Yardım & Destek": sayfa_yardim,
    }
    secim = st.radio("Menü", list(SAYFALAR), label_visibility="collapsed")

SAYFALAR[secim]()
