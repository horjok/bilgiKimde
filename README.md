# 🛡️ Bilgi Kimde?

**Ekip bilgi riski ve öğrenme yatkınlığı analizi**

CV / yetkinlik verisinden yola çıkarak bir ekipteki bilgiyi vektör uzayında konumlandırır;
_"Bu yetkinlik kimde?"_, _"Bu kişi ayrılırsa ne kaybederiz?"_ ve _"Bu bilgiyi en kolay kim
öğrenir?"_ sorularına **embedding tabanlı** cevap verir. Doğal dilde sorgulama için ayrıca
bir **RAG** katmanı ile genişletilir.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-arayüz-FF4B4B?logo=streamlit&logoColor=white)
![Model](https://img.shields.io/badge/embedding-MiniLM--L12--multilingual-2563eb)

---

## 🎯 Ne işe yarar?

Ekiplerde bilgi çoğu zaman belgelerde değil, **kişilerin kafasında** durur. Bu proje bu
görünmez riski ölçülebilir hale getirir:

- **Bus factor riski** — bir yetkinliği yalnızca bir kişi biliyorsa, o kişi ayrıldığında bilgi
  tamamen kaybolur. Bu "tek kişiye bağlı" noktaları bulur.
- **Öğrenme yatkınlığı** — bir yetkinliği henüz bilmeyen ama mevcut becerileri sayesinde en
  kolay öğrenebilecek kişileri, embedding yakınlığıyla önerir (bus factor'ü düşürmek için).

## ✨ Özellikler

Streamlit panosu üç sekmeden oluşur:

| Sekme | Ne gösterir |
|-------|-------------|
| ⚠️ **Risk Panosu** | Tek kişiye bağlı (bus factor 1) yetkinlikler + kimsenin bilmediği kapsam dışı yetkinlikler + özet metrikler |
| 🎯 **Kim Yatkın?** | Seçilen bir yetkinliğe en yatkın 5 kişi; skor + hangi bilgisi sayesinde yatkın olduğu ("köprü yetkinlik") |
| 🚪 **Ya Ayrılırsa?** | Seçilen kişi ayrılırsa hangi yetkinlikler tamamen kaybolur, hangileri riske girer |

## 🧠 Nasıl çalışır?

1. **Embedding** — 51 yetkinliğin her biri çok dilli bir cümle-embedding modeliyle
   (`paraphrase-multilingual-MiniLM-L12-v2`) vektöre çevrilir.
2. **Yatkınlık** — bir kişinin hedef yetkinliğe yatkınlığı, bildiği her yetkinlik `y` için:

   ```
   skor = max_y ( cos_sim(embedding[y], embedding[hedef]) × seviye_ağırlığı[y] )
   ```

   En yüksek çarpımı veren `y`, kişinin hangi bilgisi sayesinde yatkın olduğunu gösteren
   **köprü yetkinliktir**. Seviye ağırlıkları: `başlangıç 0.25 · orta 0.50 · ileri 0.75 · uzman 1.0`.
3. **Bus factor** — her yetkinliği kaç kişinin bildiği sayılır. `1 kişi` → gerçek risk,
   `0 kişi` → kapsam dışı (kimsede yok).

## 📁 Proje yapısı

```
capstone/
├── app.py                  # Streamlit arayüzü ("Bilgi Kimde?")
├── yatkinlik_motoru.py     # Motor: embedding, yatkinlik(), bus_factor(), kisi_ayrilirsa()
├── embed_worker.py         # Embedding'i izole alt süreçte hesaplar (bkz. Teknik notlar)
├── generate_data.py        # Sentetik profil/veri üreteci
├── teshis.py               # Risk dağılımı teşhis scripti
├── data/
│   ├── yetkinlik_havuzu.json   # 51 yetkinlik, 5 kategori
│   └── profiller_zengin.json   # 37 kişi profili (cv_metni + yetkinlikler)
└── .streamlit/config.toml  # Kurumsal koyu-mavi tema
```

## 🚀 Kurulum ve çalıştırma

Gereksinimler (ilk çalıştırmada model ~470 MB indirilir):

```powershell
pip install sentence-transformers streamlit numpy
```

Çalıştırma (capstone dizininden):

```powershell
cd capstone
streamlit run app.py
```

Tarayıcıda `http://localhost:8501` açılır. Tema ve ayarlar `.streamlit/config.toml`'dan gelir.

Motoru CLI'de denemek için:

```powershell
python yatkinlik_motoru.py
```

## 🔎 RAG Sorgu Katmanı (Kaggle)

Doğal dilde sorgulama ("_Kubernetes'i kim öğrenmeli?_", "_En kırılgan ekip hangisi?_") bir
RAG katmanıyla yapılır. Not defteri Kaggle üzerinde barındırılmaktadır:

> 🔗 **RAG sistemi (Kaggle):** https://www.kaggle.com/code/horjok/bilgi-kimde

## 🗂️ Veri şeması

```json
{
  "kisi": "Ada Ozdemir",
  "cv_metni": "Ada Ozdemir, C# ve .NET geliştirme alanında uzmanlaşmış...",
  "yetkinlikler": [
    { "ad": "C#/.NET", "seviye": "ileri", "kanit": "..." }
  ]
}
```

Seviyeler: `baslangic · orta · ileri · uzman`. Yetkinlik havuzu 5 kategoride 51 yetkinliktir
(Backend, Frontend, Veri/Veritabanı, DevOps/Altyapı, Mimari/Mühendislik).

## ⚙️ Teknik notlar

- **Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (çok dilli, Türkçe destekli).
- **Benzerlik:** motor tarafında `util.cos_sim`; Streamlit arayüzünde sayısal olarak özdeş numpy kosinüsü.
- **İzole embedding süreci:** Bu ortamda (Windows + Python 3.13) torch, Streamlit sunucu süreci
  içinde segfault verdiği için embedding hesabı temiz bir **alt süreçte** (`embed_worker.py`)
  yapılır; arayüz süreci torch işlemi çalıştırmaz. Model yine canlı kullanılır.

## 🧩 Yol haritası

- [x] Sentetik veri üreteci
- [x] Yatkınlık motoru (embedding + kosinüs)
- [x] Bus factor / risk analizi
- [x] Streamlit arayüzü + demo
- [x] RAG sorgu katmanı (Kaggle)

---

_Yerel prototip — LLM capstone projesi._
