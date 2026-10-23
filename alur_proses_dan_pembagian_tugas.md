# Alur Proses, Pembagian Tugas, & Timeline

Dokumen ini menjelaskan **alur kerja teknis** pipeline penelitian (dari scraping sampai evaluasi sentimen), **pembagian tugas antar anggota tim**, dan **timeline realistis**. Untuk kerangka penelitian & metodologi, lihat `ringkasan_penelitian_absa_berita_ekonomi.md`.

- **Repositori:** <https://github.com/erik580/Text-mining---AOL-Project>
- **Topik:** Perbandingan model klasifikasi (IndoBERT vs XLM-RoBERTa) untuk ABSA dari hasil topic modeling — berita ekonomi 4 bank (BCA, BNI, BRI, Mandiri), CNBC Indonesia, periode 20 Okt 2024 – 20 Okt 2026.
- **Tim:** **P1** (saya), **P2**, **P3**. *(Jika tim hanya 2 orang, gabungkan peran P2 & P3 — lihat catatan di akhir.)*
- **Stack:** Python, `requests` + `BeautifulSoup` (scraping), `pandas` (data), BERTopic (topic modeling), Hugging Face `transformers` (IndoBERT & XLM-RoBERTa), dijalankan di lokal (Anaconda) + Google Colab untuk pelatihan model (opsional, bila butuh GPU).

---

## 1. Alur proses (pipeline) end-to-end

Satu notebook `.ipynb` **mandiri (self-contained)** per fase. Folder `data/` hanya tempat transit antar-fase (bukan package code).

| Fase | Notebook | Isi | Input | Output |
|---|---|---|---|---|
| **01** | `01_crawling_scraping.ipynb` | Scraping berita 4 bank dari `/search` CNBC + filter keyword + checkpoint | — | `data/raw/cnbc_4bank_full.csv` |
| **02** | `02_preprocessing_sentence_split.ipynb` | Cleaning ringan + pecah berita menjadi kalimat (+ id) | `cnbc_4bank_full.csv` | `data/processed/sentences.csv` |
| **03** | `03_bertopic_aspek.ipynb` | BERTopic pada kalimat; penamaan & penggabungan klaster → aspek (rujuk FiQA) | `sentences.csv` | `data/processed/sentences_with_aspect.csv` + `daftar_aspek.md` |
| **04** | `04_sampling_anotasi.ipynb` | Stratified sampling + sheet anotator + hitung Cohen's Kappa | `sentences_with_aspect.csv` | `data/annotation/to_annotate.csv`, `labeled.csv` |
| **05** | `05_split_dataset.ipynb` | Split train/val/test **per artikel** | `labeled.csv` | `data/final/{train,val,test}.csv` |
| **06** | `06_train_compare_models.ipynb` | Fine-tune & bandingkan IndoBERT vs XLM-RoBERTa + uji signifikansi | `data/final/*` | `results/` (metrik, confusion matrix) |
| **07** | `07_analisis_per_aspek.ipynb` | Model terbaik → data tak berlabel; distribusi & tren sentimen per aspek | model terbaik + `sentences_with_aspect.csv` | `results/` (grafik, tabel) |

### Diagram ringkas
```
[01 Scraping 4 bank] → [02 Preprocessing & split kalimat] → [03 BERTopic → Aspek (FiQA)]
        → [04 Sampling & Anotasi manual + Kappa] → [05 Split per artikel]
        → [06 Latih & bandingkan IndoBERT vs XLM-R] → [07 Analisis sentimen per aspek]
```

### Prinsip teknis yang dijaga di semua fase
- Teks yang masuk ke BERTopic & model = **kalimat utuh alami**; **tanpa** stemming / stopword-removal (merusak embedding).
- Split **per artikel**, bukan per kalimat (hindari kebocoran data antar-split).
- Label kebenaran = **anotasi manusia**; dilarang memakai LLM/model lain sebagai label akhir di test set.
- Reproducible: seed tetap, checkpoint/resume di fase scraping.

---

## 2. Rincian tiap fase (checklist teknis)

### Fase 01 — Crawling & Scraping *(SUDAH SELESAI; pilot pass)*
- Sumber: `GET /search?query=<kw>&page=<N>&fromdate=20/10/2024&todate=20/10/2026` (format `YYYY/MM/DD`).
- Keyword 4 bank (nama + kode saham): `BCA, BBCA, BRI, BBRI, BNI, BBNI, Mandiri, BMRI` → gabung & dedupe via `id_artikel`.
- Per artikel: judul (`h2`), link, waktu (`span` terakhir), isi (`div.detail-text`, gabung `<p>`). Buang video (kode tipe 18/19).
- Checkpoint: tulis *append* ke CSV; skip `id_artikel` yang sudah ada.
- Fallback bila `/search` beralih ke render JS: endpoint JSON `GET /api/v2/search?query=..&fromdate=..&todate=..&start=0&limit=100`.

### Fase 02 — Preprocessing & pemecahan kalimat
- Buang boilerplate sisa ("baca juga", iklan, dateline, tanda tangan penulis), normalisasi spasi/karakter.
- Pecah isi → kalimat; simpan `id_artikel` + `id_kalimat`.
- Buang kalimat < 5 kata (ambang bisa diuji).

### Fase 03 — BERTopic → aspek
- Embedding kalimat (model Indonesia/multilingual), HDBSCAN cluster.
- Namai klaster dari kata kunci **dan** 5–10 contoh kalimat; gabungkan klaster semakna; rujuk **FiQA (Corporate & Stock)**.
- Tangani outlier (laporkan %, kelompokkan ke "Lainnya" atau keluarkan). **Kunci daftar aspek** sebelum anotasi.

### Fase 04 — Sampling & anotasi
- Stratified sampling per aspek (tiap aspek cukup contoh). Target ~3.000 pasangan (kalimat, aspek); irisan ~100–200 dilabel dua anotator untuk **Cohen's Kappa**.
- Pedoman sentimen (positif/netral/negatif) ditulis & disepakati sebelum mulai; kasus abu-abu dicatat.

### Fase 05 — Split dataset
- Train/val/test **per artikel** (usulan 70/15/15).

### Fase 06 — Latih & bandingkan model
- Fine-tune IndoBERT & XLM-RoBERTa dengan data/split/tuning setara.
- Metrik: Macro-F1 (utama), akurasi, F1 per kelas, F1 per aspek, confusion matrix.
- **Uji signifikansi:** beberapa seed (rata ± simpangan) + uji McNemar/bootstrap.

### Fase 07 — Analisis per aspek
- Model terbaik → prediksi data tak berlabel → distribusi sentimen per aspek + tren waktu.
- Catatan: gambaran tren, **bukan** bukti kualitas model (tidak ada label manusia di sana).

---

## 3. Pembagian tugas

Peran dibagi agar paralel, tetapi tiap orang paham keseluruhan pipeline. Anotasi manual **wajib** melibatkan ≥ 2 orang (untuk Kappa).

### P1 — Data Engineering & Quality *(= saya)*
- **Fase 01:** scraping & crawling 4 bank (sudah jalan), menjalankan full-run, menjaga checkpoint.
- **EDA & Quality Data:** statistik dataset (jumlah artikel per bank/keyword, distribusi waktu, duplikat, panjang isi, artikel kosong/pendek), pembersihan masalah kualitas.
- **Fase 02:** preprocessing + pemecahan kalimat (menghasilkan `sentences.csv`).
- Pemelihara repositori (struktur folder, README, requirements, merge PR).

### P2 — Topic Modeling & Aspek
- **Fase 03:** menjalankan BERTopic, tuning jumlah topik/parameter, menangani outlier.
- Memimpin **penamaan & penggabungan klaster** menjadi aspek, menyusun `daftar_aspek.md` dengan definisi + contoh batas tiap aspek, merujuk taksonomi **FiQA (Corporate & Stock)**.
- **Fase 05:** menyiapkan split train/val/test per artikel.
- Ikut anotasi sebagai **anotator ke-2** (untuk Kappa).

### P3 — Modeling & Evaluasi
- **Fase 06:** fine-tune IndoBERT & XLM-RoBERTa, menjaga kesetaraan setup, menjalankan uji signifikansi.
- **Fase 07:** analisis sentimen per aspek + visualisasi tren.
- Menyusun tabel hasil & bagian evaluasi untuk naskah.
- Ikut anotasi sebagai **anotator pembanding** pada irisan Kappa.

### Bersama (semua anggota)
- **Fase 04 — Anotasi manual:** penyusunan pedoman sentimen, pembagian batch acak, pelabelan porsi masing-masing, pembahasan kasus abu-abu. (P1 & P2 jadi dua anotator utama; P3 melabel irisan Kappa.)
- Penulisan naskah per bagian sesuai peran.

---

## 4. Timeline realistis (1 November – 12 Desember)

Durasi ±6 minggu. Angka bisa digeser sesuai kemajuan; yang penting urutan dependensi antar-fase.

### Minggu 1 — 1 Nov s/d 7 Nov: Data siap
| Hari | P1 | P2 | P3 |
|---|---|---|---|
| 1–3 Nov | Jalankan **full-run scraping** 4 bank (Fase 01), pantau checkpoint | Studi literatur FiQA + contoh taksonomi aspek perbankan | Setup lingkungan training (Colab/GPU), uji load IndoBERT & XLM-R |
| 4–5 Nov | **EDA & quality check** dataset mentah (distribusi, duplikat, isi pendek) | Rancang draf awal kategori aspek (hipotesis sementara) | Siapkan kerangka notebook Fase 06 (skeleton training) |
| 6–7 Nov | **Fase 02:** preprocessing + pecah kalimat → `sentences.csv` | Review hasil EDA bersama P1 | Review skeleton bersama tim |
| **Milestone** | ✅ `cnbc_4bank_full.csv` + `sentences.csv` siap; laporan EDA | | |

### Minggu 2 — 8 Nov s/d 14 Nov: Aspek terbentuk
| Hari | P1 | P2 | P3 |
|---|---|---|---|
| 8–10 Nov | Dukung P2 (format data untuk BERTopic), mulai tulis bagian Metodologi-Data | **Fase 03:** jalankan BERTopic, tuning, tangani outlier | Finalkan pipeline training & evaluasi (dry-run data dummy) |
| 11–14 Nov | Bantu validasi penetapan aspek (sampel) | **Namai & gabung klaster → aspek (FiQA)**, susun `daftar_aspek.md` | Siapkan skrip uji signifikansi (McNemar/bootstrap) |
| **Milestone** | | ✅ **Daftar aspek DIKUNCI** + definisi tiap aspek | |

### Minggu 3 — 15 Nov s/d 21 Nov: Pedoman & mulai anotasi
| Hari | P1 | P2 | P3 |
|---|---|---|---|
| 15–16 Nov | **Fase 04:** stratified sampling → `to_annotate.csv` + sheet anotator | Tulis **pedoman sentimen** (bersama) | Review pedoman, sepakati kasus abu-abu |
| 17–21 Nov | Anotasi batch P1 | Anotasi batch P2 | Anotasi irisan Kappa + batch bantu |
| **Milestone** | | | ✅ Pedoman final; anotasi berjalan |

### Minggu 4 — 22 Nov s/d 28 Nov: Anotasi selesai + Kappa
| Hari | P1 | P2 | P3 |
|---|---|---|---|
| 22–26 Nov | Lanjutkan & selesaikan anotasi porsi P1 | Selesaikan anotasi porsi P2 | Selesaikan irisan Kappa |
| 27–28 Nov | Gabung label, **hitung Cohen's Kappa**, bereskan "aspek salah" | Review konsistensi label | **Fase 05:** split train/val/test per artikel |
| **Milestone** | | ✅ `labeled.csv` + nilai Kappa; `data/final/*` siap | |

### Minggu 5 — 29 Nov s/d 5 Des: Pelatihan & perbandingan model
| Hari | P1 | P2 | P3 |
|---|---|---|---|
| 29 Nov–2 Des | Tulis bagian Dataset & Preprocessing naskah | Analisis statistik dataset (bukti "per aspek perlu": campuran sentimen, chi-square) | **Fase 06:** fine-tune IndoBERT & XLM-R (multi-seed) |
| 3–5 Des | Review hasil model bersama | Bantu analisis error kualitatif | Hitung metrik + **uji signifikansi**, confusion matrix |
| **Milestone** | | | ✅ `results/` perbandingan model lengkap |

### Minggu 6 — 6 Des s/d 12 Des: Analisis akhir & naskah
| Hari | P1 | P2 | P3 |
|---|---|---|---|
| 6–8 Des | Kompilasi naskah, rapikan repo & reproducibility | Tulis bagian Topic Modeling & Aspek | **Fase 07:** analisis sentimen per aspek + visualisasi |
| 9–12 Des | Finalisasi naskah + keterbatasan (bersama) | Review akhir | Review akhir + grafik tren |
| **Milestone** | | ✅ **Naskah draf lengkap + seluruh hasil** | |

---

## 5. Catatan

- **Jika tim hanya 2 orang (P1 & P2):** gabungkan peran P3 ke P2 (P2 menangani modeling + evaluasi + anotator ke-2). Anotasi Kappa tetap bisa dilakukan oleh P1 & P2 pada irisan bersama. Timeline mundur ±1 minggu karena beban modeling ditanggung satu orang.
- **Ketergantungan kritis:** Fase 03 (aspek dikunci) harus selesai sebelum Fase 04 (anotasi); Fase 04 (label) harus selesai sebelum Fase 06 (training). Keterlambatan anotasi adalah risiko terbesar — mulai sedini mungkin.
- **Buffer:** sisakan jeda bila ada libur/ujian; anotasi manual biasanya makan waktu lebih lama dari perkiraan.
- Status & keputusan yang berubah dicatat kembali di dokumen ini + `ringkasan_penelitian_absa_berita_ekonomi.md`.
