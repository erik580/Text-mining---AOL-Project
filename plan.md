# Rencana: Alur Proses, Pembagian Tugas, & Timeline

Dokumen ini menjelaskan **alur kerja teknis** pipeline penelitian (dari scraping sampai evaluasi sentimen), **pembagian tugas antar anggota tim**, dan **timeline realistis**. Untuk kerangka penelitian & metodologi, lihat `ringkasan_penelitian_absa_berita_ekonomi.md`.

- **Repositori:** <https://github.com/erik580/Text-mining---AOL-Project>
- **Topik:** Perbandingan model klasifikasi (IndoBERT vs XLM-RoBERTa) untuk ABSA dari hasil topic modeling — berita ekonomi 4 bank (BCA, BNI, BRI, Mandiri), CNBC Indonesia, periode 20 Okt 2024 – 20 Okt 2026.
- **Tim:** **2 orang** — **P1** (saya) & **P2**.
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

## 3. Pembagian tugas (2 orang)

Karena tim hanya dua orang, beban dibagi rata sepanjang pipeline, dengan P1 memimpin hulu (data) dan P2 memimpin hilir (model). Anotasi manual **dikerjakan berdua** dengan irisan bersama untuk Cohen's Kappa.

### P1 — Data & Topic Modeling *(= saya)*
- **Fase 01:** scraping & crawling 4 bank (sudah jalan), menjalankan full-run, menjaga checkpoint.
- **EDA & Quality Data:** statistik dataset (jumlah artikel per bank/keyword, distribusi waktu, duplikat, panjang isi, artikel kosong/pendek), pembersihan masalah kualitas.
- **Fase 02:** preprocessing + pemecahan kalimat (menghasilkan `sentences.csv`).
- **Fase 03:** menjalankan BERTopic + memimpin penamaan/penggabungan klaster → aspek (rujuk **FiQA Corporate & Stock**), menyusun `daftar_aspek.md`.
- Pemelihara repositori (struktur folder, README, requirements, merge PR).
- **Anotator #1** pada Fase 04.

### P2 — Anotasi, Modeling, & Evaluasi
- **Fase 04:** memimpin penyusunan **pedoman sentimen**, menyiapkan mekanisme anotasi, menghitung **Cohen's Kappa**. **Anotator #2**.
- **Fase 05:** split train/val/test per artikel.
- **Fase 06:** fine-tune IndoBERT & XLM-RoBERTa, menjaga kesetaraan setup, uji signifikansi (McNemar/bootstrap), confusion matrix.
- **Fase 07:** analisis sentimen per aspek + visualisasi tren.
- Menyusun tabel hasil & bagian evaluasi untuk naskah.

### Bersama (P1 & P2)
- **Fase 04 — Anotasi manual:** sepakati pedoman, bagi batch acak, labeli porsi masing-masing, kerjakan irisan Kappa bersama, bahas & catat kasus abu-abu.
- Penulisan naskah: P1 bagian Data/Preprocessing/Topic Modeling; P2 bagian Anotasi/Model/Evaluasi; keterbatasan & kesimpulan ditulis bersama.

---

## 4. Timeline realistis (1 November – 19 Desember)

Durasi ±7 minggu (lebih longgar karena tim 2 orang, beban modeling ditanggung P2). Angka bisa digeser; yang penting urutan dependensi antar-fase.

### Minggu 1 — 1 Nov s/d 7 Nov: Data siap
| Hari | P1 | P2 |
|---|---|---|
| 1–3 Nov | Jalankan **full-run scraping** 4 bank (Fase 01), pantau checkpoint | Studi literatur FiQA + setup lingkungan training (Colab/GPU), uji load IndoBERT & XLM-R |
| 4–5 Nov | **EDA & quality check** dataset mentah (distribusi, duplikat, isi pendek) | Siapkan kerangka notebook Fase 06 (skeleton training) |
| 6–7 Nov | **Fase 02:** preprocessing + pecah kalimat → `sentences.csv` | Review hasil EDA bersama P1 |
| **Milestone** | ✅ `cnbc_4bank_full.csv` + `sentences.csv` siap; laporan EDA | |

### Minggu 2 — 8 Nov s/d 14 Nov: Aspek terbentuk
| Hari | P1 | P2 |
|---|---|---|
| 8–11 Nov | **Fase 03:** jalankan BERTopic, tuning, tangani outlier | Finalkan pipeline training & evaluasi (dry-run data dummy) |
| 12–14 Nov | **Namai & gabung klaster → aspek (FiQA)**, susun `daftar_aspek.md` | Siapkan skrip uji signifikansi (McNemar/bootstrap) |
| **Milestone** | ✅ **Daftar aspek DIKUNCI** + definisi tiap aspek | |

### Minggu 3 — 15 Nov s/d 21 Nov: Pedoman & mulai anotasi
| Hari | P1 | P2 |
|---|---|---|
| 15–16 Nov | **Fase 04:** stratified sampling → `to_annotate.csv` + sheet anotator | Tulis **pedoman sentimen**, sepakati kasus abu-abu (bersama P1) |
| 17–21 Nov | Anotasi batch P1 | Anotasi batch P2 + siapkan irisan Kappa |
| **Milestone** | ✅ Pedoman final; anotasi berjalan | |

### Minggu 4 — 22 Nov s/d 28 Nov: Anotasi selesai + Kappa
| Hari | P1 | P2 |
|---|---|---|
| 22–26 Nov | Selesaikan anotasi porsi P1 + irisan Kappa | Selesaikan anotasi porsi P2 + irisan Kappa |
| 27–28 Nov | Gabung label, bereskan "aspek salah" | **Hitung Cohen's Kappa**; **Fase 05:** split per artikel |
| **Milestone** | | ✅ `labeled.csv` + nilai Kappa; `data/final/*` siap |

### Minggu 5 — 29 Nov s/d 5 Des: Pelatihan & perbandingan model
| Hari | P1 | P2 |
|---|---|---|
| 29 Nov–2 Des | Analisis statistik dataset (bukti "per aspek perlu": campuran sentimen, chi-square) | **Fase 06:** fine-tune IndoBERT & XLM-R (multi-seed) |
| 3–5 Des | Tulis bagian Dataset/Preprocessing/Topic Modeling naskah | Hitung metrik + **uji signifikansi**, confusion matrix |
| **Milestone** | | ✅ `results/` perbandingan model lengkap |

### Minggu 6 — 6 Des s/d 12 Des: Analisis akhir
| Hari | P1 | P2 |
|---|---|---|
| 6–9 Des | Review hasil model bersama, bantu analisis error kualitatif | **Fase 07:** analisis sentimen per aspek + visualisasi tren |
| 10–12 Des | Rapikan repo & reproducibility | Tulis bagian Model & Evaluasi naskah |
| **Milestone** | | ✅ Seluruh hasil & grafik tren selesai |

### Minggu 7 — 13 Des s/d 19 Des: Finalisasi naskah
| Hari | P1 | P2 |
|---|---|---|
| 13–16 Des | Kompilasi naskah, tulis keterbatasan (bersama) | Review akhir hasil & tabel |
| 17–19 Des | Finalisasi & proofread (bersama) | Finalisasi & proofread (bersama) |
| **Milestone** | ✅ **Naskah draf lengkap + seluruh hasil** | |

---

## 5. Catatan

- **Ketergantungan kritis:** Fase 03 (aspek dikunci) harus selesai sebelum Fase 04 (anotasi); Fase 04 (label) harus selesai sebelum Fase 06 (training). **Keterlambatan anotasi adalah risiko terbesar** — mulai sedini mungkin dan kerjakan paralel berdua.
- **Beban modeling di P2** cukup berat (Fase 05–07); P1 membantu di analisis statistik & penulisan agar seimbang.
- **Buffer:** sisakan jeda bila ada libur/ujian; anotasi manual biasanya makan waktu lebih lama dari perkiraan.
- Status & keputusan yang berubah dicatat kembali di dokumen ini + `ringkasan_penelitian_absa_berita_ekonomi.md`.
