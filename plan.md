# Rencana Teknis — Pipeline Text Mining ABSA Berita Ekonomi CNBC Indonesia

Dokumen ini merangkum **keputusan teknis** (implementasi text mining) yang disepakati dalam sesi diskusi. Untuk kerangka penelitian, metodologi, dan anotasi, lihat `ringkasan_penelitian_absa_berita_ekonomi.md`. Dokumen ini fokus pada **bagaimana data ditarik, diparsing, difilter, dan disimpan**.

- **Repositori:** <https://github.com/erik580/Text-mining---AOL-Project>
- **Bahasa & stack:** Python, BeautifulSoup (parser utama), requests, pandas.
- **Bentuk kerja:** satu notebook `.ipynb` per fase, reproducible, dijalankan di lokal.
- **Sumber data:** CNBC Indonesia, fokus kanal ekonomi.
- **Periode:** 1 Januari 2021 – 22 Oktober 2026.
- **Model yang dibandingkan (fase hilir):** IndoBERT vs XLM-RoBERTa.

---

## 1. Struktur fase (notebook)

| Notebook | Isi | Output ke disk |
|---|---|---|
| `01_crawling_scraping.ipynb` | Kumpulkan URL + scrape isi artikel (fase ini sudah dibuat) | `data/raw/urls_*.csv`, `data/raw/articles_*.jsonl` |
| `02_preprocessing_sentence_split.ipynb` | Cleaning + pecah kalimat (+id) | `data/processed/sentences.csv` |
| `03_bertopic_aspek.ipynb` | BERTopic, penamaan & penggabungan aspek | `data/processed/sentences_with_aspect.csv` |
| `04_sampling_anotasi.ipynb` | Stratified sampling + sheet anotator + Cohen's Kappa | `data/annotation/to_annotate.csv` |
| `05_split_dataset.ipynb` | Split train/val/test **per artikel** | `data/final/{train,val,test}.csv` |
| `06_train_compare_models.ipynb` | Fine-tune IndoBERT vs XLM-R, evaluasi | `results/` |
| `07_analisis_per_aspek.ipynb` | Model terbaik ke data tak berlabel, tren per aspek | `results/` |

Folder `data/` hanya tempat transit antar-notebook (bukan package code).

---

## 2. Fase 01 — Crawling & Scraping (sudah diimplementasikan)

### 2.1 Domain & kanal

- **Kanal utama (ekonomi):** `market` (pasar modal, moneter, fiskal, nilai tukar, perbankan).
- **Kanal tambahan opsional:** `research`, `syariah` (masih dalam ranah ekonomi; aktif saat mode full).
- **Hindari** kanal non-ekonomi (lifestyle, tech, entertainment) agar aspek BERTopic tetap bersih.

### 2.2 Sumber URL artikel — DUA JALUR

**Jalur A — Sitemap (TERBUKTI BEKERJA, jalur utama saat ini).**
- Sitemap index: `https://www.cnbcindonesia.com/sitemap.xml` → sub-sitemap per kanal.
- Per kanal: `sitemap_news.xml` (artikel terbaru, metadata kaya) dan `sitemap_web.xml` (arsip lebih dalam).
- Tiap entri memberi: `loc` (URL), `news:publication_date`, `news:title`, `news:keywords`.
- Statis, legal (disediakan resmi), dan parsing-nya sudah diverifikasi.

**Jalur B — Halaman indeks per tanggal (PERLU VERIFIKASI LOKAL).**
- URL yang disarankan: `https://www.cnbcindonesia.com/market/indeks/5?date=DD/MM/YYYY&tipe=artikel`.
- **Kendala terverifikasi:** halaman ini **dirender JavaScript**; `requests + BeautifulSoup` **tidak** melihat daftar artikelnya (fetch statis mengembalikan halaman kosong). Selain itu `robots.txt` men-*disallow* `/indeks/*` dan URL ber-`?` untuk Googlebot (untuk `User-agent: *` masih `Allow`).
- **Cara memakainya:** daftar artikel pada halaman indeks dimuat via **panggilan API internal** (JSON) di belakang layar. Endpoint persisnya **harus diintip dari lokal** (buka DevTools → tab Network → filter XHR saat mengganti tanggal). Setelah endpoint diketahui, panggil langsung dengan `requests` (tetap tanpa browser), lalu parsing isi tiap artikel dengan BeautifulSoup.
- **Kegunaan jalur B:** memungkinkan filter **per tanggal/tahun** yang presisi (2021–2026) dan menjangkau arsip lebih dalam daripada sitemap.

> **Keputusan:** mulai dengan **Jalur A** untuk membuktikan pipeline (sudah jalan). Lengkapi dengan **Jalur B** setelah endpoint API indeks dikonfirmasi dari lokal. Keduanya menghasilkan daftar URL yang diparsing isinya dengan cara yang sama (lihat 2.4).

### 2.3 Pola URL artikel (sumber ID & metadata)

```
https://www.cnbcindonesia.com/market/20261022185649-17-778698/kemenkeu-tanggapi-efek-msci-...
                                 └kanal┘ └─timestamp──┘ └┘ └─id─┘ └────────── slug ──────────┘
```
- `20261022185649` = waktu terbit (YYYYMMDDhhmmss) → sumber tanggal andal.
- kode tipe: `17` = artikel teks; `18`/`19` = **video** (dibuang).
- `778698` = **id_artikel unik** → dipakai sebagai kunci dedupe & resume.

### 2.4 Ekstraksi isi artikel (TERVERIFIKASI pada `test.html`)

Diverifikasi terhadap satu artikel market asli, bukan asumsi:

- **Judul & tanggal** → dari **JSON-LD** `<script type="application/ld+json">` blok `@type: NewsArticle` (`headline`, `datePublished`). Fallback: meta `og:`.
- **Isi artikel** → dari kontainer **`div.detail-text`** (kumpulkan paragraf `<p>`).
  - **PENTING:** JSON-LD CNBC **tidak** memuat `articleBody`. Jadi isi WAJIB dari `div.detail-text`, bukan JSON-LD.
- **Noise yang dibuang di dalam kontainer** (terbukti ada di HTML asli):
  - `<script>` iklan (FMG_Tag / cads-generic), `<style>`, `<iframe>`.
  - `<div>` banner Google Ads (`id` mengandung `div-gpt-ad` / `google_ads`) dan parallax.
  - Blok video sisipan: `a.autovideocnbc`, `div.video_detail`, `div.sisip_embed`.
  - `div.read-full-article` (placeholder kosong).
  - Kode penulis di akhir, mis. `(haa/haa)` / `(rob/rob)` → regex `\( [a-z]{2,4} / [a-z]{2,4} \)$`.
  - `<p>` kosong dan penanda embed `[Gambas:...]`.

Hasil uji pada `test.html`: judul & tanggal benar, isi 2.009 karakter bersih, teks video tidak bocor, kode penulis tidak bocor.

### 2.5 Filtering

**Dibuang (noise / bukan data):**
- Artikel **video** (kode tipe 18/19 atau judul diawali "Video:").
- **Duplikat** (dedupe berdasarkan `id_artikel`).
- Artikel **kosong / terlalu pendek** (isi < 150 karakter ditandai `isi_kosong_atau_pendek`).
- URL ber-`?` dan halaman `/indeks/` sebagai target scrape (sesuai `robots.txt`).

**TIDAK difilter:**
- Tidak menyaring berdasarkan tebakan sentimen (hindari bias distribusi label).
- Tidak menyaring per perusahaan tertentu (berbeda dari jurnal Wulandhari dkk. 2025 yang menyaring 5 saham LQ45 untuk tujuan prediksi harga; penelitian kita ABSA berita ekonomi umum).

### 2.6 Checkpoint & resume (anti-ulang saat koneksi putus)

- **Crawl:** hasil URL disimpan ke `urls_*.csv`. Jika file sudah ada, crawl dilewati (hapus file untuk crawl ulang).
- **Scrape:** tiap artikel ditulis *append* satu baris JSON ke `articles_*.jsonl` dengan penulisan line-buffered (langsung ter-flush). Saat sel dijalankan ulang, `id_artikel` yang sudah ada di file **dilewati** → melanjutkan, bukan mengulang.
- `KeyboardInterrupt` (Stop) ditangani rapi: progres tersimpan.

### 2.7 Etika & kesopanan ke server

- `User-Agent` jelas (identitas riset akademik + kontak).
- Jeda antar-permintaan: `REQUEST_DELAY = 1.5` detik artikel, `SITEMAP_DELAY = 1.0` detik.
- Retry dengan exponential backoff (2s, 4s, 8s); `404`/`410` dilewati tanpa retry.
- Catatan hukum: `robots.txt` hanya aturan teknis; tim tetap wajib mematuhi Ketentuan Penggunaan CNBC Indonesia. Data untuk riset akademik non-komersial.

### 2.8 Skema penyimpanan

**`data/raw/urls_*.csv`** (hasil crawl):
`id_artikel, kanal, url, tanggal_terbit, judul, keywords`

**`data/raw/articles_*.jsonl`** (hasil scrape, satu artikel per baris):
`id_artikel, kanal, url, tanggal_terbit, judul_sitemap, keywords, judul, tanggal_terbit_html, isi_mentah, isi_panjang, metode_ekstraksi, status, tanggal_scrape`

- Format **JSONL** dipilih untuk artikel karena isi panjang & mengandung koma/kutip/newline (CSV rapuh).
- `isi_mentah` disimpan **apa adanya** (cleaning mendalam & pemecahan kalimat → Fase 02).
- Ekspor ringkas `articles_*.csv` (tanpa `isi_mentah`) untuk inspeksi cepat.

### 2.9 Mode PILOT vs FULL

- `PILOT = True`: rentang pendek (20–22 Okt 2026), kanal `market` saja, maksimum 15 artikel, hanya `sitemap_news.xml`. Untuk uji alur + uji resume.
- `PILOT = False`: periode penuh 2021–2026, kanal `market`+`research`+`syariah`, `sitemap_news.xml` + `sitemap_web.xml`.

### 2.10 Risiko diketahui / pekerjaan lanjutan

- **Cakupan arsip 2021:** `sitemap_web.xml` kemungkinan tidak mundur sampai 2021. Jalur B (indeks/API) atau arsip publik (Wayback Machine CDX API) menjadi cadangan untuk menambang URL lama. Perlu diverifikasi saat run lokal.
- **Endpoint API indeks:** belum dikonfirmasi; harus diintip dari DevTools lokal.
- **Variasi format artikel:** live blog, infografis, atau artikel multi-halaman mungkin perlu penyesuaian selector kecil. Fondasi untuk artikel teks standar sudah terverifikasi.

---

## 3. Lingkungan lokal

```powershell
# dari folder project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# jalankan notebook: buka 01_crawling_scraping.ipynb di VS Code (kernel .venv), Run All
# atau headless:
pip install jupyter
jupyter nbconvert --to notebook --execute --inplace 01_crawling_scraping.ipynb
```

Dependency inti (lihat `requirements.txt`): `requests`, `beautifulsoup4`, `lxml`, `pandas`, `pyarrow`, `tqdm`.

---

## 4. Status saat ini

- [x] Eksplorasi struktur situs CNBC (sitemap, pola URL, robots.txt).
- [x] Verifikasi ekstraksi isi pada HTML artikel asli (`test.html`).
- [x] `01_crawling_scraping.ipynb` dibuat (crawl sitemap + scrape + checkpoint/resume).
- [x] `requirements.txt` dibuat.
- [ ] Pilot dijalankan di lokal (verifikasi end-to-end + resume).
- [ ] Konfirmasi endpoint API indeks (Jalur B) untuk filter per tahun.
- [ ] Verifikasi cakupan arsip sampai 2021.
- [ ] Fase 02 dan seterusnya.
