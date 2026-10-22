# Ringkasan Penelitian: Analisis Sentimen Berbasis Aspek pada Berita Ekonomi

**Judul kerja:** *Analisis Sentimen Berbasis Aspek Menggunakan Topic Modeling: Perbandingan IndoBERT dan XLM-RoBERTa pada Berita Ekonomi CNBC Indonesia*

> Dokumen ini untuk menyamakan pemahaman dengan rekan anotator. Angka dan keputusan yang bertanda **(usulan)** boleh disesuaikan setelah didiskusikan, tetapi harus dicatat begitu disepakati.

---

## 1. Apa yang kita kerjakan, dan mengapa

**Tujuan:** membandingkan dua model klasifikasi sentimen (positif / netral / negatif) pada berita ekonomi berbahasa Indonesia, dengan sentimen dinilai **per aspek**, bukan per berita. Dua model yang dibandingkan adalah **IndoBERT** (monolingual, dilatih penuh pada korpus bahasa Indonesia) dan **XLM-RoBERTa** (multilingual, dilatih pada 100 bahasa).

**Klaim yang diangkat:** *sentimen berita ekonomi perlu dianalisis per aspek, dan model yang dilatih penuh pada bahasa Indonesia (IndoBERT) memberikan hasil analisis yang lebih baik daripada model multilingual (XLM-RoBERTa) pada tugas ini.*

**Hipotesis utama:** model bahasa monolingual yang pretraining-nya sepenuhnya pada bahasa Indonesia menangkap nuansa leksikal dan domain berita ekonomi Indonesia lebih baik daripada model multilingual yang kapasitasnya terbagi ke banyak bahasa, sehingga menghasilkan Macro-F1 lebih tinggi pada ABSA berita ekonomi. Hipotesis ini diuji secara empiris; hasil apa pun (IndoBERT unggul, XLM-RoBERTa unggul, atau selisih tidak signifikan) dilaporkan jujur beserta uji signifikansinya.

> **Catatan metodologis:** IndoBERT (resep BERT) dan XLM-RoBERTa (resep RoBERTa) berbeda tidak hanya pada cakupan bahasa (monolingual vs multilingual) tetapi juga pada resep dan korpus pretraining. Karena itu, selisih skor adalah efek **gabungan** dari cakupan bahasa dan perbedaan pretraining, bukan isolasi satu variabel tunggal. Ini dituliskan di bagian keterbatasan. Perbandingan "model utuh" semacam ini lazim dan diterima di literatur applied NLP.

**Mengapa per aspek?** Satu berita ekonomi sering memuat kabar yang bertentangan, misalnya "rupiah melemah, tetapi IHSG ditutup menguat". Kalau seluruh berita hanya diberi satu label, anotator terpaksa memilih satu sentimen dan hasilnya ambigu. Dengan ABSA, kalimat pertama berlabel negatif untuk aspek nilai tukar, dan kalimat kedua positif untuk aspek pasar modal.

**Mengapa topic modeling?** Aspek tidak kita tentukan dari nol. Aspek diturunkan dari data lewat BERTopic, lalu dinamai dan dirapikan oleh kita. Topic modeling juga membantu menetapkan aspek untuk seluruh korpus (~10.000 berita), sedangkan anotasi manual hanya untuk sebagian.

**Yang dibandingkan hanya dua model sentimen.** Topic modeling bukan objek perbandingan. Ia berada di hulu pipeline untuk menghasilkan aspek.

---

## 2. Alur keseluruhan

```
Crawling & scraping berita CNBC Indonesia
        ↓
Cleaning ringan + pemecahan berita menjadi kalimat
        ↓
BERTopic pada kalimat → klaster topik
        ↓
Penamaan & penggabungan klaster → daftar aspek (10–15), DIKUNCI
        ↓
Validasi penetapan aspek (sampel)
        ↓
Anotasi sentimen manual per pasangan (kalimat, aspek)
        ↓
Split train / validation / test per artikel
        ↓
Latih & bandingkan dua model
        ↓
Model terbaik dipakai pada data tak berlabel → analisis per aspek
```

---

## 3. Studi literatur dan posisi penelitian

Perbandingan model monolingual (khusus satu bahasa) melawan model multilingual pada tugas NLP berbahasa Indonesia sudah menjadi pertanyaan yang berulang di beberapa studi. Temuan dari studi-studi ini tidak seragam: pada sebagian tugas model monolingual unggul, pada sebagian lain selisihnya tipis atau model multilingual justru menang. Ini yang membuat hipotesis kita perlu diuji langsung pada domain berita ekonomi, bukan diasumsikan.

Beberapa rujukan yang relevan untuk memosisikan penelitian ini:

1. **Perbandingan IndoBERT vs XLM-RoBERTa pada analisis sentimen ulasan aplikasi SATUSEHAT.** Studi ini membandingkan kedua model pada data ulasan pengguna berbahasa Indonesia dan menjadi pembanding langsung paling dekat dengan pekerjaan kita, karena memakai pasangan model yang sama. *(Lengkapi sitasi: penulis, tahun, venue/jurnal, dan temuan utamanya — model mana yang unggul dan berapa selisih F1-nya.)*

2. **Analisis emosi pada data Twitter berbahasa Indonesia.** Relevan sebagai contoh penerapan model transformer untuk klasifikasi afektif pada teks pendek berbahasa Indonesia, sekaligus pembanding karakteristik data media sosial melawan data berita yang lebih formal. *(Lengkapi sitasi: penulis, tahun, venue, model yang dibandingkan, dan temuan.)*

3. **Analisis sentimen/emosi pada ulasan ChatGPT (skripsi UMN, 2026).** Relevan sebagai studi terbaru yang membandingkan model pada ulasan berbahasa Indonesia. *(Lengkapi sitasi: penulis lengkap, judul skripsi, program studi, Universitas Multimedia Nusantara, 2026, dan temuan utamanya.)*

> **Catatan referensi:** Ketiga rujukan di atas masih berupa kerangka. Detail penulis, tahun, venue, dan temuan **harus diisi dari sumber asli** sebelum naskah final. Jangan mengutip angka atau kesimpulan sebelum diverifikasi dari dokumen aslinya.

**Celah yang kita isi (research gap):** studi-studi di atas umumnya bekerja pada **analisis sentimen tingkat dokumen/kalimat** dan pada domain **ulasan atau media sosial**. Belum banyak yang menguji perbandingan IndoBERT vs XLM-RoBERTa pada **analisis sentimen berbasis aspek (ABSA)** untuk domain **berita ekonomi berbahasa Indonesia**, di mana aspek diturunkan lewat topic modeling. Kombinasi inilah posisi kebaruan penelitian kita: (a) domain berita ekonomi, (b) setting ABSA dengan aspek dari BERTopic, dan (c) pengujian hipotesis monolingual vs multilingual dengan uji signifikansi yang eksplisit.

---

## 4. Pengumpulan data: crawling dan scraping

Dua tahap yang berbeda:

- **Crawling:** menelusuri halaman indeks atau kategori ekonomi di situs untuk **mengumpulkan daftar URL** artikel.
- **Scraping:** membuka tiap URL lalu **mengekstrak isinya**: judul, tanggal terbit, kategori/kanal, isi artikel, dan URL.

Catatan praktis:

- Simpan **data mentah** terpisah dari data hasil olahan, supaya proses bisa diulang.
- Simpan metadata (tanggal, kategori, URL) karena dibutuhkan untuk analisis tren dan penelusuran balik.
- Buang duplikat (URL atau judul sama) dan artikel yang isinya kosong atau terlalu pendek.
- Tentukan dan catat rentang waktu serta kategori yang diambil.
- Periksa `robots.txt` dan ketentuan penggunaan CNBC Indonesia, dan atur jeda antar permintaan agar tidak membebani server.

### 4.1 Keputusan yang sudah difinalkan (hasil sesi teknis)

- **Periode:** 1 Januari 2021 – 22 Oktober 2026 (~5,8 tahun). Mengikuti preseden Wulandhari dkk. (2025) yang memakai ~5 tahun berita CNBC Indonesia untuk prediksi harga saham; lihat bagian 3.
- **Kanal:** utama `market` (ekonomi/pasar modal); tambahan opsional `research`, `syariah`. Kanal non-ekonomi tidak diambil agar aspek BERTopic tetap bersih.
- **Sumber URL:** `robots.txt` CNBC sudah **diverifikasi** — halaman indeks (`/indeks/*`) dan URL ber-`?` di-*disallow* untuk Googlebot, dan halaman indeks dirender JavaScript sehingga tidak bisa diambil dengan BeautifulSoup biasa. Karena itu sumber URL utama adalah **sitemap resmi** (`sitemap_news.xml` + `sitemap_web.xml`), dengan jalur indeks/API sebagai pelengkap yang perlu diverifikasi dari lokal.
- **Atribut yang diambil:** `id_artikel`, kanal, URL, tanggal terbit, judul, keywords, dan isi mentah. Minimal mengikuti jurnal preseden (judul, waktu terbit, isi) + metadata tambahan untuk reproducibility.
- **Ekstraksi isi:** sudah **diverifikasi** pada HTML artikel asli — judul & tanggal dari JSON-LD (`NewsArticle`), isi dari kontainer `div.detail-text`. Artikel video dan duplikat dibuang; tidak ada penyaringan berdasarkan sentimen atau perusahaan tertentu.

> **Detail teknis implementasi** (metode BeautifulSoup, selector, checkpoint/resume, skema file, mode pilot) didokumentasikan terpisah di **`plan.md`**. Notebook dijalankan dari repositori: <https://github.com/erik580/Text-mining---AOL-Project>.

---

## 5. Preprocessing

Prinsip utama: **teks yang masuk ke BERTopic harus berupa kalimat utuh yang alami**, karena model embedding memahami makna dari urutan kata.

| Langkah | Lakukan? | Keterangan |
|---|---|---|
| Hapus HTML, URL, dan boilerplate ("baca juga", iklan, dateline, tanda tangan penulis) | Ya | Noise ini sering membentuk klaster sampah |
| Normalisasi ringan (spasi berlebih, karakter aneh) | Ya | |
| **Pecah berita menjadi kalimat** (atau paragraf) | **Ya, sebelum BERTopic** | Simpan `id_artikel` dan `id_kalimat` |
| Buang kalimat yang terlalu pendek (misalnya < 5 kata) | Ya | Batas bisa diuji |
| Tokenisasi, stemming, lemmatisasi, dan stopword removal pada teks yang di-embed | **Tidak** | Merusak makna kalimat untuk embedding |
| Stopword removal pada langkah pembuatan kata kunci topik | Opsional | Dilakukan di vectorizer BERTopic, bukan pada teksnya |

**Mengapa dipecah per kalimat?** BERTopic hanya memberi **satu topik per unit** yang dimasukkan. Supaya satu berita bisa punya beberapa aspek, beritanya harus dipecah dulu. Kalau hasil per kalimat terlalu berisik (banyak outlier), alternatifnya memakai paragraf sebagai unit.

---

## 6. Dari topik menjadi aspek

BERTopic menghasilkan klaster berdasarkan **kemiripan makna** (embedding), lalu mengambil kata kunci dari tiap klaster. Klaster itu **belum punya nama aspek**. Tugas kita:

1. **Menamai klaster** dengan melihat kata kunci **dan** 5–10 contoh kalimat representatif per klaster, bukan kata kunci saja.
2. **Menggabungkan klaster yang maknanya sama.** Dari ~20 klaster diharapkan menjadi sekitar 10–15 aspek.
3. **Menyediakan kategori "Lainnya"** untuk klaster tanpa makna jelas.
4. **Menangani outlier:** laporkan persentasenya, lalu kelompokkan ke "Lainnya" atau keluarkan dari ABSA.
5. **Merujuk studi sebelumnya** sebagai pembanding penamaan (misalnya taksonomi FiQA 2018 dan studi topic modeling berita ekonomi Indonesia). Hasilnya taksonomi yang diadaptasi dan divalidasi pada data CNBC, bukan taksonomi yang diwariskan.
6. **Menulis definisi satu kalimat dan contoh batas tiap aspek** (apa yang termasuk dan tidak) di pedoman anotasi.
7. **Mengunci daftar aspek sebelum anotasi sentimen dimulai.**

---

## 7. Anotasi manual (bagian terpenting untuk anotator)

### 7.1 Apa yang dianotasi

Yang dianotasi adalah sentimen **per pasangan (kalimat, aspek)**. Aspeknya sudah ditetapkan BERTopic dan kita periksa; anotator hanya memutuskan sentimennya.

| id_berita | kalimat | aspek (dari BERTopic) | sentimen (label manusia) |
|---|---|---|---|
| 0123 | "Penerimaan pajak turun 5% dibanding tahun lalu." | Kebijakan fiskal | negatif |
| 0123 | "IHSG ditutup menguat tipis." | Pasar modal | positif |

Jika aspek yang ditetapkan jelas salah, tandai **"aspek salah"**. Baris itu dikoreksi atau dibuang.

### 7.2 Beban kerja *(usulan)*

- Target sekitar **3.000 pasangan (kalimat, aspek)** berlabel, dibagi dua: **1.500 per anotator**.
- Bagikan data ke anotator secara **acak**, bukan per topik atau per berita, supaya perbedaan gaya anotator tidak tercampur dengan perbedaan topik.
- Sampel dipilih **stratified**: tiap aspek harus punya cukup contoh, dan jangan hanya mengambil kalimat yang gampang. Aspek dengan sampel terlalu sedikit digabung atau dikeluarkan dari evaluasi per aspek.
- Sisihkan **irisan kecil (~100–200 pasangan)** yang dilabel **kedua** anotator, lalu hitung Cohen's Kappa. Ini tidak butuh anotator ketiga, dan hasilnya menjadi bukti kesulitan tugas ini.

### 7.3 Pedoman sentimen (harus ditulis sebelum mulai)

Sepakati dan tulis satu definisi tunggal untuk sudut pandang penilaian. **Usulan:** nilai sentimen sebagai dampaknya terhadap kondisi aspek itu dari sudut pandang ekonomi atau pasar (kabar baik, buruk, atau sekadar laporan fakta).

- **Positif:** kabar yang menguntungkan atau perbaikan pada aspek tersebut.
- **Negatif:** kabar yang merugikan atau perburukan.
- **Netral:** laporan fakta, pernyataan tanpa dampak jelas, atau sinyal campuran yang seimbang.

Kasus abu-abu harus dibahas dan dicatat bersama, misalnya "inflasi turun": apakah positif (kondisi membaik) atau netral (hanya laporan angka)? Putuskan sekali, tulis, dan konsisten. Catat juga aturan untuk kalimat dengan dua sentimen berlawanan dan untuk kutipan opini pihak ketiga.

### 7.4 Larangan penting

- **Jangan** memakai LLM atau model sentimen lain untuk memberi label lalu memperlakukannya sebagai kebenaran di test set. Boleh membantu draf awal, tetapi tiap label wajib diperiksa manusia.
- **Jangan** menyalin sentimen satu berita ke semua aspeknya tanpa dilabel. Itu bukan ABSA.

### 7.5 Pembagian data

Pisahkan **train / validation / test per artikel**, bukan per kalimat (usulan: ~70 / 15 / 15). Kalau kalimat dari berita yang sama muncul di dua split, skor evaluasi menggelembung.

---

## 8. Evaluasi

### Bagian A: bukti bahwa "per aspek" memang diperlukan

1. **Proporsi berita bersentimen campuran** (aspek-aspek dalam satu berita berbeda polaritasnya).
2. **Distribusi sentimen berbeda antar aspek** (tabel atau grafik, uji chi-square).
3. **Baseline "sentimen berita dipakai untuk semua aspek":** ukur kecocokannya dengan label per aspek yang benar. Skor rendah berarti label per berita tidak cukup.
4. **Kesepakatan anotator** (Kappa) pada irisan yang dilabel berdua.

### Bagian B: bukti untuk hipotesis "IndoBERT (monolingual) lebih baik dari XLM-RoBERTa (multilingual)"

Kedua model dilatih dan diuji dengan data, split, dan upaya tuning yang setara, pada test set berlabel manusia. Uji signifikansi wajib dilaporkan, karena selisih antara model monolingual dan multilingual pada tugas berbahasa Indonesia sering tipis; tanpa uji signifikansi kita tidak bisa membedakan "unggul nyata" dari "noise".

- **Macro-F1** (metrik utama) dan akurasi.
- **F1 per kelas** (positif, netral, negatif). Kelas netral biasanya paling sulit.
- **F1 per aspek**, dengan catatan untuk aspek bersampel sedikit.
- **Confusion matrix** tiap model.
- **Uji signifikansi:** beberapa seed pelatihan (rata-rata ± simpangan) dan uji McNemar atau bootstrap.
- **Analisis kesalahan kualitatif:** contoh kalimat yang salah di satu model tetapi benar di model lain.
- **Ablation (opsional):** input dengan aspek vs tanpa aspek, untuk membuktikan informasi aspek membantu.

### Bagian C: analisis dengan model terbaik

Model terbaik diterapkan ke data tak berlabel untuk menunjukkan distribusi sentimen per aspek dan tren per waktu. Ini **gambaran tren, bukan bukti kualitas model**, karena tidak ada label manusia di sana, dan hasilnya mewarisi noise dari BERTopic dan prediksi model.

### Urutan penyajian hasil di naskah

1. Kualitas topic modeling dan penetapan aspek (jumlah aspek, outlier, akurasi validasi).
2. Statistik dataset berlabel dan bukti Bagian A.
3. Perbandingan model: keseluruhan, per kelas, per aspek.
4. Analisis sentimen per aspek (Bagian C).
5. Keterbatasan.

---

## 9. Keterbatasan yang perlu ditulis jujur

- Hanya dua anotator; kesepakatan diukur hanya pada irisan kecil.
- Kualitas penetapan aspek bergantung pada BERTopic, termasuk outlier.
- Hasil berlaku untuk dataset dan konfigurasi ini, bukan klaim umum bahwa satu model selalu lebih baik.
- Batas panjang input model (misalnya 512 token) membuat unit teks harus diputuskan dan dilaporkan.
- **Perbandingan tidak mengisolasi satu variabel.** IndoBERT dan XLM-RoBERTa berbeda pada cakupan bahasa (monolingual vs multilingual), resep pretraining (BERT vs RoBERTa), dan korpus pretraining sekaligus. Jadi selisih skor mencerminkan efek gabungan ketiganya, bukan pengaruh monolingual vs multilingual secara murni. Kesimpulan dibatasi pada "model utuh", bukan pada satu faktor tunggal.

---

## 10. Ringkasan tugas anotator

- [ ] Memahami definisi aspek dan pedoman sentimen, lalu menyepakati kasus abu-abu.
- [ ] Menandai "aspek salah" bila penetapan BERTopic jelas keliru.
- [ ] Melabel sentimen (positif / netral / negatif) per pasangan (kalimat, aspek) sesuai porsi masing-masing.
- [ ] Mengerjakan irisan bersama untuk perhitungan Kappa.
- [ ] Mencatat setiap kasus sulit beserta keputusannya.
- [ ] Tidak memakai LLM atau model lain sebagai sumber label akhir.
