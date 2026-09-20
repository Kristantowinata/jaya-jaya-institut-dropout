# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

- Nama: Kristanto Winata
- Email: kristanto.winata@binus.ac.id
- Id Dicoding: kristanto_winata95bh

## Business Understanding

Jaya Jaya Institut merupakan sebuah perguruan tinggi yang telah berdiri sejak tahun 2000 dan telah mencetak banyak lulusan dengan reputasi yang sangat baik. Namun demikian, institusi ini menghadapi masalah serius: cukup banyak siswa yang tidak menyelesaikan pendidikannya alias *dropout*. Dari total 4.424 siswa pada data historis, **32,1% di antaranya berstatus Dropout**, hampir sebanding dengan siswa yang berhasil lulus (49,9% Graduate) dan sisanya masih terdaftar (17,9% Enrolled).

Tingginya angka dropout ini merugikan institusi dalam berbagai aspek: menurunkan reputasi dan tingkat kelulusan, mengurangi pendapatan dari uang kuliah yang tidak terselesaikan, serta menyia-nyiakan sumber daya yang sudah diinvestasikan pada siswa yang akhirnya tidak lulus. Selama ini, pihak akademik baru menyadari seorang siswa berisiko dropout setelah nilai atau kehadirannya sudah sangat buruk — pada titik itu, intervensi seringkali sudah terlambat.

Jaya Jaya Institut membutuhkan solusi berbasis data untuk mendeteksi *sedini mungkin* siswa yang berisiko dropout, sehingga bimbingan khusus dapat diberikan sebelum siswa tersebut benar-benar berhenti kuliah, bukan setelahnya.

### Permasalahan Bisnis

1. Jaya Jaya Institut belum memiliki cara sistematis untuk mendeteksi siswa yang berisiko dropout sejak awal masa studinya, sehingga bimbingan khusus baru diberikan setelah performa siswa sudah sangat menurun.
2. Institusi belum mengetahui secara pasti faktor-faktor akademik maupun administratif apa saja yang paling berkaitan dengan keputusan siswa untuk dropout, sehingga program pencegahan yang ada masih bersifat umum dan belum tepat sasaran.
3. Tanpa alat monitoring yang memadai, tim akademik kesulitan memantau performa siswa secara agregat maupun individual untuk mengambil tindakan pencegahan secara proaktif.

### Cakupan Proyek

1. Melakukan exploratory data analysis (EDA) terhadap data akademik dan administratif siswa untuk mengidentifikasi faktor-faktor yang berkaitan dengan dropout.
2. Membangun model machine learning untuk memprediksi status siswa (Dropout, Enrolled, atau Graduate) berdasarkan data akademik dan administratifnya.
3. Membangun business dashboard untuk memantau performa siswa dan indikator dropout secara berkelanjutan.
4. Mengembangkan prototype sistem machine learning berbasis Streamlit yang siap digunakan tim akademik untuk memprediksi status siswa secara mandiri.
5. Menyusun rekomendasi action items berbasis data bagi Jaya Jaya Institut.

### Persiapan

Sumber data: [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) — UCI Machine Learning Repository (Realinho et al., 2021), 4.424 baris data siswa. Data ini dimuat otomatis langsung dari URL di dalam `notebook.ipynb`, sehingga tidak perlu diunduh manual untuk menjalankan notebook.

#### Setup Environment - Anaconda

```
conda create --name main-ds python=3.11
conda activate main-ds
pip install -r requirements.txt
```

#### Setup Environment - Shell/Terminal (Pipenv)

```
pip install pipenv
pipenv install
pipenv shell
pip install -r requirements.txt
```

#### Menjalankan Notebook

```
jupyter notebook notebook.ipynb
```

Jalankan seluruh cell secara berurutan (Run All / Kernel > Restart & Run All). Menjalankan notebook ini juga akan menghasilkan `students_data_clean.csv` (untuk dashboard) serta `model/model.joblib` dan `model/selected_features.joblib` (untuk prototype Streamlit).

## Business Dashboard

Dashboard dibuat menggunakan **Looker Studio**, berisi visualisasi untuk memantau performa dan risiko dropout siswa, di antaranya:

- Distribusi status siswa (Dropout / Enrolled / Graduate).
- Status siswa berdasarkan jumlah mata kuliah yang berhasil diselesaikan tiap semester.
- Status siswa berdasarkan status pembayaran uang kuliah (tepat waktu / menunggak).
- Status siswa berdasarkan kepemilikan beasiswa.
- Status siswa berdasarkan usia saat mendaftar.

Dashboard bersumber dari `students_data_clean.csv` (hasil Data Preparation pada notebook).

- Link dashboard: https://datastudio.google.com/reporting/7a2b4929-e8d4-4566-aeae-da31af0acdb9

## Menjalankan Sistem Machine Learning

Prototype sistem machine learning dibuat menggunakan **Streamlit** (`app.py`). Prototype ini memprediksi status siswa (Dropout/Enrolled/Graduate) berdasarkan 12 fitur akademik dan administratif yang paling berpengaruh, menggunakan model `RandomForestClassifier` yang telah dilatih dan disimpan di folder `model/`.

### Menjalankan secara lokal

```
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi akan otomatis terbuka di browser pada `http://localhost:8501`. Isi form dengan data akademik siswa, lalu klik "Prediksi Status Siswa" untuk melihat hasil prediksi beserta probabilitas tiap kelas.

### Mengakses via Streamlit Community Cloud

- Link prototype: `https://jaya-jaya-institut-dropout-brwmna5zwjwx4q6jrsdlos.streamlit.app/`  

## Conclusion

**Ringkasan performa model:** Random Forest yang dilatih dengan seluruh 35 fitur mencapai **accuracy 76,5%** dalam memprediksi status siswa (Dropout/Enrolled/Graduate) pada data uji, sedangkan model deployment yang hanya menggunakan 12 fitur terkurasi (demi kemudahan penggunaan pada prototype) mencapai **accuracy 74,8%** — penurunan yang kecil dan sepadan dengan kemudahan penggunaannya. Model paling akurat dalam mengenali siswa Graduate (recall 91-93%) dan Dropout (recall 74-75%), namun masih kesulitan membedakan siswa Enrolled dari dua kelas lainnya (recall hanya 33-34%) — hal ini wajar karena status Enrolled berada "di tengah" dan siswa yang masih terdaftar bisa saja nantinya lulus atau dropout.

Fitur yang paling berpengaruh terhadap prediksi (feature importance) adalah jumlah mata kuliah yang lulus di semester 2 dan 1, rata-rata nilai semester 2 dan 1, nilai penerimaan (admission grade), status pembayaran uang kuliah, dan usia saat mendaftar.

**Insight utama dari EDA dan mengapa itu terjadi:**
- **Performa akademik** adalah prediktor terkuat: siswa Dropout rata-rata hanya lulus **1,94 mata kuliah** di semester 2, jauh di bawah siswa Enrolled (4,06) dan Graduate (6,18). Ini masuk akal karena kegagalan akademik yang terus-menerus membuat siswa kehilangan motivasi atau tidak memenuhi syarat melanjutkan studi.
- **Status pembayaran uang kuliah** sangat berkaitan dengan dropout: **32,2%** siswa Dropout belum melunasi uang kuliah tepat waktu, dibanding hanya **1,3%** pada siswa Graduate. Kesulitan finansial tampaknya menjadi salah satu pendorong utama dropout.
- **Status debtor (tunggakan)** menunjukkan pola serupa: 22,0% siswa Dropout berstatus debtor, dibanding hanya 4,6% pada siswa Graduate.
- **Beasiswa berkaitan kuat dengan kelulusan**: hanya 9,4% siswa Dropout yang menerima beasiswa, dibanding 37,8% pada siswa Graduate — dukungan finansial tampaknya membantu siswa bertahan hingga lulus.
- **Usia saat mendaftar**: siswa Dropout rata-rata mendaftar di usia lebih tua (26,1 tahun) dibanding siswa Graduate (21,8 tahun), kemungkinan karena siswa yang lebih tua lebih sering harus membagi waktu dengan pekerjaan atau tanggung jawab keluarga.

### Rekomendasi Action Items

1. **Bangun sistem peringatan dini berbasis akademik**: karena jumlah mata kuliah lulus per semester adalah prediktor terkuat, pantau siswa yang lulus kurang dari 2-3 mata kuliah pada semester pertama dan segera tawarkan bimbingan akademik atau kelas remedial sebelum semester berikutnya dimulai.
2. **Prioritaskan intervensi finansial**: siswa yang menunggak uang kuliah (belum lunas / berstatus debtor) punya risiko dropout jauh lebih tinggi (32% vs 1,3% pada aspek tunggakan uang kuliah). Sediakan skema cicilan fleksibel atau dana talangan darurat bagi siswa dengan catatan pembayaran bermasalah pada bulan pertama keterlambatan, sebelum tunggakan menumpuk.
3. **Perluas cakupan beasiswa atau bantuan finansial**, khususnya untuk siswa dengan performa akademik baik namun berisiko finansial, karena data menunjukkan penerima beasiswa jauh lebih banyak yang lulus (37,8%) dibanding yang dropout (9,4%).
4. **Beri perhatian khusus pada siswa yang mendaftar di usia lebih tua** (di atas rata-rata 24-26 tahun), misalnya lewat kelas dengan jadwal lebih fleksibel atau layanan konseling manajemen waktu, karena kelompok ini secara konsisten menunjukkan tingkat dropout lebih tinggi.
5. **Gunakan prototype machine learning** (`app.py`) di awal semester untuk memprediksi status tiap siswa baru/aktif berdasarkan data yang sudah tersedia, sehingga siswa dengan probabilitas Dropout tertinggi bisa langsung masuk daftar prioritas bimbingan.
6. **Pantau seluruh indikator di atas secara berkelanjutan lewat business dashboard** agar tim akademik dapat mendeteksi perubahan pola dropout secara dini, bukan hanya setelah siswa berhenti kuliah.
