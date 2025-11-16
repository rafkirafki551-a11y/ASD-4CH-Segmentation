# Segmentasi Atrial Septal Defect (ASD) View 4CH menggunakan YOLOv8

Repositori ini berisi hasil dan evaluasi dari model *instance segmentation* **YOLOv8** yang dilatih untuk mendeteksi dan mensegmentasi struktur jantung pada citra ekokardiogram. Fokus utama proyek ini adalah identifikasi defek septum jantung (kemungkinan Atrial Septal Defect/ASD).

**Streamlit app:** https://asd-4ch-segmentation-vzeh7gzbmbnga3m7nndmtx.streamlit.app/

Model ini dilatih untuk mengklasifikasikan 3 kelas:

* **IVS** (Interventricular Septum): Dinding pemisah antara ventrikel (bilik) jantung.
* **IAS** (Interatrial Septum): Dinding pemisah antara atrium (serambi) jantung.
* **H** (Hole): Merujuk pada defek atau lubang pada septum.

---

## 📈 Analisis Dataset

Analisis dataset menunjukkan distribusi kelas dan karakteristik anotasi yang digunakan untuk pelatihan.

* **Distribusi Kelas:** Dataset memiliki ketidakseimbangan (*imbalance*), di mana jumlah *instance* untuk kelas `H` (Hole) secara signifikan lebih sedikit dibandingkan dua kelas lainnya.
    * `IVS`: 805 instances
    * `IAS`: 806 instances
    * `H`: 406 instances
* **Karakteristik Objek:** Plot di bagian bawah menunjukkan bahwa objek (septa dan defek) umumnya memiliki bentuk yang tinggi dan sempit (rata-rata lebar < 0.05, tinggi > 0.05) dan terdistribusi di bagian tengah gambar (x sekitar 0.45-0.55).

---

## 📊 Hasil Evaluasi Model (Segmentasi Mask)

Evaluasi ini berfokus pada performa *instance segmentation* (mask).

### 1. Precision-Recall (PR) Curve & mAP

*Mean Average Precision* (mAP) adalah metrik utama untuk evaluasi deteksi dan segmentasi objek.

* **mAP@0.5 (all classes): 0.803**
* **Average Precision (AP) per kelas:**
    * `IVS`: 0.953 (Sangat Baik)
    * `IAS`: 0.849 (Baik)
    * `H`: 0.606 (Cukup)

Performa yang lebih rendah pada kelas `H` (Hole) kemungkinan besar disebabkan oleh jumlah data yang lebih sedikit dan variasi bentuk yang lebih tinggi.

### 2. F1-Score vs. Confidence

Kurva F1-Score membantu menentukan *confidence threshold* optimal untuk menyeimbangkan Precision dan Recall.

* Skor F1-Score puncak untuk **semua kelas** adalah **0.80** yang dicapai pada *confidence threshold* **0.393**.
* Kelas `IVS` dan `IAS` menunjukkan F1-Score puncak yang tinggi (di atas 0.8), sementara kelas `H` memiliki F1-Score puncak yang lebih rendah (sekitar 0.65).

### 3. Confusion Matrix (Normalized)

*Confusion matrix* menunjukkan detail performa klasifikasi model, dinormalisasi berdasarkan *true labels* (Recall).

* **Performa Kuat (True Positives):**
    * 98% dari `IVS` asli berhasil diprediksi dengan benar.
    * 85% dari `IAS` asli berhasil diprediksi dengan benar.
    * 81% dari `H` (Hole) asli berhasil diprediksi dengan benar.
* **Tantangan Utama (False Negatives):**
    * Masalah utama model adalah kecenderungan untuk melewatkan deteksi (mengklasifikasikannya sebagai `background`).
    * 15% dari `IAS` asli terlewat (diprediksi sebagai `background`).
    * 19% dari `H` (Hole) asli terlewat (diprediksi sebagai `background`).

### 4. Average IoU (Intersection over Union)

IoU mengukur seberapa baik *mask* yang diprediksi tumpang tindih dengan *ground truth*.

* `h`: Average IoU = 0.3025
* `ias`: Average IoU = 0.5870
* `ivs`: Average IoU = 0.5868
* **Overall Average IoU = 0.5483**

Hasil IoU ini konsisten dengan metrik mAP, di mana kelas `h` (Hole) memiliki skor tumpang tindih yang paling rendah, menunjukkan tantangan terbesar dalam segmentasi yang presisi untuk kelas ini.

---

## 📦 Hasil Evaluasi (Bounding Box)

Sebagai perbandingan, berikut adalah hasil untuk deteksi *bounding box* saja.

* **mAP@0.5 (all classes): 0.858**
* **AP per kelas:**
    * `IVS`: 0.961
    * `IAS`: 0.853
    * `H`: 0.759
* **F1-Score Puncak:** **0.85** pada *confidence threshold* **0.363**.

Secara umum, model lebih mudah untuk *mendeteksi* (menemukan lokasi dengan *bounding box*) daripada *mensegmentasi* (menggambar garis luar yang presisi) untuk ketiga kelas.
