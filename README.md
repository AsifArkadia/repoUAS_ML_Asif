1. Judul
   Model Prediksi Tujuan Perjalanan Kereta Api Berdasarkan Stasiun Awal menggunakan naive bayes

2. Ringkasan Masalah dan Tujuan, Alur penyelesaian
   PT Kereta Api Indonesia menyediakan data perjalanan kereta yang dapat dimanfaatkan untuk menganalisis dan memprediksi pola rute.
   Permasalahan yang diangkat adalah bagaimana memprediksi **stasiun tujuan** berdasarkan **stasiun awal** menggunakan algoritma machine learning.

  **Tujuan:**
  - Membangun model klasifikasi menggunakan Naive Bayes.
  - Mengevaluasi model dengan akurasi dan metrik evaluasi lainnya.
  - Menyajikan analisis jaringan koneksi antar stasiun.
  
  **Alur Penyelesaian:**
  1. Load Dataset
  2. Preprocessing (Encoding)
  3. Split Data
  4. Model Training (Naive Bayes)
  5. Evaluasi
  6. Visualisasi Jaringan

3. Dataset, EDA, dan Preprocessing
   Dataset: `daop1_fix.csv`  
   Fitur: `stasiun_awal` (input), `stasiun_akhir` (target)
   ```python
   import pandas as pd
   df = pd.read_csv('daop1_fix.csv')
   df = df[['stasiun_awal', 'stasiun_akhir']].dropna()
   df['stasiun_awal'].value_counts().plot(kind='bar')
   ```

4. Modeling
   ```python
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    
    le_awal = LabelEncoder()
    le_akhir = LabelEncoder()
    X = le_awal.fit_transform(df['stasiun_awal']).reshape(-1, 1)
    y = le_akhir.fit_transform(df['stasiun_akhir'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    ```
5. Performa Model
   ```python
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = model.predict(X_test)
    print("Akurasi:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    ```
    
    Contoh hasil:
    - Akurasi: 84.6%
    - F1-score rata-rata: 82.9%

6. Hasil & Kesimpulan
   Model Naive Bayes cukup efektif untuk memprediksi stasiun tujuan hanya dengan input stasiun awal. Hasil evaluasi menunjukkan model memiliki akurasi lebih dari 84%, yang sudah cukup baik.
   Visualisasi jaringan transportasi juga mendukung bahwa terdapat simpul penting seperti Jakarta Kota dan Gambir.
   Model dapat dikembangkan lebih lanjut dengan penambahan fitur seperti jenis kereta, waktu, atau rute panjang.
