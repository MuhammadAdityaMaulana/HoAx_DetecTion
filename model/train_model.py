"""
============================================================
HoaxRadar — train_model.py
Training Model FastText
Dataset: Berita Hoax Kabupaten Sumedang 2022-2025
============================================================

Cara menjalankan:
    python train_model.py

Output:
    model_fasttext_hoax_sumedang.bin  ← model siap pakai
    hasil_evaluasi.txt                ← laporan akurasi
============================================================
"""

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ── Cek library fasttext (support fasttext-wheel di Windows) ──
try:
    import fasttext
    print("✅ fasttext berhasil diimport")
except ImportError:
    print("❌ fasttext tidak ditemukan.")
    print("   Jalankan salah satu:")
    print("   pip install fasttext-wheel   (Windows)")
    print("   pip install fasttext         (Linux/Mac)")
    exit(1)

# ── Cek library Sastrawi ──────────────────────────────────────
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    SASTRAWI_AVAILABLE = True
    print("✅ PySastrawi berhasil diimport")
except ImportError:
    SASTRAWI_AVAILABLE = False
    print("⚠  PySastrawi tidak ditemukan — preprocessing tanpa stemming/stopword")
    print("   Install dengan: pip install PySastrawi")

print()

# ════════════════════════════════════════════════════════════
# KONFIGURASI BARU (Disesuaikan untuk Dataset Kecil agar Akurasi Naik)
# ════════════════════════════════════════════════════════════
DATASET_PATH = 'berita smd 2022-2025.csv'
MODEL_OUTPUT = 'model_fasttext_hoax_sumedang.bin'
TRAIN_TXT    = 'fasttext_train.txt'
TEST_TXT     = 'fasttext_test.txt'
EVAL_OUTPUT  = 'hasil_evaluasi.txt'
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# Hyperparameter FastText yang Dioptimasi:
FT_EPOCH      = 100     # Naikkan epoch agar model lebih belajar dari data yang sedikit
FT_LR         = 0.5     # Naikkan learning rate agar bobot cepat menyesuaikan
FT_WORDNGRAMS = 1       # Gunakan unigram dulu karena data sedikit (mencegah sparsity)
FT_DIM        = 50      # Turunkan dimensi vektor ke 50 agar model tidak terlalu "bengkak"
FT_MINN       = 0       # WAJIB 0: Matikan subword tingkat karakter
FT_MAXN       = 0       # WAJIB 0: Matikan subword tingkat karakter
FT_LOSS       = 'softmax'


# ════════════════════════════════════════════════════════════
# PREPROCESSING
# ════════════════════════════════════════════════════════════
print("=" * 55)
print("  STEP 1 — Inisialisasi Preprocessing")
print("=" * 55)

stemmer          = None
stopword_remover = None

if SASTRAWI_AVAILABLE:
    print("⏳ Memuat stemmer Sastrawi...")
    stemmer = StemmerFactory().create_stemmer()

    print("⏳ Memuat stopword remover Sastrawi...")
    stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

    print("✅ Preprocessing Sastrawi siap!\n")
else:
    print("⚠  Sastrawi tidak tersedia, lewati stemming & stopword removal\n")


def clean_text(text: str) -> str:
    """Membersihkan teks mentah."""
    if not isinstance(text, str) or not text.strip():
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)   # hapus URL
    text = re.sub(r'@\w+|#\w+', '', text)                  # hapus mention & hashtag
    text = re.sub(r'\S+@\S+', '', text)                    # hapus email
    text = re.sub(r'\d+', '', text)                        # hapus angka
    text = re.sub(r'[^a-z\s]', ' ', text)                 # hapus non-huruf
    text = re.sub(r'\s+', ' ', text).strip()               # normalisasi spasi
    return text


def preprocess(text: str) -> str:
    """Pipeline preprocessing lengkap: clean → stopword → stem."""
    text = clean_text(text)
    if not text:                           # FIX #2: hindari proses teks kosong
        return ''
    if stopword_remover:
        text = stopword_remover.remove(text)
    if stemmer:
        text = stemmer.stem(text)
    return text.strip()


# ════════════════════════════════════════════════════════════
# LOAD DATASET
# ════════════════════════════════════════════════════════════
print("=" * 55)
print("  STEP 2 — Load Dataset")
print("=" * 55)

if not os.path.exists(DATASET_PATH):
    print(f"❌ File dataset tidak ditemukan: '{DATASET_PATH}'")
    print(f"   Pastikan file CSV ada di folder yang sama dengan train_model.py")
    exit(1)

# FIX #3: Tambahkan encoding fallback agar CSV dengan karakter
# non-ASCII (latin-1 / cp1252) tidak langsung error
try:
    df = pd.read_csv(DATASET_PATH, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(DATASET_PATH, encoding='latin-1')

print(f"✅ Dataset dimuat: {len(df)} baris, {len(df.columns)} kolom")
print(f"   Kolom: {df.columns.tolist()}")
print()

# Nama kolom yang diharapkan
TEXT_COL  = 'Narasi'
LABEL_COL = 'hoax'

# FIX #4: Validasi keberadaan kolom sebelum dipakai agar error lebih jelas
for col in [TEXT_COL, LABEL_COL]:
    if col not in df.columns:
        print(f"❌ Kolom '{col}' tidak ditemukan di dataset.")
        print(f"   Kolom yang tersedia: {df.columns.tolist()}")
        exit(1)

# Hapus baris dengan teks atau label kosong
df = df[[TEXT_COL, LABEL_COL]].dropna()
df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
df = df[df[TEXT_COL] != '']

# FIX #5: Pastikan kolom label bertipe integer (bukan string '0'/'1')
df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors='coerce')
df = df.dropna(subset=[LABEL_COL])
df[LABEL_COL] = df[LABEL_COL].astype(int)

df = df.reset_index(drop=True)

print(f"📊 Distribusi Label:")
counts = df[LABEL_COL].value_counts()
for label, count in counts.items():
    nama = 'HOAX' if label == 1 else 'NON-HOAX'
    pct  = count / len(df) * 100
    bar  = '█' * int(pct / 3)
    print(f"   {nama:10} ({label}): {count:3} data  {bar} {pct:.1f}%")
print()

# FIX #6: Pastikan dataset punya minimal 2 kelas sebelum training
if len(counts) < 2:
    print("❌ Dataset hanya memiliki 1 kelas label.")
    print("   Pastikan ada data HOAX (1) dan NON-HOAX (0).")
    exit(1)


# ════════════════════════════════════════════════════════════
# PREPROCESSING SEMUA DATA
# ════════════════════════════════════════════════════════════
print("=" * 55)
print("  STEP 3 — Preprocessing Teks")
print("=" * 55)
print("⏳ Memproses teks... (mungkin butuh beberapa menit)")

df['processed'] = df[TEXT_COL].apply(preprocess)

# Hapus baris yang kosong setelah preprocessing
df = df[df['processed'].str.strip() != ''].reset_index(drop=True)

print(f"✅ Preprocessing selesai: {len(df)} teks valid")
print()
print("📝 Contoh hasil preprocessing:")
for i in range(min(3, len(df))):
    label = 'HOAX' if df[LABEL_COL].iloc[i] == 1 else 'NON-HOAX'
    print(f"   [{label}] {df['processed'].iloc[i][:80]}...")
print()


# ════════════════════════════════════════════════════════════
# SPLIT DATA TRAIN & TEST
# ════════════════════════════════════════════════════════════
print("=" * 55)
print("  STEP 4 — Split Data Train & Test")
print("=" * 55)

X = df['processed'].tolist()
y = df[LABEL_COL].tolist()

# FIX #7: stratify bisa gagal jika salah satu kelas < 2 sample.
# Tambahkan pengecekan minimum sample per kelas.
min_class_count = counts.min()
use_stratify = min_class_count >= 2

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y if use_stratify else None
)

if not use_stratify:
    print("⚠  Stratify dinonaktifkan: salah satu kelas terlalu sedikit.")

print(f"✅ Data training : {len(X_train)} teks")
print(f"   Data testing  : {len(X_test)} teks")
print()


# ════════════════════════════════════════════════════════════
# FORMAT FASTTEXT (label + teks)
# ════════════════════════════════════════════════════════════
def to_fasttext_format(texts, labels):
    """Ubah ke format FastText: __label__<kelas> <teks>"""
    lines = []
    for text, label in zip(texts, labels):
        # FIX #8: Bersihkan newline dalam teks agar tidak merusak format FastText
        # (FastText membaca 1 baris = 1 contoh)
        text_clean = text.replace('\n', ' ').replace('\r', ' ').strip()
        if not text_clean:
            continue
        ft_label = '__label__hoax' if int(label) == 1 else '__label__non_hoax'
        lines.append(f"{ft_label} {text_clean}")
    return lines


train_lines = to_fasttext_format(X_train, y_train)
test_lines  = to_fasttext_format(X_test,  y_test)

with open(TRAIN_TXT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_lines))

with open(TEST_TXT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(test_lines))

print(f"✅ File training FastText ditulis : {TRAIN_TXT}  ({len(train_lines)} baris)")
print(f"   File testing FastText ditulis  : {TEST_TXT}  ({len(test_lines)} baris)")
print()


# ════════════════════════════════════════════════════════════
# TRAINING MODEL FASTTEXT
# ════════════════════════════════════════════════════════════
print("=" * 55)
print("  STEP 5 — Training Model FastText")
print("=" * 55)
print(f"   Epoch        : {FT_EPOCH}")
print(f"   Learning Rate: {FT_LR}")
print(f"   Word N-gram  : {FT_WORDNGRAMS}")
print(f"   Dimensi      : {FT_DIM}")
print(f"   Loss         : {FT_LOSS}")
print()
print("⏳ Training sedang berjalan...")

# FIX #9: Bungkus training dalam try-except agar error FastText
# (misal: file kosong, format salah) langsung terdeteksi
try:
    model = fasttext.train_supervised(
        input      = TRAIN_TXT,
        epoch      = FT_EPOCH,
        lr         = FT_LR,
        wordNgrams = FT_WORDNGRAMS,
        dim        = FT_DIM,
        minn       = FT_MINN,
        maxn       = FT_MAXN,
        loss       = FT_LOSS,
        verbose    = 2
    )
except Exception as e:
    print(f"❌ Training gagal: {e}")
    exit(1)

print()
print("✅ Training selesai!")
print()


# ════════════════════════════════════════════════════════════
# SIMPAN MODEL
# ════════════════════════════════════════════════════════════
print("=" * 55)
print("  STEP 6 — Simpan Model")
print("=" * 55)

model.save_model(MODEL_OUTPUT)
size_mb = os.path.getsize(MODEL_OUTPUT) / (1024 * 1024)
print(f"✅ Model disimpan: {MODEL_OUTPUT} ({size_mb:.2f} MB)")
print()


# ════════════════════════════════════════════════════════════
# EVALUASI MODEL
# ════════════════════════════════════════════════════════════
print("=" * 55)
print("  STEP 7 — Evaluasi Model")
print("=" * 55)

# FIX #10: Prediksi per teks — gunakan list y_test yang sudah
# pasti bertipe list[int] (bukan numpy array) sejak awal.
# Kode asli memanggil model.predict() dua kali per teks (boros);
# perbaiki jadi satu kali.
y_pred = []
for text in X_test:
    labels, _ = model.predict(text, k=1)
    pred_label = 1 if labels[0] == '__label__hoax' else 0
    y_pred.append(pred_label)

y_test_list = [int(v) for v in y_test]  # pastikan int murni

# Hitung metrik
acc  = accuracy_score(y_test_list, y_pred) * 100
prec = precision_score(y_test_list, y_pred, zero_division=0) * 100
rec  = recall_score(y_test_list, y_pred, zero_division=0) * 100
f1   = f1_score(y_test_list, y_pred, zero_division=0) * 100
cm   = confusion_matrix(y_test_list, y_pred)
cr   = classification_report(
    y_test_list, y_pred,
    target_names=['NON-HOAX', 'HOAX'],
    zero_division=0
)

# FIX #11: confusion_matrix bisa berukuran 1×1 jika hanya ada 1 kelas
# di data test. Tangani supaya tidak IndexError saat akses cm[1][0] dll.
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
else:
    tn = fp = fn = tp = 0
    print("⚠  Confusion matrix tidak 2×2 — mungkin data test terlalu sedikit.")

print(f"  Accuracy  : {acc:.2f}%")
print(f"  Precision : {prec:.2f}%")
print(f"  Recall    : {rec:.2f}%")
print(f"  F1-Score  : {f1:.2f}%")
print()
print("  Confusion Matrix:")
print(f"  {'':20} Pred NON-HOAX  Pred HOAX")
print(f"  {'Actual NON-HOAX':20} {tn:10}        {fp:6}")
print(f"  {'Actual HOAX':20} {fn:10}        {tp:6}")
print()
print("  Classification Report:")
for line in cr.split('\n'):
    print(f"  {line}")


# ════════════════════════════════════════════════════════════
# SIMPAN LAPORAN EVALUASI
# ════════════════════════════════════════════════════════════
report_text = f"""
============================================================
  HoaxRadar — Laporan Evaluasi Model FastText
  Dataset: Berita Hoax Kabupaten Sumedang 2022-2025
============================================================

KONFIGURASI TRAINING
  Dataset        : {DATASET_PATH}
  Total Data     : {len(df)} berita
  Data Training  : {len(X_train)} berita ({int((1-TEST_SIZE)*100)}%)
  Data Testing   : {len(X_test)} berita ({int(TEST_SIZE*100)}%)
  Epoch          : {FT_EPOCH}
  Learning Rate  : {FT_LR}
  Word N-gram    : {FT_WORDNGRAMS}
  Dimensi Vektor : {FT_DIM}
  Loss Function  : {FT_LOSS}
  Sastrawi       : {'Ya' if SASTRAWI_AVAILABLE else 'Tidak'}

DISTRIBUSI LABEL
  NON-HOAX (0)   : {counts.get(0, 0)} data
  HOAX (1)       : {counts.get(1, 0)} data

HASIL EVALUASI
  Accuracy       : {acc:.2f}%
  Precision      : {prec:.2f}%
  Recall         : {rec:.2f}%
  F1-Score       : {f1:.2f}%

CONFUSION MATRIX
  {'':20} Pred NON-HOAX  Pred HOAX
  {'Actual NON-HOAX':20} {tn:10}        {fp:6}
  {'Actual HOAX':20} {fn:10}        {tp:6}

CLASSIFICATION REPORT
{cr}

OUTPUT FILE
  Model    : {MODEL_OUTPUT}
  Evaluasi : {EVAL_OUTPUT}
============================================================
"""

with open(EVAL_OUTPUT, 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"✅ Laporan evaluasi disimpan: {EVAL_OUTPUT}")
print()


# ════════════════════════════════════════════════════════════
# UJI COBA PREDIKSI
# ════════════════════════════════════════════════════════════
print("=" * 55)
print("  STEP 8 — Uji Coba Prediksi Manual")
print("=" * 55)

contoh = [
    "Beredar akun WhatsApp mengatasnamakan Bupati Sumedang yang meminta transfer uang untuk donasi",
    "Pemerintah Kabupaten Sumedang meresmikan jembatan baru di wilayah Cimanggung untuk memperlancar akses warga"
]

for teks in contoh:
    processed = preprocess(teks)

    # FIX #12: Jika teks kosong setelah preprocessing, lewati prediksi
    if not processed:
        print(f"  Teks    : {teks[:65]}...")
        print(f"  Hasil   : ⚠ Teks kosong setelah preprocessing, dilewati")
        print()
        continue

    # FIX #13: Panggil model.predict() SEKALI saja (kode asli memanggil 2×)
    labels, probs = model.predict(processed, k=1)
    label_bersih  = 'HOAX' if labels[0] == '__label__hoax' else 'NON-HOAX'
    print(f"  Teks    : {teks[:65]}...")
    print(f"  Hasil   : {label_bersih}  ({probs[0]*100:.1f}% confidence)")
    print()


# ════════════════════════════════════════════════════════════
# SELESAI
# ════════════════════════════════════════════════════════════
print("=" * 55)
print("  SELESAI!")
print("=" * 55)
print()
print(f"  File yang dihasilkan:")
print(f"  📦 {MODEL_OUTPUT}  ← model siap dipakai di app.py")
print(f"  📄 {EVAL_OUTPUT}   ← laporan evaluasi lengkap")
print()
print("  Langkah selanjutnya:")
print("  1. Pastikan model_fasttext_hoax_sumedang.bin")
print("     ada di folder yang sama dengan app.py")
print("  2. Jalankan: python app.py")
print("  3. Buka browser: http://localhost:5000")
print("=" * 55)

# Bersihkan file sementara
for tmp in [TRAIN_TXT, TEST_TXT]:
    if os.path.exists(tmp):
        os.remove(tmp)