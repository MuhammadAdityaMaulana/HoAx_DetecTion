"""
============================================================
HoaxRadar — app.py
Flask API Server — Deteksi Berita Hoax Kabupaten Sumedang
Fitur: Prediksi via Teks + Prediksi via URL Berita
============================================================
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import re
import os
import logging

# ── Inisialisasi Flask ────────────────────────────────────────
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# ── Konfigurasi ───────────────────────────────────────────────
MODEL_PATH = os.getenv('MODEL_PATH', 'model/model_fasttext_hoax_sumedang.bin')

# ── Global model ──────────────────────────────────────────────
model            = None
stemmer          = None
stopword_remover = None


# ════════════════════════════════════════════════════════════
# LOAD MODEL
# ════════════════════════════════════════════════════════════
def load_model():
    global model, stemmer, stopword_remover
    try:
        import fasttext
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

        if not os.path.exists(MODEL_PATH):
            log.warning(f"⚠  File model tidak ditemukan: '{MODEL_PATH}'")
            return

        log.info(f"⏳ Memuat model dari: {MODEL_PATH}")
        model = fasttext.load_model(MODEL_PATH)

        log.info("⏳ Memuat Sastrawi...")
        stemmer          = StemmerFactory().create_stemmer()
        stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

        log.info("✅ Model siap!")

    except ImportError as e:
        log.error(f"❌ Library belum terinstall: {e}")
    except Exception as e:
        log.error(f"❌ Gagal memuat model: {e}")


# ════════════════════════════════════════════════════════════
# PREPROCESSING
# ════════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    if not text:
        return ''
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_text(text: str) -> str:
    text = clean_text(text)
    if stopword_remover:
        text = stopword_remover.remove(text)
    if stemmer:
        text = stemmer.stem(text)
    return text


def get_confidence_level(pct: float) -> str:
    if pct >= 85:
        return 'Sangat Tinggi'
    elif pct >= 70:
        return 'Tinggi'
    elif pct >= 55:
        return 'Sedang'
    else:
        return 'Rendah'


def build_result(raw_text: str, processed: str, source_url: str = None) -> dict:
    """Jalankan prediksi dan susun response JSON."""
    labels, probabilities = model.predict(processed, k=2)

    prob_dict = {}
    for label, prob in zip(labels, probabilities):
        key = label.replace('__label__', '')
        prob_dict[key] = round(float(prob), 4)

    main_label = labels[0].replace('__label__', '')
    main_prob  = round(float(probabilities[0]) * 100, 2)
    is_hoax    = main_label.lower() == 'hoax'

    result = {
        'status':           'success',
        'input_text':       raw_text,
        'processed_text':   processed,
        'prediction':       'HOAX' if is_hoax else 'NON-HOAX',
        'is_hoax':          is_hoax,
        'confidence':       main_prob,
        'confidence_level': get_confidence_level(main_prob),
        'probabilities':    prob_dict,
    }

    if source_url:
        result['source_url'] = source_url

    return result


# ════════════════════════════════════════════════════════════
# SCRAPING URL
# ════════════════════════════════════════════════════════════
def scrape_article(url: str) -> dict:
    """
    Ambil teks artikel dari URL berita.
    Mencoba newspaper3k dulu, fallback ke BeautifulSoup.
    Return: { 'title': ..., 'text': ..., 'method': ... }
    """
    import requests
    from bs4 import BeautifulSoup

    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        ),
        'Accept-Language': 'id-ID,id;q=0.9,en-US;q=0.8',
    }

    title  = ''
    text   = ''
    method = ''

    # ── Metode 1: newspaper3k ─────────────────────────────────
    try:
        import newspaper
        article = newspaper.Article(url, language='id')
        article.download()
        article.parse()
        title  = article.title or ''
        text   = article.text  or ''
        method = 'newspaper3k'
        log.info(f"Scraped via newspaper3k: {len(text)} chars")
    except Exception as e:
        log.warning(f"newspaper3k gagal: {e}, mencoba BeautifulSoup...")

    # ── Metode 2: BeautifulSoup fallback ─────────────────────
    if not text.strip():
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')

            # Ambil judul
            title_tag = soup.find('h1') or soup.find('title')
            if title_tag:
                title = title_tag.get_text(strip=True)

            # Hapus elemen tidak perlu
            for tag in soup(['script', 'style', 'nav', 'header',
                             'footer', 'aside', 'iframe', 'figure',
                             'figcaption', 'form', 'button', 'input']):
                tag.decompose()

            # Cari konten artikel dari tag umum berita Indonesia
            content_selectors = [
                'article', '.article-content', '.article-body',
                '.post-content', '.entry-content', '.content-body',
                '.detail-content', '.read-content', '.berita-content',
                '.news-content', '[itemprop="articleBody"]',
                '.detail__body', '.itp_bodycontent',
            ]

            article_text = ''
            for selector in content_selectors:
                tag = soup.select_one(selector)
                if tag:
                    paragraphs   = tag.find_all('p')
                    article_text = ' '.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
                    if len(article_text) > 100:
                        break

            # Fallback: ambil semua <p>
            if not article_text.strip():
                all_p = soup.find_all('p')
                article_text = ' '.join(
                    p.get_text(strip=True) for p in all_p
                    if len(p.get_text(strip=True)) > 40
                )

            text   = article_text
            method = 'beautifulsoup'
            log.info(f"Scraped via BeautifulSoup: {len(text)} chars")

        except requests.exceptions.Timeout:
            raise ValueError('Koneksi timeout. Coba lagi atau masukkan teks secara manual.')
        except requests.exceptions.ConnectionError:
            raise ValueError('Tidak dapat terhubung ke URL. Periksa koneksi internet Anda.')
        except requests.exceptions.HTTPError as e:
            raise ValueError(f'URL tidak dapat diakses: HTTP {e.response.status_code}')
        except Exception as e:
            raise ValueError(f'Gagal mengambil konten dari URL: {str(e)}')

    # Gabung judul + isi
    full_text = f"{title}. {text}".strip() if title else text.strip()

    if len(full_text) < 30:
        raise ValueError(
            'Konten artikel terlalu pendek atau tidak dapat diekstrak. '
            'Coba salin teks berita secara manual.'
        )

    return {
        'title':  title,
        'text':   full_text[:3000],   # batasi 3000 karakter
        'method': method,
        'chars':  len(full_text)
    }


# ════════════════════════════════════════════════════════════
# STATIC FILES
# ════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory('css', filename)

@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory('js', filename)


# ════════════════════════════════════════════════════════════
# API: STATUS
# ════════════════════════════════════════════════════════════
@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({
        'status':             'running',
        'model_loaded':       model is not None,
        'model_path':         MODEL_PATH,
        'preprocessor_ready': stemmer is not None and stopword_remover is not None
    })


# ════════════════════════════════════════════════════════════
# API: PREDICT (teks manual)
# ════════════════════════════════════════════════════════════
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Request: { "text": "isi berita..." }
    """
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({'error': 'Harap sertakan field "text".'}), 400

    raw_text = str(data['text']).strip()
    if not raw_text:
        return jsonify({'error': 'Teks tidak boleh kosong.'}), 400

    if model is None:
        return jsonify({
            'error': 'Model belum dimuat.',
            'hint':  f"Pastikan '{MODEL_PATH}' ada dan restart server."
        }), 503

    try:
        processed = preprocess_text(raw_text)
        if not processed.strip():
            return jsonify({'error': 'Teks tidak dapat diproses.'}), 400

        log.info(f"Predict teks: '{raw_text[:60]}...'")
        return jsonify(build_result(raw_text, processed))

    except Exception as e:
        log.error(f"Predict error: {e}")
        return jsonify({'error': str(e)}), 500


# ════════════════════════════════════════════════════════════
# API: PREDICT URL (scraping otomatis)
# ════════════════════════════════════════════════════════════
@app.route('/api/predict-url', methods=['POST'])
def api_predict_url():
    """
    Request: { "url": "https://..." }
    Response: hasil prediksi + judul + teks artikel
    """
    data = request.get_json(silent=True)
    if not data or 'url' not in data:
        return jsonify({'error': 'Harap sertakan field "url".'}), 400

    url = str(data['url']).strip()
    if not url:
        return jsonify({'error': 'URL tidak boleh kosong.'}), 400

    # Validasi format URL
    if not re.match(r'^https?://', url):
        return jsonify({'error': 'URL harus dimulai dengan http:// atau https://'}), 400

    if model is None:
        return jsonify({
            'error': 'Model belum dimuat.',
            'hint':  f"Pastikan '{MODEL_PATH}' ada dan restart server."
        }), 503

    try:
        # Scraping artikel
        log.info(f"Scraping URL: {url}")
        article = scrape_article(url)

        # Preprocessing + prediksi
        processed = preprocess_text(article['text'])
        if not processed.strip():
            return jsonify({'error': 'Konten artikel tidak dapat diproses.'}), 400

        result = build_result(article['text'], processed, source_url=url)

        # Tambah info artikel
        result['article_title']  = article['title']
        result['article_chars']  = article['chars']
        result['scrape_method']  = article['method']

        log.info(f"URL predict: {url} → {result['prediction']} ({result['confidence']}%)")
        return jsonify(result)

    except ValueError as e:
        return jsonify({'error': str(e)}), 422
    except Exception as e:
        log.error(f"URL predict error: {e}")
        return jsonify({'error': f'Gagal memproses URL: {str(e)}'}), 500


# ════════════════════════════════════════════════════════════
# API: BATCH
# ════════════════════════════════════════════════════════════
@app.route('/api/batch', methods=['POST'])
def api_batch():
    """
    Request: { "texts": ["berita 1", "berita 2", ...] }
    """
    data = request.get_json(silent=True)
    if not data or 'texts' not in data:
        return jsonify({'error': 'Harap sertakan field "texts".'}), 400

    if model is None:
        return jsonify({'error': 'Model belum dimuat.'}), 503

    texts = data['texts']
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({'error': '"texts" harus array yang tidak kosong.'}), 400

    if len(texts) > 50:
        return jsonify({'error': 'Maksimal 50 teks per batch.'}), 400

    results = []
    for idx, text in enumerate(texts):
        try:
            raw = str(text).strip()
            if not raw:
                results.append({'index': idx, 'error': 'Teks kosong'})
                continue

            processed = preprocess_text(raw)
            if not processed.strip():
                results.append({'index': idx, 'error': 'Tidak dapat diproses'})
                continue

            labels, probs = model.predict(processed, k=2)
            main_label    = labels[0].replace('__label__', '')
            main_prob     = round(float(probs[0]) * 100, 2)
            is_hoax       = main_label.lower() == 'hoax'

            results.append({
                'index':      idx,
                'text':       raw[:120] + ('...' if len(raw) > 120 else ''),
                'prediction': 'HOAX' if is_hoax else 'NON-HOAX',
                'is_hoax':    is_hoax,
                'confidence': main_prob
            })

        except Exception as e:
            results.append({'index': idx, 'error': str(e)})

    hoax_count = sum(1 for r in results if r.get('is_hoax') is True)

    return jsonify({
        'status':         'success',
        'total':          len(results),
        'hoax_count':     hoax_count,
        'non_hoax_count': len(results) - hoax_count,
        'results':        results
    })


# ════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ════════════════════════════════════════════════════════════
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint tidak ditemukan.'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Terjadi kesalahan internal server.'}), 500


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 55)
    print("  HoaxRadar — Deteksi Berita Hoax Kabupaten Sumedang")
    print("=" * 55)

    load_model()

    print()
    print(f"  Model Path : {MODEL_PATH}")
    print(f"  Model OK   : {'YES ✅' if model is not None else 'NO ❌ <- cek path model!'}")
    print()
    print("  Endpoint tersedia:")
    print("    GET  /                -> halaman frontend")
    print("    GET  /api/status      -> cek status model")
    print("    POST /api/predict     -> prediksi dari teks")
    print("    POST /api/predict-url -> prediksi dari URL berita")
    print("    POST /api/batch       -> prediksi banyak teks")
    print()
    print("  Buka browser: http://localhost:5000")
    print("=" * 55)

    app.run(debug=False, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))