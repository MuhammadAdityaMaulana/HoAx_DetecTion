/* ============================================================
   HoaxRadar — js/main.js  (Updated: fitur URL)
   ============================================================ */

'use strict';

/* ── KONFIGURASI ─────────────────────────────────────────────
   Ganti API_BASE sesuai URL Flask server Anda.
   ────────────────────────────────────────────────────────── */
const API_BASE = 'https://hoaxradar-production.up.railway.app';

/* ── CONTOH TEKS ─────────────────────────────────────────── */
const EXAMPLES = {
  hoax:  'Beredar akun WhatsApp mengatasnamakan Bupati Sumedang Dony Ahmad Munir yang menyatakan akan membagikan bantuan uang tunai sebesar Rp 500.000 kepada seluruh warga Sumedang. Warga diminta mengirimkan data diri dan nomor rekening untuk mendapatkan bantuan tersebut.',
  legit: 'Pemerintah Kabupaten Sumedang resmi membuka pendaftaran beasiswa pendidikan bagi siswa kurang mampu berprestasi tingkat SMA/SMK se-Kabupaten Sumedang. Pendaftaran dibuka mulai 1 hingga 31 Mei melalui website resmi Dinas Pendidikan Sumedang.'
};

/* ── MODE AKTIF ──────────────────────────────────────────── */
let currentMode = 'text'; // 'text' | 'url'

/* ── RIWAYAT ─────────────────────────────────────────────── */
let history = [];

function loadHistory() {
  try { history = JSON.parse(localStorage.getItem('hoaxradar_history') || '[]'); }
  catch { history = []; }
}

function saveHistory() {
  try { localStorage.setItem('hoaxradar_history', JSON.stringify(history)); }
  catch { /* storage penuh */ }
}

/* ════════════════════════════════════════════════════════════
   STATUS API
   ════════════════════════════════════════════════════════════ */
async function checkStatus() {
  const dot   = document.getElementById('statusDot');
  const label = document.getElementById('statusLabel');
  try {
    const res  = await fetch(`${API_BASE}/api/status`, { signal: AbortSignal.timeout(4000) });
    const data = await res.json();
    if (data.model_loaded) {
      dot.className   = 'status-dot is-online';
      label.textContent = 'Model Aktif';
      label.className   = 'status-label is-online';
    } else {
      dot.className   = 'status-dot is-offline';
      label.textContent = 'Model Belum Dimuat';
      label.className   = 'status-label is-offline';
    }
  } catch {
    dot.className   = 'status-dot is-offline';
    label.textContent = 'Server Offline';
    label.className   = 'status-label is-offline';
  }
}

/* ════════════════════════════════════════════════════════════
   TAB SWITCH
   ════════════════════════════════════════════════════════════ */
function switchTab(tab) {
  ['single','batch'].forEach(key => {
    document.getElementById(`panel${key.charAt(0).toUpperCase()+key.slice(1)}`).classList.toggle('hidden', key !== tab);
    document.getElementById(`tab${key.charAt(0).toUpperCase()+key.slice(1)}`).classList.toggle('tab--active', key === tab);
  });
}

/* ════════════════════════════════════════════════════════════
   MODE SWITCH: TEKS ↔ URL
   ════════════════════════════════════════════════════════════ */
function switchMode(mode) {
  currentMode = mode;

  const textMode = document.getElementById('inputTextMode');
  const urlMode  = document.getElementById('inputUrlMode');
  const btnText  = document.getElementById('modeText');
  const btnUrl   = document.getElementById('modeUrl');

  if (mode === 'text') {
    textMode.classList.remove('hidden');
    urlMode.classList.add('hidden');
    btnText.classList.add('mode-btn--active');
    btnText.classList.remove('url-mode');
    btnUrl.classList.remove('mode-btn--active');
  } else {
    urlMode.classList.remove('hidden');
    textMode.classList.add('hidden');
    btnUrl.classList.add('mode-btn--active', 'url-mode');
    btnText.classList.remove('mode-btn--active');
    document.getElementById('urlInput').focus();
  }

  // Reset hasil setiap ganti mode
  resetResult();
}

/* ════════════════════════════════════════════════════════════
   INPUT HANDLERS
   ════════════════════════════════════════════════════════════ */
function onInput() {
  const len = document.getElementById('newsInput').value.length;
  const el  = document.getElementById('charCount');
  el.textContent = `${len} / 2000`;
  el.classList.toggle('is-danger', len > 1800);
}

function clearInput() {
  document.getElementById('newsInput').value = '';
  onInput();
}

function loadExample(type) {
  document.getElementById('newsInput').value = EXAMPLES[type] || '';
  onInput();
}

function onUrlInput() {
  const val = document.getElementById('urlInput').value.trim();
  document.getElementById('urlClearBtn').classList.toggle('visible', val.length > 0);
}

function clearUrl() {
  document.getElementById('urlInput').value = '';
  document.getElementById('urlClearBtn').classList.remove('visible');
  resetResult();
}

function onBatchInput() {
  const n = document.getElementById('batchInput').value
    .split('\n').filter(l => l.trim()).length;
  document.getElementById('batchCount').textContent = n > 0 ? `${n} berita terdeteksi` : '';
}

/* ════════════════════════════════════════════════════════════
   RESET RESULT
   ════════════════════════════════════════════════════════════ */
function resetResult() {
  document.getElementById('resultPlaceholder').classList.remove('hidden');
  document.getElementById('resultCard').classList.add('hidden');
  document.getElementById('resultCard').innerHTML = '';
}

/* ════════════════════════════════════════════════════════════
   LOADING STATE
   ════════════════════════════════════════════════════════════ */
function setLoading(btnId, textId, isLoading, originalText) {
  const btn = document.getElementById(btnId);
  if (!btn) return;
  btn.disabled = isLoading;
  if (textId) {
    document.getElementById(textId).textContent = isLoading ? 'Menganalisis…' : originalText;
  }
  // Toggle spinner icon
  const icon = btn.querySelector('.btn-icon');
  if (icon) {
    icon.classList.toggle('spinner', isLoading);
  }
}

/* ════════════════════════════════════════════════════════════
   RENDER HASIL PREDIKSI
   ════════════════════════════════════════════════════════════ */
function renderResult(data) {
  document.getElementById('resultPlaceholder').classList.add('hidden');
  const card = document.getElementById('resultCard');
  card.classList.remove('hidden');

  const isHoax   = data.is_hoax;
  const conf     = data.confidence;
  const pHoax    = ((data.probabilities?.hoax     || 0) * 100).toFixed(1);
  const pLegit   = ((data.probabilities?.non_hoax || 0) * 100).toFixed(1);
  const clrClass = isHoax ? 'is-hoax' : 'is-legit';
  const confColor = conf >= 85 ? '#fde047' : conf >= 70 ? '#fb923c' : conf >= 55 ? '#e5e7eb' : '#9ca3af';

  const iconPath = isHoax
    ? `<path stroke-linecap="round" stroke-linejoin="round"
         d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>`
    : `<path stroke-linecap="round" stroke-linejoin="round"
         d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>`;

  /* Info artikel (jika dari URL) */
  let articleInfoHtml = '';
  if (data.source_url) {
    articleInfoHtml = `
      <div class="article-info">
        <div class="article-info-icon">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"/>
            <line x1="2" y1="12" x2="22" y2="12"/>
            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
          </svg>
        </div>
        <div class="article-info-content">
          ${data.article_title
            ? `<div class="article-info-title">${escHtml(data.article_title)}</div>`
            : ''}
          <div class="article-info-meta">${data.article_chars || 0} karakter · via ${escHtml(data.scrape_method || '')}</div>
          <a class="article-info-link" href="${escHtml(data.source_url)}" target="_blank" rel="noopener">
            ${escHtml(data.source_url.length > 60 ? data.source_url.slice(0, 60) + '…' : data.source_url)}
          </a>
        </div>
      </div>`;
  }

  card.innerHTML = `
    <div class="result-content ${clrClass}">

      <!-- Verdict -->
      <div class="result-header">
        <div class="result-icon ${clrClass}">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            ${iconPath}
          </svg>
        </div>
        <div>
          <p class="result-label-sm">${data.source_url ? 'Hasil Analisis URL' : 'Hasil Prediksi'}</p>
          <p class="result-verdict ${clrClass}">${escHtml(data.prediction)}</p>
        </div>
      </div>

      ${articleInfoHtml}

      <!-- Confidence -->
      <div class="bar-row">
        <div class="bar-label-row">
          <span>Kepercayaan</span>
          <span class="bar-value" style="color:${confColor}">${conf}% · ${escHtml(data.confidence_level)}</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill ${isHoax ? 'red' : 'green'}" id="confBar" data-target="${conf}"></div>
        </div>
      </div>

      <!-- Probability -->
      <div class="prob-section">
        <div class="prob-row">
          <div class="prob-label-row">
            <span class="lbl-red">HOAX</span>
            <span class="lbl-val">${pHoax}%</span>
          </div>
          <div class="bar-track thin">
            <div class="bar-fill red" id="barHoax" data-target="${pHoax}"></div>
          </div>
        </div>
        <div class="prob-row">
          <div class="prob-label-row">
            <span class="lbl-green">NON-HOAX</span>
            <span class="lbl-val">${pLegit}%</span>
          </div>
          <div class="bar-track thin">
            <div class="bar-fill green" id="barLegit" data-target="${pLegit}"></div>
          </div>
        </div>
      </div>

      <div class="divider"></div>

      <!-- Processed text -->
      <details>
        <summary>Teks setelah preprocessing ▾</summary>
        <div class="processed-text-box">${escHtml(data.processed_text || '')}</div>
      </details>

    </div>`;

  /* Animasikan bar */
  requestAnimationFrame(() => {
    setTimeout(() => {
      ['confBar','barHoax','barLegit'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.width = el.dataset.target + '%';
      });
    }, 60);
  });
}

/* ════════════════════════════════════════════════════════════
   PREDICT — TEKS
   ════════════════════════════════════════════════════════════ */
async function doPredict() {
  const text = (document.getElementById('newsInput').value || '').trim();
  if (!text) { showToast('Masukkan teks berita terlebih dahulu.', 'warn'); return; }

  setLoading('btnAnalyze', 'btnText', true);

  try {
    const res  = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const data = await res.json();
    if (!res.ok) { showToast(data.error || 'Terjadi kesalahan.', 'error'); return; }
    renderResult(data);
    addToHistory(data, 'text');
  } catch {
    showToast('Tidak dapat terhubung ke server Flask (localhost:5000).', 'error');
  } finally {
    setLoading('btnAnalyze', 'btnText', false, 'Analisis Sekarang');
  }
}

/* ════════════════════════════════════════════════════════════
   PREDICT — URL
   ════════════════════════════════════════════════════════════ */
async function doUrlPredict() {
  const url = (document.getElementById('urlInput').value || '').trim();
  if (!url) { showToast('Masukkan URL berita terlebih dahulu.', 'warn'); return; }

  if (!url.match(/^https?:\/\//)) {
    showToast('URL harus dimulai dengan http:// atau https://', 'warn');
    return;
  }

  setLoading('btnUrlAnalyze', 'btnUrlText', true);
  showToast('Mengambil konten dari URL…', 'info');

  try {
    const res  = await fetch(`${API_BASE}/api/predict-url`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });
    const data = await res.json();

    if (!res.ok) {
      showToast(data.error || 'Gagal mengambil konten URL.', 'error');
      return;
    }

    renderResult(data);
    addToHistory(data, 'url');

  } catch {
    showToast('Tidak dapat terhubung ke server Flask (localhost:5000).', 'error');
  } finally {
    setLoading('btnUrlAnalyze', 'btnUrlText', false, 'Analisis URL');
  }
}

/* ════════════════════════════════════════════════════════════
   RIWAYAT
   ════════════════════════════════════════════════════════════ */
function addToHistory(data, type = 'text') {
  const label = type === 'url'
    ? (data.article_title || data.source_url || 'URL Berita').slice(0, 78)
    : (data.input_text || '').slice(0, 78) + ((data.input_text || '').length > 78 ? '…' : '');

  history.unshift({
    id:    Date.now(),
    label,
    type,
    pred:  data.prediction,
    hoax:  data.is_hoax,
    conf:  data.confidence,
    url:   data.source_url || null,
    text:  type === 'text' ? (data.input_text || '') : '',
    time:  new Date().toLocaleTimeString('id-ID', { hour:'2-digit', minute:'2-digit' })
  });

  if (history.length > 12) history.pop();
  saveHistory();
  renderHistory();
}

function renderHistory() {
  const section = document.getElementById('historySection');
  const list    = document.getElementById('historyList');
  if (!history.length) { section.classList.add('hidden'); return; }
  section.classList.remove('hidden');

  list.innerHTML = history.map(e => {
    const icon = e.type === 'url'
      ? `<svg class="hist-icon" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
           <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>
           <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>
         </svg>`
      : `<span class="hist-dot ${e.hoax ? 'red' : 'green'}"></span>`;

    return `
      <div class="hist-row" onclick="loadFromHistory(${e.id})" title="Klik untuk muat ulang">
        ${icon}
        <span class="hist-text">${escHtml(e.label)}</span>
        <span class="hist-pred ${e.hoax ? 'red' : 'green'}">${escHtml(e.pred)}</span>
        <span class="hist-time">${escHtml(e.time)}</span>
      </div>`;
  }).join('');
}

function loadFromHistory(id) {
  const e = history.find(h => h.id === id);
  if (!e) return;

  if (e.type === 'url' && e.url) {
    switchMode('url');
    document.getElementById('urlInput').value = e.url;
    onUrlInput();
  } else {
    switchMode('text');
    document.getElementById('newsInput').value = e.text || e.label.replace(/…$/, '');
    onInput();
  }

  switchTab('single');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function clearHistory() {
  history = [];
  saveHistory();
  renderHistory();
}

/* ════════════════════════════════════════════════════════════
   BATCH PREDICT
   ════════════════════════════════════════════════════════════ */
async function doBatch() {
  const texts = (document.getElementById('batchInput').value || '')
    .split('\n').map(l => l.trim()).filter(l => l);

  if (!texts.length) { showToast('Masukkan minimal satu berita.', 'warn'); return; }
  if (texts.length > 50) { showToast('Maksimal 50 berita per batch.', 'warn'); return; }

  setLoading('btnBatch', 'btnBatchText', true);

  try {
    const res  = await fetch(`${API_BASE}/api/batch`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts })
    });
    const data = await res.json();
    if (!res.ok) { showToast(data.error || 'Gagal.', 'error'); return; }
    renderBatchResults(data);
  } catch {
    showToast('Tidak dapat terhubung ke server Flask.', 'error');
  } finally {
    setLoading('btnBatch', 'btnBatchText', false, 'Analisis Batch');
  }
}

function renderBatchResults(data) {
  const out      = document.getElementById('batchOutput');
  const hoaxPct  = data.total > 0 ? ((data.hoax_count / data.total) * 100).toFixed(0) : 0;

  const rows = (data.results || []).map((r, i) => {
    if (r.error) return `
      <div class="batch-row error">
        <span class="batch-num">${i+1}</span>
        <span class="batch-text" style="color:#fca5a5;">${escHtml(r.error)}</span>
      </div>`;
    const cls = r.is_hoax ? 'hoax' : 'legit';
    return `
      <div class="batch-row ${cls}">
        <span class="batch-num">${i+1}</span>
        <span class="batch-text">${escHtml(r.text||'')}</span>
        <div class="batch-result">
          <span class="batch-pred ${cls}">${escHtml(r.prediction||'')}</span>
          ${r.confidence != null ? `<span class="batch-conf">${r.confidence}%</span>` : ''}
        </div>
      </div>`;
  }).join('');

  out.className = 'animate-up';
  out.innerHTML = `
    <div class="card batch-summary" style="margin-top:20px;">
      <p class="batch-summary-title">Ringkasan Batch</p>
      <div class="batch-stats-grid">
        <div class="batch-stat neutral"><span class="batch-stat-val neutral">${data.total}</span><span class="batch-stat-lbl">Total</span></div>
        <div class="batch-stat hoax">  <span class="batch-stat-val hoax">${data.hoax_count}</span><span class="batch-stat-lbl">Hoax</span></div>
        <div class="batch-stat legit"> <span class="batch-stat-val legit">${data.non_hoax_count}</span><span class="batch-stat-lbl">Non-Hoax</span></div>
      </div>
      <div class="bar-track">
        <div class="bar-fill red" id="batchBar" data-target="${hoaxPct}" style="width:0%"></div>
      </div>
      <p class="batch-pct-note">${hoaxPct}% terdeteksi hoax</p>
    </div>
    <div class="batch-list" style="margin-top:10px;">${rows}</div>`;

  requestAnimationFrame(() => {
    setTimeout(() => {
      const b = document.getElementById('batchBar');
      if (b) b.style.width = b.dataset.target + '%';
    }, 60);
  });
}

/* ════════════════════════════════════════════════════════════
   TOAST
   ════════════════════════════════════════════════════════════ */
function showToast(msg, type = 'info') {
  // Hapus toast lama dengan pesan sama
  document.querySelectorAll('.toast').forEach(t => { if (t.textContent === msg) t.remove(); });

  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(10px)';
    el.style.transition = 'opacity .3s,transform .3s';
    setTimeout(() => el.remove(), 320);
  }, 3500);
}

/* ════════════════════════════════════════════════════════════
   UTILS
   ════════════════════════════════════════════════════════════ */
function escHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#x27;');
}

/* ════════════════════════════════════════════════════════════
   KEYBOARD SHORTCUTS
   ════════════════════════════════════════════════════════════ */
document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    if (currentMode === 'url') doUrlPredict();
    else doPredict();
  }
});

/* ════════════════════════════════════════════════════════════
   INIT
   ════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  loadHistory();
  renderHistory();
  checkStatus();
  setInterval(checkStatus, 30000);
});