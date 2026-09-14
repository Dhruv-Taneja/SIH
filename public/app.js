// DocuVerify frontend — upload / drag-drop / camera scan → POST /api/upload
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const docType = document.getElementById('docType');
const verifyBtn = document.getElementById('verifyBtn');
const statusEl = document.getElementById('status');
const previewBox = document.getElementById('previewBox');
const previewImg = document.getElementById('previewImg');
const fileName = document.getElementById('fileName');
const clearBtn = document.getElementById('clearBtn');

const scanBtn = document.getElementById('scanBtn');
const cameraBox = document.getElementById('cameraBox');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('captureBtn');
const closeCamBtn = document.getElementById('closeCamBtn');

let selectedFile = null;
let stream = null;

function setStatus(msg, cls = '') {
  statusEl.textContent = msg;
  statusEl.className = 'status ' + cls;
}

function setFile(file) {
  selectedFile = file;
  verifyBtn.disabled = !file;
  if (!file) { previewBox.classList.add('hidden'); return; }
  fileName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  if (file.type.startsWith('image/')) {
    previewImg.src = URL.createObjectURL(file);
    previewBox.classList.remove('hidden');
  } else {
    previewImg.src = '';
    previewBox.classList.remove('hidden');
  }
  setStatus('');
}

browseBtn.onclick = (e) => { e.stopPropagation(); fileInput.click(); };
dropzone.onclick = () => fileInput.click();
fileInput.onchange = (e) => { if (e.target.files[0]) setFile(e.target.files[0]); };
clearBtn.onclick = () => { setFile(null); fileInput.value = ''; };

['dragover', 'dragenter'].forEach(ev => dropzone.addEventListener(ev, (e) => {
  e.preventDefault(); dropzone.classList.add('drag');
}));
['dragleave', 'drop'].forEach(ev => dropzone.addEventListener(ev, (e) => {
  e.preventDefault(); dropzone.classList.remove('drag');
}));
dropzone.addEventListener('drop', (e) => {
  const f = e.dataTransfer.files[0];
  if (f) setFile(f);
});

// ---- Camera scan ----
scanBtn.onclick = async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
    video.srcObject = stream;
    cameraBox.classList.remove('hidden');
    scanBtn.classList.add('hidden');
  } catch (err) {
    setStatus('Camera not available: ' + err.message, 'error');
  }
};
function stopCamera() {
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null;
  cameraBox.classList.add('hidden');
  scanBtn.classList.remove('hidden');
}
closeCamBtn.onclick = stopCamera;
captureBtn.onclick = () => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  canvas.toBlob((blob) => {
    const f = new File([blob], 'scan-' + Date.now() + '.png', { type: 'image/png' });
    setFile(f);
    stopCamera();
    setStatus('Scan captured — hit Verify.');
    document.getElementById('verify').scrollIntoView({ behavior: 'smooth' });
  }, 'image/png');
};

// ---- Engine status: warn early if photo OCR isn't installed ----
fetch('/api/health').then(r => r.json()).then(h => {
  const e = h.engines;
  if (e && e.image_ocr_ready === false) {
    setStatus('⚠ Photo OCR engine not installed here — digital PDFs verify fine; for JPG/PNG photos run: pip install rapidocr-onnxruntime (or see README).', 'error');
  }
}).catch(() => {});

// ---- Verify ----
verifyBtn.onclick = async () => {
  if (!selectedFile) return;
  const fd = new FormData();
  fd.append('document', selectedFile);
  fd.append('documentType', docType.value);
  setStatus('Uploading & verifying… (OCR can take 20–60s first run)', 'loading');
  verifyBtn.disabled = true;
  try {
    const res = await fetch('/api/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (res.status === 401) {
      // Session expired / signed out — hand off to the auth UI (auth.js).
      document.dispatchEvent(new CustomEvent('docuverify:auth-required'));
      throw new Error('Please sign in as an organization to verify documents.');
    }
    if (!res.ok) throw new Error(data.error || 'Verification failed');
    renderResult(data);
    setStatus('Done ✔', '');
  } catch (err) {
    setStatus('Error: ' + err.message, 'error');
  } finally {
    verifyBtn.disabled = false;
  }
};

function renderResult(d) {
  document.getElementById('emptyResult').classList.add('hidden');
  document.getElementById('resultContent').classList.remove('hidden');
  const badge = document.getElementById('verdictBadge');
  badge.className = 'verdict ' + d.verdict;
  const icon = d.verdict === 'VERIFIED' ? '✔' : d.verdict === 'NEEDS_REVIEW' ? '⚠' : '✖';
  const dtype = d.documentLabel || d.documentType || '';
  badge.textContent = `${icon} ${d.verdict.replace('_', ' ')}${dtype ? ' · ' + dtype : ''}`;

  // Explainable evidence (§16): reasons + per-layer status, not just a score.
  const reasons = d.reasons || [];
  document.getElementById('reasonsBox').innerHTML =
    reasons.length ? '<h4>Why this decision</h4><ul>' +
      reasons.map(r => `<li>${r}</li>`).join('') + '</ul>' : '';
  const pct = (x) => x == null ? '—' : Math.round(x * 100) + '%';
  const cls = d.document_classification || {};
  const alts = cls.alternatives ? Object.entries(cls.alternatives).slice(0, 2)
    .map(([t, p]) => `${t} ${pct(p)}`).join(', ') : '';
  document.getElementById('evType').innerHTML =
    `<b>${dtype}</b> · ${pct(cls.confidence)} confidence` + (alts ? `<br/><small>also: ${alts}</small>` : '');
  const q = d.image_quality || {};
  const qSym = q.status === 'PASS' ? '✓' : q.status === 'N/A' ? '○' : '⚠';
  document.getElementById('evQuality').innerHTML =
    `${qSym} ${q.status || '—'} · ${pct(q.overall)}` +
    (q.blur_variance != null ? `<br/><small>blur ${q.blur_variance}, brightness ${q.mean_gray}</small>` : '') +
    (q.note ? `<br/><small>${q.note}</small>` : '');
  const ev = d.evidence || {};
  const sym = (s) => s === 'PASS' ? '✓' : s === 'FAIL' ? '✖' : s === 'REVIEW' ? '⚠' : '○';
  document.getElementById('evOcr').textContent =
    `${sym(ev.ocr && ev.ocr.status)} ${ev.ocr && ev.ocr.status || '—'} · OCR confidence ${pct(d.ocrConfidence)}`;
  document.getElementById('evStruct').textContent =
    `${sym(ev.structure && ev.structure.status)} ${ev.structure && ev.structure.status || '—'} · score ${pct(d.syntaxScore)}`;
  const cf = (d.cross_field || {});
  const nConf = (cf.conflicts || []).length;
  document.getElementById('evCross').innerHTML =
    `${sym(cf.status)} ${cf.status || '—'}` +
    (nConf ? `<br/><small>${nConf} conflict(s): ${cf.conflicts.map(c => `${c.field}: ${c.value_1} vs ${c.value_2}`).join('; ').slice(0, 160)}</small>`
           : (cf.note ? `<br/><small>${cf.note}</small>` : ''));
  const tp = d.tamper || {};
  const qr = (tp.qr || {});
  const qrTxt = `QR: ${qr.detected ? (qr.readable ? 'detected ✓ readable ✓' : 'detected, unreadable ⚠') : 'none detected'} · authenticity not established`;
  document.getElementById('evTamper').innerHTML =
    `${sym(tp.status)} ${tp.status || '—'}<br/><small>${qrTxt}</small>` +
    ((tp.flags || []).length ? `<br/><small>⚠ ${tp.flags.join('; ').slice(0, 200)}</small>` : '');

  const setBar = (id, vId, v) => {
    document.getElementById(id).style.width = Math.round((v || 0) * 100) + '%';
    document.getElementById(vId).textContent = (v ?? 0).toFixed(2);
  };
  setBar('sOverall', 'sOverallV', d.overallScore);
  setBar('sOcr', 'sOcrV', d.ocrConfidence);
  setBar('sSyn', 'sSynV', d.syntaxScore);
  setBar('sSem', 'sSemV', d.semanticScore);
  document.getElementById('engine').textContent = d.engine || 'paddleocr';

  const list = document.getElementById('checksList');
  list.innerHTML = '';
  (d.checks || []).forEach(c => {
    const li = document.createElement('li');
    li.className = c.status;
    li.innerHTML = `<b class="${c.status}">${c.status}</b> · ${c.check}<br/>${c.message}` +
      (c.value ? `<br/><small>→ ${String(c.value).slice(0, 120)}</small>` : '');
    list.appendChild(li);
  });
  document.getElementById('ocrText').textContent = d.extractedText || '(no text extracted)';
  // Dashboard hook (dashboard.js listens) — verify flow above is untouched.
  document.dispatchEvent(new CustomEvent('docuverify:result', { detail: d }));
}
