// DocuVerify dashboard — session analytics, browser-local only.
// Listens for 'docuverify:result' (dispatched by app.js); never touches
// the verify/upload flow itself.
(function () {
  const KEY = 'docuverify_history_v1';
  const MAX = 100;

  function load() {
    try { return JSON.parse(localStorage.getItem(KEY)) || []; }
    catch (e) { return []; }
  }
  function save(h) {
    try { localStorage.setItem(KEY, JSON.stringify(h.slice(0, MAX))); }
    catch (e) { /* private mode etc. — dashboard simply stays empty */ }
  }

  function badge(dec) {
    const cls = dec === 'VERIFIED' ? 'v-ok' : dec === 'NEEDS_REVIEW' ? 'v-warn' : 'v-bad';
    return `<b class="${cls}">${dec.replace('_', ' ')}</b>`;
  }

  function render() {
    const h = load();
    const n = h.length;
    const cnt = { VERIFIED: 0, NEEDS_REVIEW: 0, SUSPICIOUS: 0 };
    const byType = {};
    let sum = 0, scored = 0;
    h.forEach(r => {
      if (cnt[r.decision] == null) cnt[r.decision] = 0;
      cnt[r.decision]++;
      byType[r.type] = (byType[r.type] || 0) + 1;
      if (typeof r.overall === 'number') { sum += r.overall; scored++; }
    });

    document.getElementById('stTotal').textContent = n;
    document.getElementById('stVerified').textContent = cnt.VERIFIED || 0;
    document.getElementById('stReview').textContent = cnt.NEEDS_REVIEW || 0;
    document.getElementById('stSuspicious').textContent = cnt.SUSPICIOUS || 0;
    document.getElementById('stAvg').textContent = scored ? (sum / scored).toFixed(2) : '—';

    // Donut via conic-gradient (no chart library needed).
    const donut = document.getElementById('donut');
    if (donut) {
      if (!n) {
        donut.style.background = '#1a2450';
        donut.innerHTML = '<span>—</span>';
      } else {
        const v = cnt.VERIFIED || 0, r2 = cnt.NEEDS_REVIEW || 0, s = cnt.SUSPICIOUS || 0;
        const pv = (v / n) * 100, pr = (r2 / n) * 100;
        donut.style.background =
          `conic-gradient(#22c55e 0 ${pv}%, #f59e0b ${pv}% ${pv + pr}%, #ef4444 ${pv + pr}% 100%)`;
        donut.innerHTML = `<span>${n}</span>`;
      }
    }
    const legend = document.getElementById('donutLegend');
    if (legend) {
      legend.innerHTML = ['VERIFIED', 'NEEDS_REVIEW', 'SUSPICIOUS'].map(k =>
        `<li><i class="dot ${k === 'VERIFIED' ? 'ok' : k === 'NEEDS_REVIEW' ? 'warn' : 'bad'}"></i>` +
        `${k.replace('_', ' ')} — <b>${cnt[k] || 0}</b></li>`).join('');
    }

    // Per-type bars.
    const tb = document.getElementById('typeBars');
    if (tb) {
      const types = Object.entries(byType).sort((a, b) => b[1] - a[1]).slice(0, 8);
      const max = types.length ? types[0][1] : 1;
      tb.innerHTML = types.length ? types.map(([t, c]) =>
        `<div class="trow"><label>${t}</label><div class="track"><div class="fill" style="width:${Math.round((c / max) * 100)}%"></div></div><b>${c}</b></div>`
      ).join('') : '<p class="muted">No data yet.</p>';
    }

    // History (latest first).
    const body = document.getElementById('histBody');
    const empty = document.getElementById('histEmpty');
    if (body) {
      const rows = h.slice(0, 12);
      body.innerHTML = rows.map(r =>
        `<tr><td>${r.time}</td><td>${r.type}</td><td>${badge(r.decision)}</td>` +
        `<td>${typeof r.overall === 'number' ? r.overall.toFixed(2) : '—'}</td>` +
        `<td>${typeof r.ocr === 'number' ? r.ocr.toFixed(2) : '—'}</td></tr>`
      ).join('');
      if (empty) empty.style.display = rows.length ? 'none' : 'block';
    }
  }

  document.addEventListener('docuverify:result', (e) => {
    const d = e.detail || {};
    const h = load();
    h.unshift({
      time: new Date().toLocaleString(),
      type: d.documentLabel || d.documentType || 'Unknown',
      decision: d.verdict || d.decision || 'NEEDS_REVIEW',
      overall: d.overallScore,
      ocr: d.ocrConfidence
    });
    save(h);
    render();
  });

  document.addEventListener('DOMContentLoaded', () => {
    render();
    const btn = document.getElementById('clearHistBtn');
    if (btn) btn.addEventListener('click', () => { save([]); render(); });
  });
})();
