// =============================================
//  main.js
// =============================================
const API_BASE = 'https://xxxx-xxxx.ngrok-free.app';
const MODULE_DATA = {
  '1': { index: '01', label: 'MODULE 01', title: '图像阈值分割' },
  '2': { index: '02', label: 'MODULE 02', title: '图像特征提取' },
  '3': { index: '03', label: 'MODULE 03', title: '剩余寿命预测' },
};

// ─── Modal ───────────────────────────────────
const backdrop   = document.getElementById('modalBackdrop');
const modalIndex = document.getElementById('modalIndex');
const modalLabel = document.getElementById('modalLabel');
const modalTitle = document.getElementById('modalTitle');
const modalBody  = document.getElementById('modalBody');
const closeBtn   = document.getElementById('modalClose');
let lastFocused  = null;

function openModal(moduleId) {
  const data = MODULE_DATA[moduleId];
  if (!data) return;
  modalIndex.textContent = data.index;
  modalLabel.textContent = data.label;
  modalTitle.textContent = data.title;

  if      (moduleId === '1') buildSegmentationUI();
  else if (moduleId === '2') buildExtractionUI();
  else if (moduleId === '3') buildPredictionUI();

  backdrop.setAttribute('aria-hidden', 'false');
  backdrop.classList.add('is-open');
  document.body.style.overflow = 'hidden';
  lastFocused = document.activeElement;
  setTimeout(() => closeBtn.focus(), 50);
}

function closeModal() {
  backdrop.classList.remove('is-open');
  backdrop.setAttribute('aria-hidden', 'true');
  document.body.style.overflow = '';
  if (lastFocused) lastFocused.focus();
}

document.querySelectorAll('.module').forEach(mod => {
  mod.addEventListener('click', () => openModal(mod.dataset.module));
  mod.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openModal(mod.dataset.module); }
  });
});
closeBtn.addEventListener('click', closeModal);
backdrop.addEventListener('click', e => { if (e.target === backdrop) closeModal(); });
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && backdrop.classList.contains('is-open')) closeModal();
});

// =============================================
//  MODULE 1 — Segmentation
// =============================================
function buildSegmentationUI() {
  const tpl = document.getElementById('tpl-segmentation');
  modalBody.innerHTML = '';
  modalBody.appendChild(tpl.content.cloneNode(true));

  const statusDot         = modalBody.querySelector('#segStatusDot');
  const statusText        = modalBody.querySelector('#segStatusText');
  const dropZone          = modalBody.querySelector('#dropZone');
  const fileInput         = modalBody.querySelector('#fileInput');
  const segFileList       = modalBody.querySelector('#segFileList');
  const inputPreviewWrap  = modalBody.querySelector('#inputPreviewWrap');
  const inputPreview      = modalBody.querySelector('#inputPreview');
  const inputMeta         = modalBody.querySelector('#inputMeta');
  const runBtn            = modalBody.querySelector('#runBtn');
  const resultPlaceholder = modalBody.querySelector('#resultPlaceholder');
  const resultLoading     = modalBody.querySelector('#resultLoading');
  const loaderText        = modalBody.querySelector('#loaderText');
  const resultImg         = modalBody.querySelector('#resultImg');
  const segActions        = modalBody.querySelector('#segActions');
  const downloadBtn       = modalBody.querySelector('#downloadBtn');
  const downloadAllBtn    = modalBody.querySelector('#downloadAllBtn');
  const resetBtn          = modalBody.querySelector('#resetBtn');
  const segNav            = modalBody.querySelector('#segNav');
  const segNavLabel       = modalBody.querySelector('#segNavLabel');
  const segPrev           = modalBody.querySelector('#segPrev');
  const segNext           = modalBody.querySelector('#segNext');

  let selectedFiles = [];   // 批量文件列表
  let resultBlobs   = [];   // 每张图对应的结果 Blob
  let currentIdx    = 0;

  function setStatus(state, text) {
    statusDot.className = 'seg-status-dot ' + state;
    statusText.textContent = text;
  }
  setStatus('loading', '正在检测模型状态…');
  fetch(API_BASE + '/api/status').then(r => r.json()).then(data => {
    if (data.model_loaded) setStatus('ready', `模型已就绪：${data.model_file}`);
    else { setStatus('error', '模型未加载 — 请将 .pth 文件放入 models/ 并重启服务'); runBtn.disabled = true; }
  }).catch(() => setStatus('error', '无法连接后端 — 请确认 app.py 已启动（localhost:5000）'));

  function renderFileList() {
    if (!selectedFiles.length) { segFileList.style.display = 'none'; return; }
    segFileList.style.display = 'block';
    segFileList.innerHTML = selectedFiles.map((f, i) =>
      `<div class="ext-file-item">
        <span class="fname">${f.name}</span>
        <span>${(f.size/1024).toFixed(1)} KB</span>
      </div>`
    ).join('');
  }

  function loadFiles(files) {
    selectedFiles = Array.from(files);
    resultBlobs = []; currentIdx = 0;
    renderFileList();
    dropZone.style.display = 'none';
    // 预览第一张
    inputPreview.src = URL.createObjectURL(selectedFiles[0]);
    inputMeta.textContent = `已选择 ${selectedFiles.length} 张图像，当前预览：${selectedFiles[0].name}`;
    inputPreviewWrap.style.display = 'block';
    runBtn.disabled = false;
    resultImg.style.display = 'none'; resultImg.src = '';
    resultPlaceholder.style.display = 'flex';
    resultPlaceholder.innerHTML = `<span class="result-ph-icon">◈</span><span>结果将在此处显示</span>`;
    segActions.style.display = 'none';
    segNav.style.display = 'none';
  }

  dropZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', e => { if (e.target.files.length) loadFiles(e.target.files); });
  dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) loadFiles(e.dataTransfer.files);
  });

  function showResult(idx) {
    if (!resultBlobs[idx]) return;
    resultImg.src = URL.createObjectURL(resultBlobs[idx]);
    resultImg.style.display = 'block';
    resultPlaceholder.style.display = 'none';
    inputPreview.src = URL.createObjectURL(selectedFiles[idx]);
    inputMeta.textContent = `${idx + 1} / ${selectedFiles.length}：${selectedFiles[idx].name}`;
    if (selectedFiles.length > 1) {
      segNav.style.display = 'flex';
      segNavLabel.textContent = `图像 ${idx + 1} / ${selectedFiles.length}`;
    }
  }

  runBtn.addEventListener('click', async () => {
    if (!selectedFiles.length) return;
    runBtn.disabled = true;
    resultBlobs = [];
    resultPlaceholder.style.display = 'none'; resultImg.style.display = 'none';
    resultLoading.style.display = 'flex';
    segActions.style.display = 'none'; segNav.style.display = 'none';

    for (let i = 0; i < selectedFiles.length; i++) {
      loaderText.textContent = `正在分割 ${i + 1} / ${selectedFiles.length}：${selectedFiles[i].name}`;
      try {
        const fd = new FormData(); fd.append('image', selectedFiles[i]);
        const resp = await fetch(API_BASE + '/api/segment', { method: 'POST', body: fd });
        if (!resp.ok) { const e = await resp.json(); throw new Error(e.error || `HTTP ${resp.status}`); }
        resultBlobs.push(await resp.blob());
      } catch (err) {
        resultBlobs.push(null);
        console.warn(`图像 ${selectedFiles[i].name} 分割失败：${err.message}`);
      }
    }

    resultLoading.style.display = 'none';
    currentIdx = 0;
    showResult(0);
    segActions.style.display = 'flex';
    runBtn.disabled = false;
  });

  segPrev.addEventListener('click', () => { if (currentIdx > 0) { currentIdx--; showResult(currentIdx); } });
  segNext.addEventListener('click', () => { if (currentIdx < selectedFiles.length - 1) { currentIdx++; showResult(currentIdx); } });

  downloadBtn.addEventListener('click', () => {
    const blob = resultBlobs[currentIdx];
    if (!blob) return;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `seg_${selectedFiles[currentIdx].name.replace(/\.[^.]+$/, '')}.png`;
    a.click();
  });

  downloadAllBtn.addEventListener('click', async () => {
    // 逐个下载（无 JSZip 依赖）
    resultBlobs.forEach((blob, i) => {
      if (!blob) return;
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `seg_${selectedFiles[i].name.replace(/\.[^.]+$/, '')}.png`;
      a.click();
    });
  });

  resetBtn.addEventListener('click', () => {
    selectedFiles = []; resultBlobs = []; currentIdx = 0; fileInput.value = '';
    segFileList.style.display = 'none'; segFileList.innerHTML = '';
    inputPreviewWrap.style.display = 'none'; dropZone.style.display = '';
    runBtn.disabled = true; resultImg.style.display = 'none'; resultImg.src = '';
    resultPlaceholder.style.display = 'flex';
    resultPlaceholder.innerHTML = `<span class="result-ph-icon">◈</span><span>结果将在此处显示</span>`;
    segActions.style.display = 'none'; segNav.style.display = 'none';
  });
}

// =============================================
//  MODULE 2 — Feature Extraction
// =============================================
function buildExtractionUI() {
  const tpl = document.getElementById('tpl-extraction');
  modalBody.innerHTML = '';
  modalBody.appendChild(tpl.content.cloneNode(true));

  const extDropZone    = modalBody.querySelector('#extDropZone');
  const extFileInput   = modalBody.querySelector('#extFileInput');
  const extFileList    = modalBody.querySelector('#extFileList');
  const extMag         = modalBody.querySelector('#extMag');
  const extAngle       = modalBody.querySelector('#extAngle');
  const extRunBtn      = modalBody.querySelector('#extRunBtn');
  const extLoading     = modalBody.querySelector('#extLoading');
  const extLoaderText  = modalBody.querySelector('#extLoaderText');
  const extResultWrap  = modalBody.querySelector('#extResultWrap');
  const extTableBody   = modalBody.querySelector('#extTableBody');
  const extNav         = modalBody.querySelector('#extNav');
  const extNavLabel    = modalBody.querySelector('#extNavLabel');
  const extPrev        = modalBody.querySelector('#extPrev');
  const extNext        = modalBody.querySelector('#extNext');
  const extDownloadBtn = modalBody.querySelector('#extDownloadBtn');
  const extResetBtn    = modalBody.querySelector('#extResetBtn');

  let selectedFiles = [], allResults = [], currentIdx = 0;

  const ROWS = [
    { label: 'γ 相宽度 ω (μm)',  rawKey: 'omega_raw',       normKey: 'omega_norm' },
    { label: "γ' 相宽度 l (μm)", rawKey: 'l_raw',           normKey: 'l_norm' },
    { label: 'UMAP1',             rawKey: 'umap1_raw',       normKey: 'umap1_norm' },
    { label: 'UMAP2',             rawKey: 'umap2_raw',       normKey: 'umap2_norm' },
    { label: '体积分数',           rawKey: 'volume_fraction', normKey: 'volume_fraction' },
  ];

  function renderFileList() {
    extFileList.style.display = selectedFiles.length ? 'block' : 'none';
    extFileList.innerHTML = selectedFiles.map(f =>
      `<div class="ext-file-item"><span class="fname">${f.name}</span>
       <span style="color:var(--clr-muted)">${(f.size/1024).toFixed(0)} KB</span></div>`
    ).join('');
    extRunBtn.disabled = false;
  }

  function loadFiles(files) {
    selectedFiles = Array.from(files);
    renderFileList();
    extDropZone.style.display = 'none';
    extResultWrap.style.display = 'none'; allResults = []; currentIdx = 0;
  }

  extDropZone.addEventListener('click', () => extFileInput.click());
  extFileInput.addEventListener('change', e => { if (e.target.files.length) loadFiles(e.target.files); });
  extDropZone.addEventListener('dragover',  e => { e.preventDefault(); extDropZone.classList.add('drag-over'); });
  extDropZone.addEventListener('dragleave', () => extDropZone.classList.remove('drag-over'));
  extDropZone.addEventListener('drop', e => {
    e.preventDefault(); extDropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) loadFiles(e.dataTransfer.files);
  });

  extRunBtn.addEventListener('click', async () => {
    if (!selectedFiles.length) return;
    extRunBtn.disabled = true; extResultWrap.style.display = 'none';
    extLoading.style.display = 'flex'; extLoaderText.textContent = '正在计算 CCA 交叉相关特征…';
    try {
      const fd = new FormData();
      selectedFiles.forEach(f => fd.append('images', f));
      fd.append('magnification', extMag.value);
      fd.append('angle', extAngle.value);
      const steps = [[1200,'UMAP 降维中…'],[2500,'运行 CLD 割线分析…'],[4000,'GMM 拟合相宽度分布…'],[5500,'汇总特征结果…']];
      const timers = steps.map(([ms, txt]) => setTimeout(() => { extLoaderText.textContent = txt; }, ms));
      const resp = await fetch(API_BASE + '/api/extract', { method: 'POST', body: fd });
      timers.forEach(clearTimeout);
      if (!resp.ok) { const e = await resp.json(); throw new Error(e.error || `HTTP ${resp.status}`); }
      allResults = await resp.json(); currentIdx = 0;
      extLoading.style.display = 'none';
      renderTable(allResults[currentIdx]);
      extResultWrap.style.display = 'block';
      extNav.style.display = allResults.length > 1 ? 'flex' : 'none';
      if (allResults.length > 1) updateNav();
    } catch (err) {
      extLoading.style.display = 'none';
      const errBox = document.createElement('div');
      errBox.style.cssText = 'border:1px solid var(--clr-red);padding:16px;font-size:0.8rem;color:var(--clr-red);margin-top:16px;';
      errBox.textContent = '错误: ' + err.message;
      modalBody.querySelector('.ext-layout').appendChild(errBox);
    } finally { extRunBtn.disabled = false; }
  });

  function renderTable(result) {
    extTableBody.innerHTML = ROWS.map(row => {
      const rv = result[row.rawKey], nv = result[row.normKey];
      const rs = (rv == null || isNaN(rv)) ? '—' : Number(rv).toFixed(4);
      const ns = (nv == null || isNaN(nv)) ? '—' : Number(nv).toFixed(4);
      return `<tr><td class="col-feature">${row.label}</td>
        <td class="col-raw val">${rs}</td>
        <td class="col-norm val val-cyan">${ns}</td></tr>`;
    }).join('');
    const existing = extResultWrap.querySelector('.ext-filename');
    if (existing) existing.remove();
    if (result.filename) {
      const cap = document.createElement('p');
      cap.className = 'ext-filename seg-col-label';
      cap.style.cssText = 'margin-bottom:8px;color:var(--clr-cyan);';
      cap.textContent = `// ${result.filename}`;
      extResultWrap.querySelector('.ext-table-wrap').before(cap);
    }
  }

  function updateNav() {
    extNavLabel.textContent = `图像 ${currentIdx + 1} / ${allResults.length}`;
    extPrev.disabled = currentIdx === 0;
    extNext.disabled = currentIdx === allResults.length - 1;
  }
  extPrev.addEventListener('click', () => { if (currentIdx > 0) { currentIdx--; renderTable(allResults[currentIdx]); updateNav(); } });
  extNext.addEventListener('click', () => { if (currentIdx < allResults.length-1) { currentIdx++; renderTable(allResults[currentIdx]); updateNav(); } });

  extDownloadBtn.addEventListener('click', () => {
    if (!allResults.length) return;
    const header = 'filename,omega_raw,omega_norm,l_raw,l_norm,umap1_raw,umap1_norm,umap2_raw,umap2_norm,volume_fraction';
    const rows = allResults.map(r => [r.filename,r.omega_raw,r.omega_norm,r.l_raw,r.l_norm,r.umap1_raw,r.umap1_norm,r.umap2_raw,r.umap2_norm,r.volume_fraction].join(','));
    const blob = new Blob([[header,...rows].join('\n')], { type: 'text/csv;charset=utf-8;' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
    a.download = 'feature_extraction_results.csv'; a.click();
  });

  extResetBtn.addEventListener('click', () => {
    selectedFiles = []; allResults = []; currentIdx = 0; extFileInput.value = '';
    extFileList.style.display = 'none'; extDropZone.style.display = '';
    extRunBtn.disabled = true; extResultWrap.style.display = 'none'; extLoading.style.display = 'none';
  });
}

// =============================================
//  MODULE 3 — Life Prediction
// =============================================
function buildPredictionUI() {
  const tpl = document.getElementById('tpl-prediction');
  modalBody.innerHTML = '';
  modalBody.appendChild(tpl.content.cloneNode(true));

  const pTemp    = modalBody.querySelector('#pTemp');
  const pStress  = modalBody.querySelector('#pStress');
  const pUmap1   = modalBody.querySelector('#pUmap1');
  const pUmap2   = modalBody.querySelector('#pUmap2');
  const pOmega   = modalBody.querySelector('#pOmega');
  const pL       = modalBody.querySelector('#pL');
  const runBtn   = modalBody.querySelector('#predRunBtn');
  const loading  = modalBody.querySelector('#predLoading');
  const result   = modalBody.querySelector('#predResult');
  const lifeVal  = modalBody.querySelector('#predLifeVal');
  const lifeSub  = modalBody.querySelector('#predLifeSub');
  const predLow  = modalBody.querySelector('#predLow');
  const predHigh = modalBody.querySelector('#predHigh');
  const confBar  = modalBody.querySelector('#predConfBar');
  const confMark = modalBody.querySelector('#predConfMarker');
  const resetBtn = modalBody.querySelector('#predResetBtn');

  const inputs   = [pTemp, pStress, pUmap1, pUmap2, pOmega, pL];

  // Clear error highlight on input
  inputs.forEach(inp => inp.addEventListener('input', () => inp.classList.remove('error')));

  function validate() {
    let ok = true;
    inputs.forEach(inp => {
      if (inp.value.trim() === '' || isNaN(Number(inp.value))) {
        inp.classList.add('error'); ok = false;
      } else { inp.classList.remove('error'); }
    });
    return ok;
  }

  runBtn.addEventListener('click', async () => {
    if (!validate()) return;

    runBtn.disabled = true;
    result.style.display = 'none';
    loading.style.display = 'flex';

    const payload = {
      temp:       parseFloat(pTemp.value),
      stress:     parseFloat(pStress.value),
      umap1_norm: parseFloat(pUmap1.value),
      umap2_norm: parseFloat(pUmap2.value),
      omega_norm: parseFloat(pOmega.value),
      l_norm:     parseFloat(pL.value),
    };

    try {
      const resp = await fetch(API_BASE + '/api/predict', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);

      loading.style.display = 'none';

      // Animate the life value counting up
      const target = data.life_h;
      const duration = 900;
      const start = performance.now();
      const tick = (now) => {
        const t = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - t, 3);
        lifeVal.textContent = (target * eased).toFixed(1);
        if (t < 1) requestAnimationFrame(tick);
        else lifeVal.textContent = target.toFixed(1);
      };
      requestAnimationFrame(tick);

      lifeSub.textContent = `± ${data.life_std_h.toFixed(1)} h（标准差）`;
      predLow.textContent  = `${data.life_low.toFixed(1)} h`;
      predHigh.textContent = `${data.life_high.toFixed(1)} h`;

      // Confidence bar: position within a 0–300h reference range
      const REF_MAX = 300;
      const leftPct  = Math.max(0, Math.min(100, (data.life_low  / REF_MAX) * 100));
      const rightPct = Math.max(0, Math.min(100, (data.life_high / REF_MAX) * 100));
      const midPct   = Math.max(0, Math.min(100, (data.life_h    / REF_MAX) * 100));
      confBar.style.left  = leftPct + '%';
      confBar.style.width = (rightPct - leftPct) + '%';
      confMark.style.left = midPct + '%';

      result.style.display = 'flex';
    } catch (err) {
      loading.style.display = 'none';
      // Show error inline
      const errBox = document.createElement('div');
      errBox.style.cssText = 'border:1px solid var(--clr-red);padding:14px;font-size:0.8rem;color:var(--clr-red);';
      errBox.textContent = '错误: ' + err.message;
      modalBody.querySelector('.pred-layout').appendChild(errBox);
    } finally { runBtn.disabled = false; }
  });

  resetBtn.addEventListener('click', () => {
    inputs.forEach(inp => { inp.value = ''; inp.classList.remove('error'); });
    result.style.display = 'none';
    loading.style.display = 'none';
  });
}

// =============================================
//  HERO EFFECTS
// =============================================
document.addEventListener('DOMContentLoaded', () => {
  const label = document.querySelector('.hero-label');
  if (label) {
    const orig = label.textContent; label.textContent = '';
    let i = 0;
    const type = () => { if (i < orig.length) { label.textContent += orig[i++]; setTimeout(type, 38); } };
    setTimeout(type, 300);
  }
  const title = document.querySelector('.hero-title');
  if (title) {
    const cursor = document.createElement('span');
    cursor.textContent = '_';
    cursor.style.cssText = 'display:inline-block;color:#00eeff;animation:blink 1s step-end infinite;';
    if (!document.getElementById('blink-style')) {
      const s = document.createElement('style');
      s.id = 'blink-style';
      s.textContent = '@keyframes blink{50%{opacity:0}}';
      document.head.appendChild(s);
    }
    title.appendChild(cursor);
  }
  document.querySelectorAll('.module').forEach(mod => {
    const t = mod.querySelector('.module-title');
    mod.addEventListener('mouseenter', () => {
      if (!t) return;
      t.style.textShadow = '0 0 8px #00eeff,2px 0 #ff003c,-2px 0 #00eeff';
      setTimeout(() => { t.style.textShadow = ''; }, 150);
    });
  });
});

// =============================================
//  MODULE 4 additions to MODULE_DATA & router
// =============================================

// Patch MODULE_DATA to include mod 4
MODULE_DATA['4'] = { index: '04', label: 'MODULE 04', title: '一键寿命评估' };

// Override openModal to handle module 4
const _origOpenModal = openModal;
// Re-define to catch module 4
document.querySelectorAll('.module').forEach(mod => {
  // remove existing listeners by cloning
  const clone = mod.cloneNode(true);
  mod.parentNode.replaceChild(clone, mod);
  clone.addEventListener('click', () => {
    const id = clone.dataset.module;
    const data = MODULE_DATA[id]; if (!data) return;
    modalIndex.textContent = data.index;
    modalLabel.textContent = data.label;
    modalTitle.textContent = data.title;
    if      (id === '1') buildSegmentationUI();
    else if (id === '2') buildExtractionUI();
    else if (id === '3') buildPredictionUI();
    else if (id === '4') buildFullPipelineUI();
    backdrop.setAttribute('aria-hidden','false');
    backdrop.classList.add('is-open');
    document.body.style.overflow = 'hidden';
    lastFocused = document.activeElement;
    setTimeout(() => closeBtn.focus(), 50);
  });
  clone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); clone.click(); }
  });
  // glitch effect
  const t = clone.querySelector('.module-title');
  clone.addEventListener('mouseenter', () => {
    if (!t) return;
    t.style.textShadow = '0 0 8px #00eeff,2px 0 #ff003c,-2px 0 #00eeff';
    setTimeout(() => { t.style.textShadow = ''; }, 150);
  });
});

// =============================================
//  MODULE 4 — Full Pipeline UI
// =============================================
function buildFullPipelineUI() {
  const tpl = document.getElementById('tpl-full-pipeline');
  modalBody.innerHTML = '';
  modalBody.appendChild(tpl.content.cloneNode(true));

  const fullDropZone   = modalBody.querySelector('#fullDropZone');
  const fullFileInput  = modalBody.querySelector('#fullFileInput');
  const fullFileList   = modalBody.querySelector('#fullFileList');
  const fullTemp       = modalBody.querySelector('#fullTemp');
  const fullStress     = modalBody.querySelector('#fullStress');
  const fullMag        = modalBody.querySelector('#fullMag');
  const fullAngle      = modalBody.querySelector('#fullAngle');
  const fullRunBtn     = modalBody.querySelector('#fullRunBtn');
  const fullLoading    = modalBody.querySelector('#fullLoading');
  const fullLoaderText = modalBody.querySelector('#fullLoaderText');
  const fullResultWrap = modalBody.querySelector('#fullResultWrap');
  const fullResultCard = modalBody.querySelector('#fullResultCard');
  const fullFilename   = modalBody.querySelector('#fullFilename');
  const fullLifeVal    = modalBody.querySelector('#fullLifeVal');
  const fullLifeSub    = modalBody.querySelector('#fullLifeSub');
  const fullLow        = modalBody.querySelector('#fullLow');
  const fullHigh       = modalBody.querySelector('#fullHigh');
  const fullConfBar    = modalBody.querySelector('#fullConfBar');
  const fullConfMarker = modalBody.querySelector('#fullConfMarker');
  const fullNav        = modalBody.querySelector('#fullNav');
  const fullNavLabel   = modalBody.querySelector('#fullNavLabel');
  const fullPrev       = modalBody.querySelector('#fullPrev');
  const fullNext       = modalBody.querySelector('#fullNext');
  const fullDownloadBtn= modalBody.querySelector('#fullDownloadBtn');
  const fullResetBtn   = modalBody.querySelector('#fullResetBtn');

  let selectedFiles = [], allResults = [], currentIdx = 0;

  // ── File handling ─────────────────────────
  function renderFileList() {
    fullFileList.style.display = selectedFiles.length ? 'block' : 'none';
    fullFileList.innerHTML = selectedFiles.map(f =>
      `<div class="ext-file-item"><span class="fname">${f.name}</span>
       <span style="color:var(--clr-muted)">${(f.size/1024).toFixed(0)} KB</span></div>`
    ).join('');
    updateRunBtn();
  }

  function updateRunBtn() {
    fullRunBtn.disabled = !(selectedFiles.length && fullTemp.value && fullStress.value);
  }

  function loadFiles(files) {
    selectedFiles = Array.from(files);
    fullDropZone.style.display = 'none';
    renderFileList();
    fullResultWrap.style.display = 'none';
    allResults = []; currentIdx = 0;
  }

  fullDropZone.addEventListener('click', () => fullFileInput.click());
  fullFileInput.addEventListener('change', e => { if (e.target.files.length) loadFiles(e.target.files); });
  fullDropZone.addEventListener('dragover',  e => { e.preventDefault(); fullDropZone.classList.add('drag-over'); });
  fullDropZone.addEventListener('dragleave', () => fullDropZone.classList.remove('drag-over'));
  fullDropZone.addEventListener('drop', e => {
    e.preventDefault(); fullDropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) loadFiles(e.dataTransfer.files);
  });
  fullTemp.addEventListener('input', updateRunBtn);
  fullStress.addEventListener('input', updateRunBtn);

  // ── Run ───────────────────────────────────
  fullRunBtn.addEventListener('click', async () => {
    if (!selectedFiles.length || !fullTemp.value || !fullStress.value) return;

    fullRunBtn.disabled = true;
    fullResultWrap.style.display = 'none';
    fullLoading.style.display = 'flex';

    const steps = [
      [800,  '正在计算 CCA 交叉相关特征…'],
      [2000, '运行 CLD 割线分析…'],
      [3500, 'UMAP 降维 + GPR 推理…'],
      [5000, '汇总结果…'],
    ];
    const timers = steps.map(([ms, txt]) =>
      setTimeout(() => { fullLoaderText.textContent = txt; }, ms)
    );

    try {
      const fd = new FormData();
      selectedFiles.forEach(f => fd.append('images', f));
      fd.append('temp',          fullTemp.value);
      fd.append('stress',        fullStress.value);
      fd.append('magnification', fullMag.value);
      fd.append('angle',         fullAngle.value);

      const resp = await fetch(API_BASE + '/api/predict_full', { method: 'POST', body: fd });
      timers.forEach(clearTimeout);

      if (!resp.ok) { const e = await resp.json(); throw new Error(e.error || `HTTP ${resp.status}`); }
      allResults = await resp.json();
      currentIdx = 0;

      fullLoading.style.display = 'none';
      renderResult(allResults[0]);
      fullResultWrap.style.display = 'block';

      fullNav.style.display = allResults.length > 1 ? 'flex' : 'none';
      if (allResults.length > 1) updateNav();

    } catch (err) {
      timers.forEach(clearTimeout);
      fullLoading.style.display = 'none';
      const errBox = document.createElement('div');
      errBox.style.cssText = 'border:1px solid var(--clr-red);padding:14px;font-size:0.8rem;color:var(--clr-red);';
      errBox.textContent = '错误: ' + err.message;
      modalBody.querySelector('.full-layout').appendChild(errBox);
    } finally { fullRunBtn.disabled = false; }
  });

  // ── Render one result ─────────────────────
  function renderResult(r) {
    fullFilename.textContent = r.filename ? `// ${r.filename}` : '';

    // Animate life value
    const target = r.life_h;
    const start  = performance.now();
    const tick   = now => {
      const t = Math.min((now - start) / 900, 1);
      const e = 1 - Math.pow(1 - t, 3);
      fullLifeVal.textContent = (target * e).toFixed(1);
      if (t < 1) requestAnimationFrame(tick);
      else fullLifeVal.textContent = target.toFixed(1);
    };
    requestAnimationFrame(tick);

    fullLifeSub.textContent = `± ${r.life_std_h.toFixed(1)} h（标准差）`;
    fullLow.textContent  = `${r.life_low.toFixed(1)} h`;
    fullHigh.textContent = `${r.life_high.toFixed(1)} h`;

    const REF = 300;
    const lp  = Math.max(0, Math.min(100, r.life_low  / REF * 100));
    const hp  = Math.max(0, Math.min(100, r.life_high / REF * 100));
    const mp  = Math.max(0, Math.min(100, r.life_h    / REF * 100));
    fullConfBar.style.left    = lp + '%';
    fullConfBar.style.width   = (hp - lp) + '%';
    fullConfMarker.style.left = mp + '%';
  }

  // ── Navigation ────────────────────────────
  function updateNav() {
    fullNavLabel.textContent = `图像 ${currentIdx + 1} / ${allResults.length}`;
    fullPrev.disabled = currentIdx === 0;
    fullNext.disabled = currentIdx === allResults.length - 1;
  }
  fullPrev.addEventListener('click', () => {
    if (currentIdx > 0) { currentIdx--; renderResult(allResults[currentIdx]); updateNav(); }
  });
  fullNext.addEventListener('click', () => {
    if (currentIdx < allResults.length - 1) { currentIdx++; renderResult(allResults[currentIdx]); updateNav(); }
  });

  // ── Download CSV ──────────────────────────
  fullDownloadBtn.addEventListener('click', () => {
    if (!allResults.length) return;
    const hdr  = 'filename,life_h,life_std_h,life_low,life_high,umap1_norm,umap2_norm,volume_fraction';
    const rows = allResults.map(r =>
      [r.filename, r.life_h, r.life_std_h, r.life_low, r.life_high,
       r.umap1_norm, r.umap2_norm, r.volume_fraction].join(',')
    );
    const blob = new Blob([[hdr, ...rows].join('\n')], { type: 'text/csv;charset=utf-8;' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'life_prediction_results.csv';
    a.click();
  });

  // ── Reset ─────────────────────────────────
  fullResetBtn.addEventListener('click', () => {
    selectedFiles = []; allResults = []; currentIdx = 0;
    fullFileInput.value = '';
    fullFileList.style.display = 'none';
    fullDropZone.style.display = '';
    fullTemp.value = ''; fullStress.value = '';
    fullRunBtn.disabled = true;
    fullResultWrap.style.display = 'none';
    fullLoading.style.display = 'none';
  });
}
