/* ==========================================================================
   FloraGuard AI - Interactive Application Controller & Zoom Engine
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initUploadDropzone();
    initPlotFilter();
    initImageZoomModal();
    loadLeaderboard();
    loadCheckpoints();
});

let selectedFile = null;

/* --------------------------------------------------------------------------
   1. Tab Navigation
   -------------------------------------------------------------------------- */
function initTabs() {
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');

            navButtons.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const activePane = document.getElementById(targetTab);
            if (activePane) {
                activePane.classList.add('active');
            }

            if (targetTab === 'tab-leaderboard') {
                loadLeaderboard();
            } else if (targetTab === 'tab-checkpoints') {
                loadCheckpoints();
            } else if (targetTab === 'tab-plantdoc') {
                loadPlantDocResults();
            }
        });
    });
}

/* --------------------------------------------------------------------------
   2. File Upload & Dropzone Controller
   -------------------------------------------------------------------------- */
function initUploadDropzone() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const btnDiagnose = document.getElementById('btn-diagnose');
    const btnReset = document.getElementById('btn-reset');
    const dropzoneContent = dropzone.querySelector('.dropzone-content');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.remove('dragover'), false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file.type.match('image.*')) {
            alert('Please select a valid image file (JPEG or PNG).');
            return;
        }

        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove('hidden');
            dropzoneContent.classList.add('hidden');
            btnDiagnose.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    btnReset.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        imagePreview.classList.add('hidden');
        dropzoneContent.classList.remove('hidden');
        btnDiagnose.disabled = true;
        
        // Reset Result UI
        document.getElementById('result-empty').classList.remove('hidden');
        document.getElementById('result-loading').classList.add('hidden');
        document.getElementById('result-content').classList.add('hidden');
        document.getElementById('latency-badge').classList.add('hidden');
    });

    btnDiagnose.addEventListener('click', runDiagnosis);
}

/* --------------------------------------------------------------------------
   3. Run Disease Inference API Call
   -------------------------------------------------------------------------- */
async function runDiagnosis() {
    if (!selectedFile) return;

    const modelSelect = document.getElementById('model-select');
    const selectedModel = modelSelect.value;

    const resultEmpty = document.getElementById('result-empty');
    const resultLoading = document.getElementById('result-loading');
    const resultContent = document.getElementById('result-content');
    const latencyBadge = document.getElementById('latency-badge');

    resultEmpty.classList.add('hidden');
    resultContent.classList.add('hidden');
    resultLoading.classList.remove('hidden');

    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('model', selectedModel);

    try {
        const response = await fetch('/api/classify', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server returned HTTP status ${response.status}`);
        }

        const data = await response.json();
        renderDiagnosisResults(data);
    } catch (err) {
        alert('Diagnosis failed: ' + err.message);
        resultLoading.classList.add('hidden');
        resultEmpty.classList.remove('hidden');
    }
}

function renderDiagnosisResults(data) {
    const resultLoading = document.getElementById('result-loading');
    const resultContent = document.getElementById('result-content');
    const latencyBadge = document.getElementById('latency-badge');

    resultLoading.classList.add('hidden');
    resultContent.classList.remove('hidden');

    // Latency
    latencyBadge.textContent = `${data.latency_ms} ms`;
    latencyBadge.classList.remove('hidden');

    // Status Banner
    const statusBanner = document.getElementById('status-banner');
    const statusIcon = document.getElementById('status-icon');
    const statusTag = document.getElementById('status-tag');
    const fullClassName = document.getElementById('full-class-name');
    const confidenceVal = document.getElementById('confidence-val');

    fullClassName.textContent = data.predicted_class;
    confidenceVal.textContent = `${data.confidence}%`;

    if (data.is_healthy) {
        statusBanner.className = 'status-banner healthy';
        statusIcon.textContent = '✅';
        statusTag.textContent = 'Healthy Plant';
    } else {
        statusBanner.className = 'status-banner diseased';
        statusIcon.textContent = '⚠️';
        statusTag.textContent = 'Disease Detected';
    }

    // Summary Details
    document.getElementById('res-crop').textContent = data.crop;
    document.getElementById('res-disease').textContent = data.disease;
    document.getElementById('res-model').textContent = data.model_used;

    // Probability Bars
    const barsContainer = document.getElementById('probability-bars');
    barsContainer.innerHTML = '';

    data.top_k.forEach(item => {
        const row = document.createElement('div');
        row.className = 'prob-row';
        row.innerHTML = `
            <div class="prob-header">
                <span>${item.class_name}</span>
                <strong>${item.confidence}%</strong>
            </div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width: ${item.confidence}%;"></div>
            </div>
        `;
        barsContainer.appendChild(row);
    });
}

/* --------------------------------------------------------------------------
   4. Method Leaderboard API Loader
   -------------------------------------------------------------------------- */
async function loadLeaderboard() {
    try {
        const res = await fetch('/api/experiments');
        const data = await res.json();
        const experiments = data.experiments || [];

        const cardsGrid = document.getElementById('leaderboard-cards');
        const tbody = document.querySelector('#leaderboard-table tbody');

        cardsGrid.innerHTML = '';
        tbody.innerHTML = '';

        if (experiments.length === 0) {
            tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;">No cross-method ranking data available yet. Run rank_experiments.py or complete training experiments.</td></tr>';
            return;
        }

        experiments.forEach((exp, idx) => {
            // Card
            if (idx === 0) {
                const card = document.createElement('div');
                card.className = 'leader-card glass-card';
                card.innerHTML = `
                    <div class="rank-badge">#1</div>
                    <div class="leader-info">
                        <h4>Winning Approach: ${exp.method}</h4>
                        <div class="leader-metrics">
                            <span>Accuracy: <strong>${(exp.test_accuracy * 100).toFixed(2)}%</strong></span>
                            <span>F1: <strong>${(exp.f1_macro * 100).toFixed(2)}%</strong></span>
                            <span>GPU: <strong>${exp.peak_gpu_memory_gb ? exp.peak_gpu_memory_gb.toFixed(2) : 0} GB</strong></span>
                        </div>
                    </div>
                `;
                cardsGrid.appendChild(card);
            }

            // Table Row
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>#${exp.overall_rank || (idx + 1)}</strong></td>
                <td><span class="badge badge-tech">${exp.method}</span></td>
                <td>${exp.best_checkpoint || '-'}</td>
                <td><strong>${exp.test_accuracy ? (exp.test_accuracy * 100).toFixed(2) + '%' : '-'}</strong></td>
                <td>${exp.f1_macro ? (exp.f1_macro * 100).toFixed(2) + '%' : '-'}</td>
                <td>${exp.binary_accuracy ? (exp.binary_accuracy * 100).toFixed(2) + '%' : '-'}</td>
                <td>${exp.trainable_parameters ? exp.trainable_parameters.toLocaleString() : '-'}</td>
                <td>${exp.peak_gpu_memory_gb ? exp.peak_gpu_memory_gb.toFixed(2) + ' GB' : '-'}</td>
                <td>${exp.checkpoint_size_mb ? exp.checkpoint_size_mb.toFixed(1) + ' MB' : '-'}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        console.error("Error loading leaderboard:", err);
    }
}

/* --------------------------------------------------------------------------
   5. Checkpoint Explorer API Loader
   -------------------------------------------------------------------------- */
async function loadCheckpoints() {
    try {
        const res = await fetch('/api/checkpoints');
        const data = await res.json();
        const checkpoints = data.checkpoints || [];

        const tbody = document.querySelector('#checkpoints-table tbody');
        tbody.innerHTML = '';

        if (checkpoints.length === 0) {
            tbody.innerHTML = '<tr><td colspan="10" style="text-align:center;">No canonical checkpoint ranking files found in experiments/results/eval/. Expected 9 checkpoints: best, last, latest per method (LoRA, QLoRA, Q/K LoRA).</td></tr>';
            return;
        }

        const normalized = checkpoints.filter(ck => ck && ck.checkpoint).sort((a, b) => {
            const aName = (a.checkpoint || '').toLowerCase();
            const bName = (b.checkpoint || '').toLowerCase();
            if (aName.includes('best') && !bName.includes('best')) return -1;
            if (!aName.includes('best') && bName.includes('best')) return 1;
            return aName.localeCompare(bName);
        });

        normalized.forEach(ck => {
            const tr = document.createElement('tr');
            const checkpointName = ck.checkpoint ? ck.checkpoint.split('\\').pop().split('/').pop() : '-';
            const methodName = ck.method || 'UNKNOWN';
            tr.innerHTML = `
                <td><span class="badge badge-tech">${methodName}</span></td>
                <td>${checkpointName}</td>
                <td><strong>${ck.accuracy !== null && ck.accuracy !== undefined ? (ck.accuracy * 100).toFixed(2) + '%' : '-'}</strong></td>
                <td>${ck.f1_macro !== null && ck.f1_macro !== undefined ? (ck.f1_macro * 100).toFixed(2) + '%' : '-'}</td>
                <td>${ck.binary_accuracy !== null && ck.binary_accuracy !== undefined ? (ck.binary_accuracy * 100).toFixed(2) + '%' : '-'}</td>
                <td>${ck.binary_f1 !== null && ck.binary_f1 !== undefined ? (ck.binary_f1 * 100).toFixed(2) + '%' : '-'}</td>
                <td>${ck.binary_roc_auc !== null && ck.binary_roc_auc !== undefined ? ck.binary_roc_auc.toFixed(4) : '-'}</td>
                <td>${ck.both_correct_pct !== null && ck.both_correct_pct !== undefined ? ck.both_correct_pct.toFixed(2) + '%' : '-'}</td>
                <td>${ck.name_only_correct_pct !== null && ck.name_only_correct_pct !== undefined ? ck.name_only_correct_pct.toFixed(2) + '%' : '-'}</td>
                <td>${ck.size_mb !== null && ck.size_mb !== undefined ? ck.size_mb + ' MB' : '-'}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        console.error("Error loading checkpoints:", err);
    }
}

/* --------------------------------------------------------------------------
   6. Plot Filter Controller (LoRA vs QLoRA)
   -------------------------------------------------------------------------- */
function initPlotFilter() {
    const filterSelect = document.getElementById('plot-method-select');
    if (!filterSelect) return;

    const resolvePlotPath = async (method, suffix) => {
        const candidates = [
            `${method}_20260722_164943_${suffix}.png`,
            `${method}_${suffix}.png`
        ];

        const checkExists = async (name) => {
            try {
                const res = await fetch(`/plots/${name}`, { method: 'HEAD' });
                return res.ok ? name : null;
            } catch {
                return null;
            }
        };

        for (const c of candidates) {
            const found = await checkExists(c);
            if (found) return `/plots/${found}`;
        }

        return `/plots/${candidates[0]}`;
    };

    const updatePlotImages = async () => {
        const method = filterSelect.value.toLowerCase();
        const displayMethod = method.toUpperCase();

        const imgCurves = document.getElementById('img-curves');
        const imgCM = document.getElementById('img-cm');
        const imgMetrics = document.getElementById('img-metrics');

        const headingCurves = document.getElementById('heading-curves');
        const headingCM = document.getElementById('heading-cm');
        const headingMetrics = document.getElementById('heading-metrics');

        headingCurves.textContent = `📈 Training & Validation Curves (${displayMethod})`;
        headingCM.textContent = `📊 Confusion Matrix Heatmap (${displayMethod})`;
        headingMetrics.textContent = `🎯 Per-Class Precision, Recall & F1 Bar Chart (${displayMethod})`;

        const curvePath = await resolvePlotPath(method, 'training_curves');
        const cmPath = await resolvePlotPath(method, 'confusion_matrix');
        const metricsPath = await resolvePlotPath(method, 'class_metrics');

        imgCurves.src = curvePath;
        imgCurves.alt = `${displayMethod} Training Curves`;

        imgCM.src = cmPath;
        imgCM.alt = `${displayMethod} Confusion Matrix`;

        imgMetrics.src = metricsPath;
        imgMetrics.alt = `${displayMethod} Class Metrics`;
    };

    filterSelect.addEventListener('change', () => updatePlotImages());
    updatePlotImages();
}

/* --------------------------------------------------------------------------
   7. Chrome-Style Fullscreen Interactive Image Zoom Viewer
   -------------------------------------------------------------------------- */
function initImageZoomModal() {
    const modal = document.getElementById('image-viewer-modal');
    const modalImg = document.getElementById('modal-zoom-img');
    const modalTitle = document.getElementById('modal-image-title');
    const viewport = document.getElementById('modal-viewport');
    const zoomText = document.getElementById('zoom-level-text');

    const btnZoomIn = document.getElementById('btn-zoom-in');
    const btnZoomOut = document.getElementById('btn-zoom-out');
    const btnZoomReset = document.getElementById('btn-zoom-reset');
    const btnModalClose = document.getElementById('btn-modal-close');

    let scale = 1;
    let translateX = 0;
    let translateY = 0;
    let isDragging = false;
    let startX = 0;
    let startY = 0;

    // Use Delegated Click listener so any image with .clickable-zoom triggers modal
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('clickable-zoom')) {
            const img = e.target;
            const src = img.getAttribute('src');
            const title = img.getAttribute('alt') || 'Plot Preview';
            
            modalImg.src = src;
            modalTitle.textContent = title;
            resetZoom();
            modal.classList.remove('hidden');
            document.body.style.overflow = 'hidden';
        }
    });

    function updateTransform() {
        modalImg.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
        zoomText.textContent = `${Math.round(scale * 100)}%`;
        if (scale > 1) {
            modalImg.style.cursor = 'grab';
        } else {
            modalImg.style.cursor = 'zoom-in';
        }
    }

    function resetZoom() {
        scale = 1;
        translateX = 0;
        translateY = 0;
        updateTransform();
    }

    function zoom(deltaScale) {
        scale = Math.min(Math.max(0.5, scale + deltaScale), 5.0);
        if (scale <= 1) {
            translateX = 0;
            translateY = 0;
        }
        updateTransform();
    }

    // Toggle zoom on double-click (Chrome style)
    modalImg.addEventListener('dblclick', () => {
        if (scale === 1) {
            scale = 2.5;
        } else {
            resetZoom();
            return;
        }
        updateTransform();
    });

    // Zoom Controls
    btnZoomIn.addEventListener('click', () => zoom(0.25));
    btnZoomOut.addEventListener('click', () => zoom(-0.25));
    btnZoomReset.addEventListener('click', resetZoom);
    btnModalClose.addEventListener('click', closeModal);

    function closeModal() {
        modal.classList.add('hidden');
        document.body.style.overflow = '';
    }

    // Close on backdrop click
    viewport.addEventListener('click', (e) => {
        if (e.target === viewport) {
            closeModal();
        }
    });

    // Close on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
            closeModal();
        }
    });

    // Mouse Wheel Zoom
    viewport.addEventListener('wheel', (e) => {
        e.preventDefault();
        const delta = e.deltaY < 0 ? 0.2 : -0.2;
        zoom(delta);
    }, { passive: false });

    // Drag to Pan
    viewport.addEventListener('mousedown', (e) => {
        if (scale > 1) {
            isDragging = true;
            startX = e.clientX - translateX;
            startY = e.clientY - translateY;
        }
    });

    window.addEventListener('mousemove', (e) => {
        if (isDragging) {
            translateX = e.clientX - startX;
            translateY = e.clientY - startY;
            updateTransform();
        }
    });

    window.addEventListener('mouseup', () => {
        isDragging = false;
    });
}

/* --------------------------------------------------------------------------
   8. PlantDoc Real-World Field Benchmark Loader
   -------------------------------------------------------------------------- */
async function loadPlantDocResults() {
    try {
        const res = await fetch('/api/plantdoc');
        const data = await res.json();
        const results = data.field_results || [];

        const tbody = document.querySelector('#plantdoc-table tbody');
        tbody.innerHTML = '';

        if (results.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;">No real-world field benchmark data found. Run launcher_plantdoc.py option 2 to evaluate.</td></tr>';
            return;
        }

        results.forEach(r => {
            const tr = document.createElement('tr');
            const accuracy = r.accuracy !== undefined && r.accuracy !== null ? r.accuracy.toFixed(2) + '%' : '-';
            const sampleCount = r.sample_count !== undefined && r.sample_count !== null ? r.sample_count : '-';
            tr.innerHTML = `
                <td><span class="badge badge-tech">${r.method}</span></td>
                <td>${r.checkpoint || '-'}</td>
                <td>${r.split || '-'}</td>
                <td><strong>${accuracy}</strong></td>
                <td>${sampleCount}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        console.error("Error loading PlantDoc results:", err);
    }
}

