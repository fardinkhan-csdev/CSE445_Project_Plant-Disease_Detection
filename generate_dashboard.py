"""
generate_dashboard.py
Run this after training/evaluation to regenerate experiments/results/dashboard.html
with all CSV data and plot images embedded directly as self-contained HTML.

Usage:
    python generate_dashboard.py
"""
import os
import csv
import glob
import json
import base64

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'experiments', 'results')
PLOTS_DIR    = os.path.join(RESULTS_DIR, 'plots')
EVAL_DIR     = os.path.join(RESULTS_DIR, 'eval')
OUT_HTML     = os.path.join(RESULTS_DIR, 'dashboard.html')


# ── Helpers ───────────────────────────────────────────────────────────────────

METHOD_ORDER = ['lora', 'qlora', 'qklora']
METHOD_LABELS = {'lora': 'LoRA', 'qlora': 'QLoRA', 'qklora': 'Q/K LoRA'}


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def img_to_b64(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def method_from_checkpoint(checkpoint_path):
    name = os.path.basename(checkpoint_path.replace('\\', '/')).replace('.pth', '')
    for method in ('qklora', 'qlora', 'lora'):
        if name.startswith(method + '_'):
            return method
    return name.split('_')[0] if '_' in name else name


def checkpoint_path_on_disk(checkpoint_path):
    if not checkpoint_path:
        return None
    if os.path.isabs(checkpoint_path):
        return checkpoint_path
    return os.path.join(PROJECT_ROOT, checkpoint_path)


def enrich_checkpoints(rows):
    enriched = []
    for row in rows:
        item = dict(row)
        item['method'] = method_from_checkpoint(item.get('checkpoint', ''))
        if not str(item.get('rank', '')).strip():
            item['rank'] = ''
        if not str(item.get('size_mb', '')).strip():
            path = checkpoint_path_on_disk(item.get('checkpoint', ''))
            item['size_mb'] = round(os.path.getsize(path) / (1024 * 1024), 2) if path and os.path.exists(path) else ''
        enriched.append(item)

    by_method = {}
    for item in enriched:
        by_method.setdefault(item['method'], []).append(item)

    for group in by_method.values():
        if all(not str(item.get('rank', '')).strip() for item in group):
            ranked = sorted(group, key=lambda row: float(row.get('accuracy') or 0), reverse=True)
            for index, item in enumerate(ranked, start=1):
                item['rank'] = str(index)

    enriched.sort(key=lambda row: (
        METHOD_ORDER.index(row['method']) if row['method'] in METHOD_ORDER else 99,
        int(row.get('rank') or 999),
    ))
    return enriched


def build_method_comparison(experiments, checkpoints):
    cross_path = os.path.join(EVAL_DIR, 'cross_method_ranking.csv')
    rows = read_csv(cross_path)
    if rows:
        return rows
    if not experiments:
        return []

    best_by_method = {}
    for ck in checkpoints:
        method = ck['method']
        rank = int(ck.get('rank') or 999)
        accuracy = float(ck.get('accuracy') or 0)
        current = best_by_method.get(method)
        if not current or rank < int(current.get('rank') or 999):
            best_by_method[method] = ck
        elif rank == int(current.get('rank') or 999) and accuracy > float(current.get('accuracy') or 0):
            best_by_method[method] = ck

    merged = []
    for exp in experiments:
        method = exp.get('experiment', '')
        ck = best_by_method.get(method, {})
        merged.append({
            'rank': '',
            'method': method,
            'experiment': method,
            'checkpoint': ck.get('checkpoint', ''),
            'size_mb': ck.get('size_mb', ''),
            'trainable_parameters': exp.get('trainable_parameters', ''),
            'training_time': exp.get('training_time', ''),
            'peak_gpu_memory': exp.get('peak_gpu_memory', ''),
            'test_accuracy': exp.get('test_accuracy', ''),
            'test_f1_macro': exp.get('test_f1_macro', ''),
            'test_precision_macro': exp.get('test_precision_macro', ''),
            'test_recall_macro': exp.get('test_recall_macro', ''),
            'binary_f1': ck.get('binary_f1', ''),
            'both_correct_pct': ck.get('both_correct_pct', ''),
        })

    merged.sort(key=lambda row: (
        -float(row.get('test_accuracy') or 0),
        -float(row.get('test_f1_macro') or 0),
        -float(row.get('binary_f1') or 0),
        -float(row.get('both_correct_pct') or 0),
        float(row.get('peak_gpu_memory') or 0),
        int(row.get('trainable_parameters') or 0),
    ))
    for index, row in enumerate(merged, start=1):
        row['rank'] = str(index)
    return merged


# ── Collect data ──────────────────────────────────────────────────────────────

experiment_summary_files = sorted(
    glob.glob(os.path.join(RESULTS_DIR, 'experiment_results*.csv'))
)
experiments = []
for path in experiment_summary_files:
    experiments.extend(read_csv(path))

checkpoints = []
for fname in sorted(os.listdir(EVAL_DIR)) if os.path.exists(EVAL_DIR) else []:
    if fname.endswith('_checkpoint_ranking.csv'):
        rows = read_csv(os.path.join(EVAL_DIR, fname))
        checkpoints.extend(rows)
checkpoints = enrich_checkpoints(checkpoints)
method_comparison = build_method_comparison(experiments, checkpoints)

PLOT_DEFS = [
    ('lora',   'training_curves',   'LoRA Training Curves',        '📈'),
    ('lora',   'confusion_matrix',  'LoRA Confusion Matrix',        '🔲'),
    ('lora',   'class_metrics',     'LoRA Class-wise Metrics',      '📊'),
    ('qlora',  'training_curves',   'QLoRA Training Curves',        '📈'),
    ('qlora',  'confusion_matrix',  'QLoRA Confusion Matrix',       '🔲'),
    ('qlora',  'class_metrics',     'QLoRA Class-wise Metrics',     '📊'),
    ('qklora', 'training_curves',   'Q/K LoRA Training Curves',     '📈'),
    ('qklora', 'confusion_matrix',  'Q/K LoRA Confusion Matrix',    '🔲'),
    ('qklora', 'class_metrics',     'Q/K LoRA Class-wise Metrics',  '📊'),
]

def latest_plot_path(model: str, kind: str):
    glob_pattern = os.path.join(PLOTS_DIR, f'{model}_{kind}*.png')
    matches = glob.glob(glob_pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)

plots = []
for model, kind, label, icon in PLOT_DEFS:
    path = latest_plot_path(model, kind)
    b64 = img_to_b64(path)
    plots.append({
        'label': label,
        'icon': icon,
        'method': model,
        'kind': kind,
        'src': f'data:image/png;base64,{b64}' if b64 else None,
        'filename': os.path.basename(path) if path else None,
    })

data_js = f"""
const EXPERIMENTS = {json.dumps(experiments)};
const CHECKPOINTS  = {json.dumps(checkpoints)};
const PLOTS        = {json.dumps(plots)};
const METHOD_COMPARISON = {json.dumps(method_comparison)};
const METHOD_LABELS = {json.dumps(METHOD_LABELS)};
"""

print(f"Loaded  {len(experiments)} experiment row(s)")
print(f"Loaded  {len(checkpoints)} checkpoint row(s)")
print(f"Loaded  {len(method_comparison)} method comparison row(s)")
print(f"Loaded  {sum(1 for p in plots if p['src'])} plot image(s) ({len(plots)} slots)")


# ── HTML Template ─────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>Leaf Disease Classification — Results Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet"/>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg:#0a0e1a; --bg2:#0f1629; --bg3:#151e35;
      --glass:rgba(255,255,255,0.04); --glass-border:rgba(255,255,255,0.08);
      --accent1:#4ade80; --accent2:#60a5fa; --accent3:#f472b6; --accent4:#fb923c;
      --text:#e2e8f0; --text-muted:#64748b; --text-dim:#94a3b8;
      --radius:16px; --radius-sm:10px;
    }
    *{margin:0;padding:0;box-sizing:border-box;}
    body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden;}
    body::before{
      content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
      background:radial-gradient(ellipse at 20% 20%,rgba(74,222,128,.06),transparent 50%),
                 radial-gradient(ellipse at 80% 80%,rgba(96,165,250,.06),transparent 50%),
                 radial-gradient(ellipse at 50% 50%,rgba(244,114,182,.03),transparent 60%);
    }
    .container{max-width:1400px;margin:0 auto;padding:0 24px;position:relative;z-index:1;}

    /* Header */
    header{padding:40px 0 32px;border-bottom:1px solid var(--glass-border);margin-bottom:40px;}
    .header-inner{display:flex;align-items:center;gap:20px;flex-wrap:wrap;justify-content:space-between;}
    .logo-icon{width:52px;height:52px;background:linear-gradient(135deg,var(--accent1),var(--accent2));border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:24px;box-shadow:0 0 30px rgba(74,222,128,.3);}
    h1{font-size:24px;font-weight:700;letter-spacing:-.5px;} h1 span{color:var(--accent1);}
    .subtitle{font-size:13px;color:var(--text-muted);margin-top:2px;}
    .badge{display:inline-flex;align-items:center;gap:6px;padding:6px 14px;border-radius:999px;font-size:12px;font-weight:600;border:1px solid;}
    .badge-green{background:rgba(74,222,128,.1);border-color:rgba(74,222,128,.3);color:var(--accent1);}
    .badge-blue{background:rgba(96,165,250,.1);border-color:rgba(96,165,250,.3);color:var(--accent2);}

    /* Upload Zone */
    .upload-zone{border:2px dashed var(--glass-border);border-radius:var(--radius);padding:22px;text-align:center;margin-bottom:32px;transition:all .3s;cursor:pointer;background:var(--glass);}
    .upload-zone:hover,.upload-zone.drag-over{border-color:var(--accent1);background:rgba(74,222,128,.04);}
    .upload-zone input{display:none;}
    .upload-zone label{cursor:pointer;display:block;}
    .upload-text{font-size:13px;color:var(--text-dim);} .upload-text strong{color:var(--accent1);}
    .upload-hint{font-size:11px;color:var(--text-muted);margin-top:3px;}

    /* Loaded files */
    .loaded-files{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:32px;}
    .file-chip{display:inline-flex;align-items:center;gap:8px;padding:6px 14px;border-radius:999px;background:rgba(74,222,128,.1);border:1px solid rgba(74,222,128,.25);font-size:12px;font-weight:500;color:var(--accent1);animation:fadeInUp .3s ease;}
    .file-chip .dot{width:6px;height:6px;background:var(--accent1);border-radius:50%;animation:pulse 2s infinite;}

    /* Section titles */
    .section-title{font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--text-muted);margin-bottom:20px;display:flex;align-items:center;gap:10px;}
    .section-title::after{content:'';flex:1;height:1px;background:var(--glass-border);}

    /* Stat cards */
    .stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:40px;}
    .stat-card{background:var(--glass);border:1px solid var(--glass-border);border-radius:var(--radius);padding:22px 24px;position:relative;overflow:hidden;transition:transform .2s,box-shadow .2s;}
    .stat-card:hover{transform:translateY(-3px);box-shadow:0 12px 40px rgba(0,0,0,.3);}
    .stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
    .stat-card.green::before{background:linear-gradient(90deg,var(--accent1),transparent);}
    .stat-card.blue::before {background:linear-gradient(90deg,var(--accent2),transparent);}
    .stat-card.pink::before {background:linear-gradient(90deg,var(--accent3),transparent);}
    .stat-card.orange::before{background:linear-gradient(90deg,var(--accent4),transparent);}
    .stat-label{font-size:11px;font-weight:600;letter-spacing:.8px;text-transform:uppercase;color:var(--text-muted);margin-bottom:12px;}
    .stat-value{font-size:32px;font-weight:800;letter-spacing:-1px;line-height:1;}
    .stat-value.green{color:var(--accent1);} .stat-value.blue{color:var(--accent2);}
    .stat-value.pink{color:var(--accent3);} .stat-value.orange{color:var(--accent4);}
    .stat-sub{font-size:12px;color:var(--text-muted);margin-top:6px;}

    /* Charts */
    .charts-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:20px;margin-bottom:40px;}
    .chart-card{background:var(--glass);border:1px solid var(--glass-border);border-radius:var(--radius);padding:24px;transition:transform .2s;}
    .chart-card:hover{transform:translateY(-2px);}
    .chart-title{font-size:14px;font-weight:600;margin-bottom:20px;display:flex;align-items:center;gap:8px;}
    .chart-title .dot{width:8px;height:8px;border-radius:50%;}
    .chart-title .dot.green{background:var(--accent1);box-shadow:0 0 8px var(--accent1);}
    .chart-title .dot.blue {background:var(--accent2);box-shadow:0 0 8px var(--accent2);}
    .chart-title .dot.pink {background:var(--accent3);box-shadow:0 0 8px var(--accent3);}
    canvas{max-height:260px;}

    /* Images */
    .images-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px;margin-bottom:40px;}
    .image-card{background:var(--glass);border:1px solid var(--glass-border);border-radius:var(--radius);overflow:hidden;transition:transform .2s,box-shadow .2s;}
    .image-card:hover{transform:translateY(-3px);box-shadow:0 16px 48px rgba(0,0,0,.4);}
    .image-card-header{padding:16px 20px 12px;border-bottom:1px solid var(--glass-border);display:flex;align-items:center;gap:10px;}
    .image-card-header span{font-size:13px;font-weight:600;}
    .image-card img{width:100%;display:block;}

    /* Tables */
    .table-card{background:var(--glass);border:1px solid var(--glass-border);border-radius:var(--radius);overflow:hidden;margin-bottom:40px;}
    .table-header{padding:20px 24px;border-bottom:1px solid var(--glass-border);display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;}
    .table-header h3{font-size:14px;font-weight:600;}
    .search-input{background:rgba(255,255,255,.05);border:1px solid var(--glass-border);border-radius:8px;padding:8px 14px;font-size:13px;color:var(--text);font-family:'Inter',sans-serif;outline:none;width:220px;transition:border-color .2s;}
    .search-input:focus{border-color:var(--accent1);}
    .search-input::placeholder{color:var(--text-muted);}
    .table-wrap{overflow-x:auto;}
    table{width:100%;border-collapse:collapse;font-size:13px;}
    thead tr{background:rgba(255,255,255,.03);}
    th{padding:14px 16px;text-align:left;font-size:11px;font-weight:700;letter-spacing:.8px;text-transform:uppercase;color:var(--text-muted);white-space:nowrap;cursor:pointer;user-select:none;}
    th:hover{color:var(--text);} th.sorted{color:var(--accent1);}
    td{padding:13px 16px;border-top:1px solid var(--glass-border);white-space:nowrap;color:var(--text-dim);}
    tbody tr{transition:background .15s;}
    tbody tr:hover td{background:rgba(255,255,255,.03);color:var(--text);}
    .td-name{font-weight:500;color:var(--text);font-family:'Menlo',monospace;font-size:12px;}
    .metric-pill{display:inline-flex;align-items:center;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600;}
    .pill-green{background:rgba(74,222,128,.12);color:var(--accent1);}
    .pill-blue {background:rgba(96,165,250,.12);color:var(--accent2);}
    .pill-pink {background:rgba(244,114,182,.12);color:var(--accent3);}
    .pill-rank {background:rgba(251,146,60,.12);color:var(--accent4);}
    .progress-bar{width:80px;height:6px;background:rgba(255,255,255,.06);border-radius:999px;overflow:hidden;display:inline-block;vertical-align:middle;margin-left:8px;}
    .progress-fill{height:100%;border-radius:999px;}
    .fill-green{background:linear-gradient(90deg,var(--accent1),#86efac);}
    .fill-blue {background:linear-gradient(90deg,var(--accent2),#93c5fd);}

    /* Tabs */
    .tabs{display:flex;gap:4px;margin-bottom:24px;background:var(--glass);border:1px solid var(--glass-border);border-radius:var(--radius-sm);padding:4px;width:fit-content;}
    .tab-btn{padding:8px 18px;border-radius:8px;border:none;cursor:pointer;font-size:13px;font-weight:500;font-family:'Inter',sans-serif;color:var(--text-muted);background:transparent;transition:all .2s;}
    .tab-btn.active{background:rgba(74,222,128,.15);color:var(--accent1);}
    .tab-btn:hover:not(.active){color:var(--text);}
    .tab-pane{display:none;} .tab-pane.active{display:block;}
    .filter-row{display:flex;align-items:center;gap:12px;margin-bottom:20px;flex-wrap:wrap;}
    .filter-select{background:rgba(255,255,255,.05);border:1px solid var(--glass-border);border-radius:8px;padding:8px 14px;font-size:13px;color:var(--text);font-family:'Inter',sans-serif;outline:none;}
    .filter-select:focus{border-color:var(--accent1);}
    .method-group{margin-bottom:36px;}
    .method-group-title{font-size:13px;font-weight:600;color:var(--text-dim);margin-bottom:14px;text-transform:uppercase;letter-spacing:1px;}
    .plot-placeholder{min-height:220px;display:flex;align-items:center;justify-content:center;padding:32px;color:var(--text-muted);font-size:13px;background:rgba(255,255,255,.02);}
    .winner-banner{display:inline-flex;align-items:center;gap:8px;padding:8px 14px;border-radius:999px;background:rgba(74,222,128,.1);border:1px solid rgba(74,222,128,.25);color:var(--accent1);font-size:12px;font-weight:600;margin-bottom:16px;}
    .link-muted{color:var(--accent2);text-decoration:none;font-size:12px;}
    .link-muted:hover{text-decoration:underline;}

    /* Empty state */
    .empty-state{text-align:center;padding:48px 20px;color:var(--text-muted);}
    .empty-state .icon{font-size:40px;margin-bottom:12px;}
    .empty-state p{font-size:14px;}

    footer{padding:40px 0;text-align:center;border-top:1px solid var(--glass-border);color:var(--text-muted);font-size:12px;margin-top:20px;}
    footer span{color:var(--accent1);}

    @keyframes fadeInUp{from{opacity:0;transform:translateY(12px);}to{opacity:1;transform:translateY(0);}}
    @keyframes pulse{0%,100%{opacity:1;}50%{opacity:.4;}}
    .fade-in{animation:fadeInUp .5s ease forwards;}
    @media(max-width:700px){.charts-grid,.images-grid{grid-template-columns:1fr;}.stats-grid{grid-template-columns:repeat(2,1fr);}}
  </style>
</head>
<body>
<div class="container">

  <header>
    <div class="header-inner">
      <div style="display:flex;align-items:center;gap:16px;">
        <div class="logo-icon">🌿</div>
        <div>
          <h1>Leaf Disease <span>Dashboard</span></h1>
          <div class="subtitle">LoRA · QLoRA · Q/K LoRA — EfficientNet-B0 Comparative Study</div>
        </div>
      </div>
      <div style="display:flex;gap:10px;flex-wrap:wrap;">
        <span class="badge badge-green">● PlantVillage Dataset</span>
        <span class="badge badge-blue">38 Classes</span>
      </div>
    </div>
  </header>

  <!-- Extra CSV Upload (optional manual override) -->
  <div class="upload-zone" id="dropZone">
    <label for="csvUpload">
      <div style="font-size:22px;margin-bottom:6px;">📂</div>
      <div class="upload-text"><strong>Drop additional CSV files here</strong> or click to browse</div>
      <div class="upload-hint">All known CSVs are already loaded below — use this to add extra results manually.</div>
    </label>
    <input type="file" id="csvUpload" accept=".csv" multiple/>
  </div>
  <div class="loaded-files" id="loadedFiles"></div>

  <!-- Tabs -->
  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('overview',this)">Overview</button>
    <button class="tab-btn" onclick="switchTab('checkpoints',this)">Checkpoint Rankings</button>
    <button class="tab-btn" onclick="switchTab('comparison',this)">Method Comparison</button>
    <button class="tab-btn" onclick="switchTab('visuals',this)">Plots &amp; Visuals</button>
  </div>

  <!-- TAB: Overview -->
  <div class="tab-pane active" id="tab-overview">
    <div class="section-title">Key Metrics</div>
    <div class="stats-grid" id="statsGrid"></div>
    <div class="section-title">Performance Charts</div>
    <div class="charts-grid">
      <div class="chart-card"><div class="chart-title"><span class="dot green"></span>Multiclass Test Metrics</div><canvas id="chartMulticlass"></canvas></div>
      <div class="chart-card"><div class="chart-title"><span class="dot blue"></span>Parameter Efficiency</div><canvas id="chartParams"></canvas></div>
    </div>
    <div class="section-title">Experiment Summary Table</div>
    <div class="table-card">
      <div class="table-header">
        <h3>All Experiments</h3>
        <input class="search-input" placeholder="🔍 Filter experiments…" oninput="filterTable('expTable',this.value)"/>
      </div>
      <div class="table-wrap">
        <table id="expTable">
          <thead><tr>
            <th onclick="sortTable('expTable',0)">Experiment</th>
            <th onclick="sortTable('expTable',1)">Trainable Params</th>
            <th onclick="sortTable('expTable',2)">Train Time</th>
            <th onclick="sortTable('expTable',3)">Peak GPU (GB)</th>
            <th onclick="sortTable('expTable',4)">Best Val Acc</th>
            <th onclick="sortTable('expTable',5)">Test Accuracy</th>
            <th onclick="sortTable('expTable',6)">Precision</th>
            <th onclick="sortTable('expTable',7)">Recall</th>
            <th onclick="sortTable('expTable',8)">F1 Macro</th>
          </tr></thead>
          <tbody id="expTableBody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- TAB: Checkpoints -->
  <div class="tab-pane" id="tab-checkpoints">
    <div class="filter-row">
      <label for="methodFilter" style="font-size:13px;color:var(--text-dim);">Method</label>
      <select id="methodFilter" class="filter-select" onchange="setMethodFilter(this.value)">
        <option value="all">All methods</option>
      </select>
    </div>
    <div class="section-title">Binary Evaluation</div>
    <div class="charts-grid">
      <div class="chart-card"><div class="chart-title"><span class="dot green"></span>Binary Accuracy / F1 / ROC AUC</div><canvas id="chartBinary"></canvas></div>
      <div class="chart-card"><div class="chart-title"><span class="dot pink"></span>Prediction Correctness Breakdown</div><canvas id="chartCorrectness"></canvas></div>
    </div>
    <div class="section-title">Checkpoint Ranking Table</div>
    <div class="table-card">
      <div class="table-header">
        <h3>All Checkpoints</h3>
        <input class="search-input" placeholder="🔍 Filter checkpoints…" oninput="filterTable('ckTable',this.value)"/>
      </div>
      <div class="table-wrap">
        <table id="ckTable">
          <thead><tr>
            <th onclick="sortTable('ckTable',0)">Rank</th>
            <th onclick="sortTable('ckTable',1)">Method</th>
            <th onclick="sortTable('ckTable',2)">Checkpoint</th>
            <th onclick="sortTable('ckTable',3)">Size (MB)</th>
            <th onclick="sortTable('ckTable',4)">Accuracy</th>
            <th onclick="sortTable('ckTable',5)">F1 Macro</th>
            <th onclick="sortTable('ckTable',6)">Binary Acc</th>
            <th onclick="sortTable('ckTable',7)">Binary F1</th>
            <th onclick="sortTable('ckTable',8)">ROC AUC</th>
            <th onclick="sortTable('ckTable',9)">Both Correct %</th>
            <th onclick="sortTable('ckTable',10)">Crop Only %</th>
            <th onclick="sortTable('ckTable',11)">Disease Only %</th>
            <th onclick="sortTable('ckTable',12)">None Correct %</th>
            <th>Confidences</th>
          </tr></thead>
          <tbody id="ckTableBody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- TAB: Method Comparison -->
  <div class="tab-pane" id="tab-comparison">
    <div id="comparisonBanner"></div>
    <div class="section-title">Cross-Method Scorecard</div>
    <div class="charts-grid">
      <div class="chart-card"><div class="chart-title"><span class="dot green"></span>Best Checkpoint per Method</div><canvas id="chartComparison"></canvas></div>
      <div class="chart-card"><div class="chart-title"><span class="dot blue"></span>Efficiency vs Accuracy</div><canvas id="chartEfficiency"></canvas></div>
    </div>
    <div class="section-title">Overall Ranking</div>
    <div class="table-card">
      <div class="table-header">
        <h3>Method Winners</h3>
        <input class="search-input" placeholder="🔍 Filter methods…" oninput="filterTable('cmpTable',this.value)"/>
      </div>
      <div class="table-wrap">
        <table id="cmpTable">
          <thead><tr>
            <th onclick="sortTable('cmpTable',0)">Rank</th>
            <th onclick="sortTable('cmpTable',1)">Method</th>
            <th onclick="sortTable('cmpTable',2)">Best Checkpoint</th>
            <th onclick="sortTable('cmpTable',3)">Size (MB)</th>
            <th onclick="sortTable('cmpTable',4)">Test Accuracy</th>
            <th onclick="sortTable('cmpTable',5)">F1 Macro</th>
            <th onclick="sortTable('cmpTable',6)">Binary F1</th>
            <th onclick="sortTable('cmpTable',7)">Both Correct %</th>
            <th onclick="sortTable('cmpTable',8)">Trainable Params</th>
            <th onclick="sortTable('cmpTable',9)">Peak GPU (GB)</th>
            <th onclick="sortTable('cmpTable',10)">Train Time</th>
          </tr></thead>
          <tbody id="cmpTableBody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- TAB: Visuals -->
  <div class="tab-pane" id="tab-visuals">
    <div class="section-title">Training Plots</div>
    <div class="images-grid" id="plotsGrid"></div>
  </div>

  <footer>Built with ❤️ for &nbsp;<span>Leaf Disease Classification</span>&nbsp; · EfficientNet-B0 PEFT Comparative Study</footer>
</div>

<script>
// ── Embedded data (auto-generated) ──────────────────────────────────────────
__DATA_JS__

// ── Chart helpers ────────────────────────────────────────────────────────────
const CI = {};
const CD = {
  plugins:{legend:{labels:{color:'#94a3b8',font:{family:'Inter',size:12},boxWidth:14}}},
  scales:{
    x:{ticks:{color:'#64748b',font:{family:'Inter',size:11}},grid:{color:'rgba(255,255,255,.04)'}},
    y:{ticks:{color:'#64748b',font:{family:'Inter',size:11}},grid:{color:'rgba(255,255,255,.04)'}}
  }
};
function dc(id){if(CI[id]){CI[id].destroy();delete CI[id];}}

function pct(v,d=2){const n=parseFloat(v);return isNaN(n)?v:(n*100).toFixed(d)+'%';}
function fmt(v,d=4){const n=parseFloat(v);return isNaN(n)?v:n.toFixed(d);}

// ── Tabs ─────────────────────────────────────────────────────────────────────
function switchTab(name,btn){
  document.querySelectorAll('.tab-pane').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  btn.classList.add('active');
}

// ── Render experiments ───────────────────────────────────────────────────────
function renderExperiments(rows){
  const grid=document.getElementById('statsGrid');
  if(!rows.length){grid.innerHTML='<div class="empty-state"><div class="icon">📊</div><p>No experiment data available</p></div>';return;}
  const r=rows[0];
  const cards=[
    {l:'Test Accuracy',   v:pct(r.test_accuracy),     s:`Best Val: ${pct(r.best_val_acc)}`,       c:'green'},
    {l:'F1 Macro',        v:fmt(r.test_f1_macro),      s:'Macro-averaged',                         c:'blue'},
    {l:'Trainable Params',v:parseInt(r.trainable_parameters||0).toLocaleString(), s:'LoRA adapter params', c:'pink'},
    {l:'Peak GPU',        v:(parseFloat(r.peak_gpu_memory||0).toFixed(2))+' GB',  s:`Train time: ${parseFloat(r.training_time||0).toFixed(0)}s`, c:'orange'},
    {l:'Precision',       v:fmt(r.test_precision_macro),s:'Macro-averaged',                         c:'blue'},
    {l:'Recall',          v:fmt(r.test_recall_macro),   s:'Macro-averaged',                         c:'green'},
  ];
  grid.innerHTML=cards.map(c=>`
    <div class="stat-card ${c.c} fade-in">
      <div class="stat-label">${c.l}</div>
      <div class="stat-value ${c.c}">${c.v}</div>
      <div class="stat-sub">${c.s}</div>
    </div>`).join('');

  const labels=rows.map(r=>(r.experiment||'').toUpperCase());
  dc('chartMulticlass');
  CI['chartMulticlass']=new Chart(document.getElementById('chartMulticlass'),{
    type:'bar',
    data:{labels,datasets:[
      {label:'Test Accuracy',data:rows.map(r=>parseFloat(r.test_accuracy)),backgroundColor:'rgba(74,222,128,.6)',borderColor:'#4ade80',borderWidth:2,borderRadius:6},
      {label:'F1 Macro',     data:rows.map(r=>parseFloat(r.test_f1_macro)), backgroundColor:'rgba(96,165,250,.6)', borderColor:'#60a5fa',borderWidth:2,borderRadius:6},
      {label:'Precision',    data:rows.map(r=>parseFloat(r.test_precision_macro)),backgroundColor:'rgba(244,114,182,.5)',borderColor:'#f472b6',borderWidth:2,borderRadius:6},
      {label:'Recall',       data:rows.map(r=>parseFloat(r.test_recall_macro)),   backgroundColor:'rgba(251,146,60,.5)', borderColor:'#fb923c',borderWidth:2,borderRadius:6},
    ]},
    options:{...CD,responsive:true,scales:{...CD.scales,y:{...CD.scales.y,min:.95,max:1.0}}}
  });
  dc('chartParams');
  CI['chartParams']=new Chart(document.getElementById('chartParams'),{
    type:'bar',
    data:{labels,datasets:[{label:'Trainable Parameters',data:rows.map(r=>parseInt(r.trainable_parameters)),backgroundColor:'rgba(96,165,250,.6)',borderColor:'#60a5fa',borderWidth:2,borderRadius:6}]},
    options:{...CD,responsive:true}
  });

  const tb=document.getElementById('expTableBody');
  tb.innerHTML=rows.map(r=>`<tr>
    <td><span class="metric-pill pill-green">${(r.experiment||'').toUpperCase()}</span></td>
    <td>${parseInt(r.trainable_parameters||0).toLocaleString()}</td>
    <td>${parseFloat(r.training_time||0).toFixed(1)}s</td>
    <td>${parseFloat(r.peak_gpu_memory||0).toFixed(2)} GB</td>
    <td><span class="metric-pill pill-blue">${pct(r.best_val_acc)}</span></td>
    <td><span class="metric-pill pill-green">${pct(r.test_accuracy)}</span><span class="progress-bar"><span class="progress-fill fill-green" style="width:${parseFloat(r.test_accuracy)*100}%"></span></span></td>
    <td>${fmt(r.test_precision_macro)}</td>
    <td>${fmt(r.test_recall_macro)}</td>
    <td>${fmt(r.test_f1_macro)}</td>
  </tr>`).join('');

  addChip('experiment_results.csv',rows.length);
}

// ── Render checkpoints ───────────────────────────────────────────────────────
function renderCheckpoints(rows){
  if(!rows.length){
    document.getElementById('ckTableBody').innerHTML='<tr><td colspan="10"><div class="empty-state"><div class="icon">📄</div><p>No checkpoint ranking data available</p></div></td></tr>';
    return;
  }
  const labels=rows.map(r=>(r.checkpoint||'').split(/[/\\]/).pop().replace('.pth',''));
  dc('chartBinary');
  CI['chartBinary']=new Chart(document.getElementById('chartBinary'),{
    type:'bar',
    data:{labels,datasets:[
      {label:'Binary Accuracy',data:rows.map(r=>parseFloat(r.binary_accuracy)),backgroundColor:'rgba(74,222,128,.6)',borderColor:'#4ade80',borderWidth:2,borderRadius:6},
      {label:'Binary F1',      data:rows.map(r=>parseFloat(r.binary_f1)),      backgroundColor:'rgba(96,165,250,.6)', borderColor:'#60a5fa',borderWidth:2,borderRadius:6},
      {label:'ROC AUC',        data:rows.map(r=>parseFloat(r.binary_roc_auc)), backgroundColor:'rgba(244,114,182,.5)',borderColor:'#f472b6',borderWidth:2,borderRadius:6},
    ]},
    options:{...CD,responsive:true,scales:{...CD.scales,y:{...CD.scales.y,min:.99,max:1.0}}}
  });
  dc('chartCorrectness');
  const cr=rows[0];
  CI['chartCorrectness']=new Chart(document.getElementById('chartCorrectness'),{
    type:'doughnut',
    data:{
      labels:['Both Correct','Crop Only Correct','Disease Only Correct','None Correct'],
      datasets:[{
        data:[parseFloat(cr.both_correct_pct||0),parseFloat(cr.name_only_correct_pct||0),parseFloat(cr.disease_only_correct_pct||0),parseFloat(cr.none_correct_pct||0)],
        backgroundColor:['rgba(74,222,128,.8)','rgba(96,165,250,.8)','rgba(251,146,60,.8)','rgba(244,114,182,.8)'],
        borderColor:['#4ade80','#60a5fa','#fb923c','#f472b6'],
        borderWidth:2,hoverOffset:8
      }]
    },
    options:{responsive:true,cutout:'68%',plugins:{legend:{position:'right',labels:{color:'#94a3b8',font:{family:'Inter',size:11},boxWidth:12,padding:14}}}}
  });

  const tb=document.getElementById('ckTableBody');
  tb.innerHTML=rows.map((r,i)=>`<tr>
    <td><span class="metric-pill pill-rank">#${i+1}</span></td>
    <td class="td-name">${(r.checkpoint||'').split(/[/\\]/).pop()}</td>
    <td><span class="metric-pill pill-green">${pct(r.accuracy)}</span></td>
    <td>${fmt(r.f1_macro)}</td>
    <td>${pct(r.binary_accuracy)}</td>
    <td>${fmt(r.binary_f1)}</td>
    <td><span class="metric-pill pill-blue">${fmt(r.binary_roc_auc)}</span></td>
    <td><strong>${parseFloat(r.both_correct_pct||0).toFixed(2)}%</strong><span class="progress-bar"><span class="progress-fill fill-green" style="width:${r.both_correct_pct||0}%"></span></span></td>
    <td>${parseFloat(r.name_only_correct_pct||0).toFixed(2)}%</td>
    <td>${parseFloat(r.none_correct_pct||0).toFixed(2)}%</td>
  </tr>`).join('');

  addChip('checkpoint_ranking.csv',rows.length);
}

// ── Render plots ─────────────────────────────────────────────────────────────
function renderPlots(plots){
  const grid=document.getElementById('plotsGrid');
  if(!plots.length){grid.innerHTML='<div class="empty-state" style="grid-column:1/-1"><div class="icon">🖼️</div><p>No plot images found — run training first</p></div>';return;}
  grid.innerHTML=plots.map(p=>`
    <div class="image-card fade-in">
      <div class="image-card-header"><span>${p.icon}</span><span>${p.label}</span></div>
      <img src="${p.src}" alt="${p.label}"/>
    </div>`).join('');
  addChip(`${plots.length} plot image(s)`,null);
}

// ── Chip helper ───────────────────────────────────────────────────────────────
const addedChips=new Set();
function addChip(name,count){
  if(addedChips.has(name)) return;
  addedChips.add(name);
  const c=document.createElement('div');
  c.className='file-chip';
  c.innerHTML=`<span class="dot"></span> ${name}${count!=null?' <span style="opacity:.5">('+count+' rows)</span>':''}`;
  document.getElementById('loadedFiles').appendChild(c);
}

// ── Table helpers ─────────────────────────────────────────────────────────────
let sortState={};
function sortTable(id,col){
  const tb=document.querySelector('#'+id+' tbody');
  const rows=Array.from(tb.querySelectorAll('tr')).filter(r=>r.cells.length>1);
  const key=id+'_'+col;
  const asc=sortState[key]=!sortState[key];
  rows.sort((a,b)=>{
    const va=a.cells[col]?.textContent.trim()||'';
    const vb=b.cells[col]?.textContent.trim()||'';
    const na=parseFloat(va.replace(/%/g,'')),nb=parseFloat(vb.replace(/%/g,''));
    if(!isNaN(na)&&!isNaN(nb)) return asc?na-nb:nb-na;
    return asc?va.localeCompare(vb):vb.localeCompare(va);
  });
  rows.forEach(r=>tb.appendChild(r));
  document.querySelectorAll('#'+id+' th').forEach((th,i)=>th.classList.toggle('sorted',i===col));
}
function filterTable(id,q){
  document.querySelectorAll('#'+id+' tbody tr').forEach(r=>{
    r.style.display=r.textContent.toLowerCase().includes(q.toLowerCase())?'':'none';
  });
}

// ── Manual CSV Upload ─────────────────────────────────────────────────────────
function parseCSV(text){
  const lines=text.trim().split('\n').filter(l=>l.trim());
  const headers=lines[0].split(',').map(h=>h.trim());
  return lines.slice(1).map(line=>{
    const vals=[];let cur='',inQ=false;
    for(const ch of line){if(ch==='"'){inQ=!inQ;}else if(ch===','&&!inQ){vals.push(cur);cur='';}else{cur+=ch;}}
    vals.push(cur);
    const obj={};headers.forEach((h,i)=>obj[h]=(vals[i]||'').trim());
    return obj;
  });
}
function handleFiles(files){
  [...files].forEach(file=>{
    const reader=new FileReader();
    reader.onload=e=>{
      const rows=parseCSV(e.target.result);
      if(!rows.length) return;
      const keys=Object.keys(rows[0]);
      if(keys.includes('test_accuracy')) renderExperiments(rows);
      else if(keys.includes('binary_accuracy')) renderCheckpoints(rows);
    };
    reader.readAsText(file);
  });
}
document.getElementById('csvUpload').addEventListener('change',e=>handleFiles(e.target.files));
const dz=document.getElementById('dropZone');
dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('drag-over');});
dz.addEventListener('dragleave',()=>dz.classList.remove('drag-over'));
dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('drag-over');handleFiles(e.dataTransfer.files);});

// ── Auto-render embedded data on load ────────────────────────────────────────
renderExperiments(EXPERIMENTS);
renderCheckpoints(CHECKPOINTS);
renderPlots(PLOTS);
</script>
</body>
</html>
"""

HTML_OUT = HTML.replace('__DATA_JS__', data_js)

os.makedirs(RESULTS_DIR, exist_ok=True)
with open(OUT_HTML, 'w', encoding='utf-8') as f:
    f.write(HTML_OUT)

print(f"\nDashboard written to: {OUT_HTML}")
print("Open it in any browser — all data is self-contained, no server needed.")
